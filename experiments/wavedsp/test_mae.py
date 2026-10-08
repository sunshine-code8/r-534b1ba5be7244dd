"""CPU checks for the independent WaveDSP reconstruction pretraining path."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import blosc2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from experiments.wavedsp.mae_data import MAECachedDataset, MAERawDataset, build_mae_cache
from experiments.wavedsp.mae_model import (
    ReconstructionWaveDSP,
    fixed_spatial_masks,
    masked_mse_parts,
    spatial_mask,
)
from experiments.wavedsp.model import ARCHITECTURE_VERSION, FastWaveDSP, FineWaveDSP
from experiments.wavedsp.pretrain import (
    EvaluationSampler,
    check_resume,
    dataset_identity,
    make_datasets,
    make_loader,
    run_epoch,
    validate_config,
)
from experiments.wavedsp.train import save_checkpoint_atomic


def fixture(root: Path) -> tuple[dict, dict, Path, Path]:
    source, cache = root / "mae_dataset", root / "cache"
    data = {
        "y_crop_top": 2,
        "y_crop_bottom": 1,
        "z_crop_front": 3,
        "z_crop_back": 2,
        "downsample_z": 32,
    }
    for split in ("train", "valid"):
        directories = []
        for scene in ("one", "two"):
            directory = source / split / scene
            directory.mkdir(parents=True)
            # Same name in different sequences; no peak/annotation files exist.
            array = np.arange(9 * 14 * 40, dtype=np.uint16).reshape(9, 14, 40)
            blosc2.save_array(array, directory / "frame_voxel.b2")
            directories.append(str(directory))
        data[f"{split}_voxel_dirs"] = directories
    data_path = root / "data.yaml"
    data_path.write_text(yaml.safe_dump(data))
    config = {
        "model": "fast",
        "model_kwargs": {},
        "data_config": str(data_path),
        "data_root": str(source),
        "cache_root": str(cache),
        "target_size": [9, 11, 32],
        "loss": "masked_mse",
        "masking": {"block_size_yx": [4, 4], "ratio": 0.7, "value": 0.0, "validation_seed": 123},
        "training": {
            "seed": 42,
            "epochs": 2,
            "batch_size_per_gpu": 1,
            "effective_batch_size": 2,
            "lr": 1e-4,
            "weight_decay": 0.01,
            "scheduler": "none",
            "gradient_clip": 1.0,
            "bf16": False,
            "num_workers": 0,
            "prefetch_factor": 1,
        },
    }
    return data, config, source, cache


class MAEDataTests(unittest.TestCase):
    def test_lossless_cache_resume_split_keys_and_no_peaks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, config, source, cache = fixture(root)
            link = root / "cache_link"
            link.symlink_to(cache, target_is_directory=True)
            size = tuple(config["target_size"])
            meta = build_mae_cache(data, size, source, link, workers=2)
            self.assertEqual(meta["split_counts"], {"train": 2, "valid": 2})
            for split in ("train", "valid"):
                raw = MAERawDataset(data, split, size, source)
                cached = MAECachedDataset(link, split, data, size, source)
                self.assertNotEqual(raw[0]["sample_id"], raw[1]["sample_id"])
                for index in range(2):
                    a, b = raw[index], cached[index]
                    self.assertEqual(b["waveform"].dtype, torch.uint16)
                    self.assertTrue(b["waveform"].is_contiguous())
                    torch.testing.assert_close(a["waveform"], b["waveform"])
                    full = blosc2.load_array(source / raw.entries[index]["voxel"])
                    expected = full[:, 1:12, 3:38][:, :, np.linspace(0, 34, 32, dtype=int)]
                    np.testing.assert_array_equal(b["waveform"].numpy(), expected)
            raw_sets = make_datasets(config, raw_data=True)
            self.assertEqual(dataset_identity(raw_sets), dataset_identity(make_datasets(config)))
            preserved = cache / raw_sets["train"].entries[0]["voxel"]
            timestamp = preserved.stat().st_mtime_ns
            self.assertEqual(meta, build_mae_cache(data, size, source, link))
            self.assertEqual(preserved.stat().st_mtime_ns, timestamp)
            (cache / "meta.json").unlink()
            (cache / raw_sets["valid"].entries[0]["voxel"]).unlink()
            with self.assertRaises(FileNotFoundError):
                MAECachedDataset(cache, "train", data, size, source)
            build_mae_cache(data, size, source, cache)
            self.assertEqual(preserved.stat().st_mtime_ns, timestamp)

    def test_cache_rejects_config_overlap_and_index_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data, config, source, cache = fixture(Path(temporary))
            size = tuple(config["target_size"])
            build_mae_cache(data, size, source, cache)
            changed = dict(data, z_crop_front=4)
            with self.assertRaisesRegex(ValueError, "changed"):
                build_mae_cache(changed, size, source, cache)
            with self.assertRaisesRegex(ValueError, "preprocessing"):
                MAECachedDataset(cache, "train", changed, size, source)
            overlap = dict(data, valid_voxel_dirs=data["train_voxel_dirs"])
            with self.assertRaisesRegex(ValueError, "overlap"):
                build_mae_cache(overlap, size, source, cache.parent / "other_cache")
            with self.assertRaisesRegex(ValueError, "split directories"):
                MAECachedDataset(cache, "valid", overlap, size, source)
            index = cache / "train_index.json"
            entries = json.loads(index.read_text())
            entries[0]["sample_id"] = "changed"
            index.write_text(json.dumps(entries))
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                MAECachedDataset(cache, "train", data, size, source)


class MAEModelTests(unittest.TestCase):
    def test_both_backbones_receive_gradients_and_transfer(self) -> None:
        for kind, cls in (("fast", FastWaveDSP), ("fine", FineWaveDSP)):
            with self.subTest(kind=kind):
                model = ReconstructionWaveDSP(kind, 32)
                target = torch.randn(2, 11, 9, 32, 1)
                mask = spatial_mask(2, 11, 9, (4, 4), 0.7, torch.device("cpu"))
                prediction = model(target, mask)
                self.assertEqual(prediction.shape, target.shape)
                num, den = masked_mse_parts(prediction, target, mask)
                (num / den).backward()
                self.assertFalse(
                    any(name.startswith("network.head.") for name, _ in model.named_parameters())
                )
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                # Changes to hidden target bins cannot leak through skip connections.
                torch.testing.assert_close(model(target.masked_fill(mask, 1234), mask), prediction)
                ordinary = cls(time_bins=32)
                missing, unexpected = ordinary.load_state_dict(
                    model.backbone_state_dict(), strict=False
                )
                self.assertEqual(set(missing), {"head.project.weight", "head.project.bias"})
                self.assertFalse(unexpected)
                full = target.masked_fill(mask, 0)
                torch.testing.assert_close(
                    ordinary.forward_features(full), model.network.forward_features(full)
                )
                self.assertEqual(ordinary(target).shape, (2, 11, 9, 32, 4))

    def test_mask_edges_fixed_ids_and_masked_loss_only(self) -> None:
        device = torch.device("cpu")
        ids = ["sequence/a", "sequence/b"]
        masks = fixed_spatial_masks(ids, 11, 9, (4, 4), 0.7, 42, device)
        reversed_masks = fixed_spatial_masks(ids[::-1], 11, 9, (4, 4), 0.7, 42, device)
        torch.testing.assert_close(masks, reversed_masks.flip(0))
        torch.testing.assert_close(
            masks[:1], fixed_spatial_masks(ids[:1], 11, 9, (4, 4), 0.7, 42, device)
        )
        self.assertEqual(masks.shape, (2, 11, 9, 1, 1))
        self.assertTrue(masks.any() and (~masks).any())
        prediction = torch.ones(2, 11, 9, 32, 1, requires_grad=True)
        target = torch.zeros_like(prediction)
        num, den = masked_mse_parts(prediction, target, masks)
        torch.testing.assert_close(num / den, torch.tensor(1.0))
        self.assertEqual(den.item(), masks.sum().item() * 32)
        (num / den).backward()
        self.assertEqual(
            prediction.grad.masked_select(~masks.expand_as(prediction)).count_nonzero(), 0
        )
        self.assertTrue((prediction.grad.masked_select(masks.expand_as(prediction)) > 0).all())
        with self.assertRaises(ValueError):
            spatial_mask(1, 2, 2, (4, 4), 0.7, device)

    def test_spatial_axis_mapping_matches_supervised(self) -> None:
        from experiments.wavedsp.train import SupervisedWaveDSP

        raw = torch.randn(2, 9, 11, 32)  # XYT
        supervised = SupervisedWaveDSP("fast", 32)
        pretrain = ReconstructionWaveDSP("fast", 32)
        missing, unexpected = pretrain.network.load_state_dict(
            supervised.network.state_dict(), strict=False
        )
        self.assertFalse(missing)
        self.assertEqual(set(unexpected), {"head.project.weight", "head.project.bias"})
        expected = supervised.network.forward_features(raw.permute(0, 2, 1, 3).unsqueeze(-1))
        seen = []
        handle = pretrain.reconstruction_head.register_forward_pre_hook(
            lambda module, inputs: seen.append(inputs[0])
        )
        pretrain(
            raw.permute(0, 2, 1, 3).unsqueeze(-1), torch.zeros(2, 11, 9, 1, 1, dtype=torch.bool)
        )
        handle.remove()
        torch.testing.assert_close(seen[0], expected)


class MAETrainingTests(unittest.TestCase):
    def test_train_valid_checkpoint_resume_and_accumulation_tail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, config, _, _ = fixture(Path(temporary))
            datasets = make_datasets(config, raw_data=True)
            device = torch.device("cpu")
            model = ReconstructionWaveDSP("fast", 32)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            samples = [datasets["train"][0], datasets["train"][1], datasets["train"][0]]
            loader = DataLoader(samples, batch_size=1)
            before = model.network.up0.project.weight.detach().clone()
            updates = []
            handle = optimizer.register_step_post_hook(lambda *args, **kwargs: updates.append(1))
            result = run_epoch(
                model, loader, config, device, 0, optimizer=optimizer, accumulation=2
            )
            handle.remove()
            self.assertEqual(len(updates), 2)
            self.assertEqual(result["frames"], 3)
            self.assertFalse(torch.equal(before, model.network.up0.project.weight))
            snapshot = copy.deepcopy(model.state_dict())
            a = run_epoch(model, DataLoader(datasets["valid"], batch_size=1), config, device, 0)
            b = run_epoch(model, DataLoader(datasets["valid"], batch_size=2), config, device, 4)
            self.assertAlmostEqual(
                a["masked_mse"], b["masked_mse"], delta=max(1, a["masked_mse"]) * 1e-5
            )
            for key, value in snapshot.items():
                torch.testing.assert_close(value, model.state_dict()[key])
            identity = dataset_identity(datasets)
            state = {
                "stage": "wavedsp_mae_reconstruction",
                "architecture_version": ARCHITECTURE_VERSION,
                "config": config,
                "model": model.state_dict(),
                "backbone": model.backbone_state_dict(),
                "optimizer": optimizer.state_dict(),
                "data_identity": identity,
                "world_size": 1,
            }
            path = Path(temporary) / "last.pt"
            save_checkpoint_atomic(state, path)
            restored = torch.load(path, weights_only=True)
            check_resume(restored, config, identity, 1)
            other = ReconstructionWaveDSP("fast", 32)
            other.load_state_dict(restored["model"], strict=True)
            with self.assertRaisesRegex(ValueError, "model differs"):
                check_resume(restored, dict(config, model="fine"), identity, 1)
            with self.assertRaisesRegex(ValueError, "dataset"):
                check_resume(restored, config, {}, 1)

    def test_validation_shards_and_loader_shuffle_resume(self) -> None:
        samples = list(range(5))
        shards = [list(EvaluationSampler(samples, rank, 3)) for rank in range(3)]
        self.assertEqual(sorted(sum(shards, [])), samples)
        with tempfile.TemporaryDirectory() as temporary:
            _, config, _, _ = fixture(Path(temporary))
            self.assertEqual(validate_config(config, 1), 2)
            self.assertEqual(validate_config(config, 2), 1)
            with self.assertRaises(ValueError):
                validate_config(config, 3)
            loader, _ = make_loader(samples, config["training"], True, 0, 1)
            resumed, _ = make_loader(samples, config["training"], True, 0, 1)
            loader.sampler.generator.manual_seed(123)
            resumed.sampler.generator.manual_seed(123)
            torch.rand(10, generator=loader.generator)
            self.assertEqual([int(x) for x in loader], [int(x) for x in resumed])


def distributed_worker(rank: int, rendezvous: str, result_path: str, config: dict) -> None:
    import datetime

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    torch.set_num_threads(1)
    torch.manual_seed(123)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=45),
    )
    try:
        model = ReconstructionWaveDSP("fast", 32)
        wrapped = DistributedDataParallel(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        samples = [
            {"waveform": torch.randn(9, 11, 32), "sample_id": f"frame/{i}"} for i in range(5)
        ]
        loader, sampler = make_loader(samples, config["training"], True, rank, 2)
        sampler.set_epoch(0)
        run_epoch(wrapped, loader, config, torch.device("cpu"), 0, rank, optimizer, accumulation=2)
        # Only rank 0 has a validation sample: tests empty/uneven validation shards.
        valid_loader, _ = make_loader(samples[:1], config["training"], False, rank, 2)
        result = run_epoch(model, valid_loader, config, torch.device("cpu"), 0, rank)
        if rank == 0:
            torch.save(
                {"model": model.state_dict(), "sample": samples[0], "result": result}, result_path
            )
    finally:
        dist.destroy_process_group()


class MAEDistributedTests(unittest.TestCase):
    def test_ddp_training_and_unpadded_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, config, _, _ = fixture(root)
            result_path = root / "distributed.pt"
            torch.multiprocessing.spawn(
                distributed_worker,
                args=(str(root / "rendezvous"), str(result_path), config),
                nprocs=2,
                join=True,
            )
            result = torch.load(result_path, weights_only=True)
            model = ReconstructionWaveDSP("fast", 32)
            model.load_state_dict(result["model"])
            expected = run_epoch(
                model, DataLoader([result["sample"]]), config, torch.device("cpu"), 0
            )
            self.assertEqual(result["result"]["frames"], 1)
            self.assertAlmostEqual(result["result"]["masked_mse"], expected["masked_mse"], places=6)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
