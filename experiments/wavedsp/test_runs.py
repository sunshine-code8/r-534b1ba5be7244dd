"""CPU-only checks for isolated checkpoint directories."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from experiments.wavedsp.train import create_run_dir, save_checkpoint_atomic


class RunDirectoryTests(unittest.TestCase):
    def test_each_launch_gets_new_directory_and_resume_copies_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / "fine_ghost"
            first = create_run_dir(base, None, rank=0)
            self.assertEqual(first.parent, base)
            state = {"epoch": 8, "best_macro_f1": 0.23, "model": {"w": torch.tensor([1.0])}}
            save_checkpoint_atomic(state, first / "last.pt")
            save_checkpoint_atomic(state, first / "best.pt")
            second = create_run_dir(base, first / "last.pt", rank=0)
            self.assertNotEqual(first, second)
            self.assertEqual(torch.load(second / "last.pt", weights_only=False)["epoch"], 8)
            self.assertEqual(torch.load(second / "best.pt", weights_only=False)["epoch"], 8)
            self.assertEqual(torch.load(first / "last.pt", weights_only=False)["epoch"], 8)
            info = json.loads((second / "run_info.json").read_text())
            self.assertEqual(info["resumed_from"], str((first / "last.pt").resolve()))


if __name__ == "__main__":
    unittest.main()
