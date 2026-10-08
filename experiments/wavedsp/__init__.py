"""Dense bin-wise WaveDSP models, independent of the existing training pipeline."""

from .model import FastWaveDSP, FineWaveDSP, waveform_auxiliary_loss

__all__ = ["FastWaveDSP", "FineWaveDSP", "waveform_auxiliary_loss"]
