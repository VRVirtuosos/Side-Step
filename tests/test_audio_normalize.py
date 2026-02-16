"""Tests for audio normalization (peak and LUFS)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch


class TestPeakNormalize(unittest.TestCase):
    """Verify peak normalization to -1.0 dBFS."""

    def test_silence_passthrough(self):
        """Near-silence (peak < 1e-6) should be returned unchanged."""
        from acestep.training_v2.audio_normalize import peak_normalize

        audio = torch.zeros(2, 48000)
        result = peak_normalize(audio, target_db=-1.0)
        self.assertTrue(torch.equal(audio, result))

    def test_peak_scales_correctly(self):
        """Peak after normalization should match target dB."""
        from acestep.training_v2.audio_normalize import peak_normalize

        audio = torch.randn(2, 48000) * 0.3
        result = peak_normalize(audio, target_db=-1.0)

        target_amp = 10 ** (-1.0 / 20.0)
        actual_peak = torch.max(torch.abs(result)).item()
        self.assertAlmostEqual(actual_peak, target_amp, places=5)

    def test_shape_preserved(self):
        """Output shape must match input."""
        from acestep.training_v2.audio_normalize import peak_normalize

        audio = torch.randn(2, 96000)
        result = peak_normalize(audio)
        self.assertEqual(result.shape, audio.shape)


class TestLufsNormalize(unittest.TestCase):
    """Verify LUFS normalization and its pyloudnorm fallback."""

    def test_fallback_to_peak_when_no_pyloudnorm(self):
        """When pyloudnorm is not installed, should fall back to peak."""
        from acestep.training_v2.audio_normalize import lufs_normalize

        audio = torch.randn(2, 48000) * 0.5

        with patch.dict("sys.modules", {"pyloudnorm": None}):
            result = lufs_normalize(audio, sample_rate=48000)

        # Should have run peak normalize -- peak should be near -1.0 dBFS
        target_amp = 10 ** (-1.0 / 20.0)
        actual_peak = torch.max(torch.abs(result)).item()
        self.assertAlmostEqual(actual_peak, target_amp, places=4)

    def test_silence_passthrough(self):
        """LUFS of silence is -inf, should be returned unchanged."""
        from acestep.training_v2.audio_normalize import lufs_normalize

        audio = torch.zeros(2, 48000)
        try:
            import pyloudnorm  # noqa: F401
            result = lufs_normalize(audio, sample_rate=48000)
            self.assertTrue(torch.equal(audio, result))
        except ImportError:
            self.skipTest("pyloudnorm not installed")


class TestNormalizeDispatch(unittest.TestCase):
    """Verify the normalize_audio dispatcher."""

    def test_none_is_passthrough(self):
        """method='none' returns input unchanged."""
        from acestep.training_v2.audio_normalize import normalize_audio

        audio = torch.randn(2, 48000)
        result = normalize_audio(audio, 48000, method="none")
        self.assertTrue(torch.equal(audio, result))

    def test_peak_dispatches(self):
        """method='peak' calls peak_normalize."""
        from acestep.training_v2.audio_normalize import normalize_audio

        audio = torch.randn(2, 48000) * 0.3
        result = normalize_audio(audio, 48000, method="peak")
        target_amp = 10 ** (-1.0 / 20.0)
        actual_peak = torch.max(torch.abs(result)).item()
        self.assertAlmostEqual(actual_peak, target_amp, places=5)

    def test_unknown_method_passthrough(self):
        """Unknown method name should return input unchanged."""
        from acestep.training_v2.audio_normalize import normalize_audio

        audio = torch.randn(2, 48000)
        result = normalize_audio(audio, 48000, method="unknown_method")
        self.assertTrue(torch.equal(audio, result))


if __name__ == "__main__":
    unittest.main()
