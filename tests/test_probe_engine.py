"""
Unit Tests for DeceptionProbe Engine and Contrastive Dataset
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))
from contrastive_dataset import get_all_pairs, FACTUAL_PAIRS, SYCOPHANCY_PAIRS
from probe_engine import DeceptionProbeEngine

class TestDeceptionProbe(unittest.TestCase):
    def setUp(self):
        self.engine = DeceptionProbeEngine()

    def test_contrastive_dataset_loading(self):
        pairs = get_all_pairs()
        self.assertGreater(len(pairs), 10)
        self.assertGreater(len(FACTUAL_PAIRS), 0)
        self.assertGreater(len(SYCOPHANCY_PAIRS), 0)

    def test_probe_evaluation_truthful(self):
        telemetry = self.engine.evaluate_token_activation(
            token_str="Canberra",
            token_idx=0,
            is_ground_truth_deceptive=False
        )
        self.assertGreater(telemetry.honesty_score, 0.70)
        self.assertLess(telemetry.deception_prob, 0.30)
        self.assertFalse(telemetry.is_spiking)

    def test_probe_evaluation_deceptive(self):
        telemetry = self.engine.evaluate_token_activation(
            token_str="absolutely",
            token_idx=0,
            is_ground_truth_deceptive=True
        )
        self.assertGreater(telemetry.deception_prob, 0.50)
        self.assertTrue(telemetry.is_spiking)

    def test_truth_steering_suppresses_deception(self):
        # Without steering
        self.engine.set_steering_alpha(0.0)
        unsteered = self.engine.evaluate_token_activation("yes", 0, is_ground_truth_deceptive=True)

        # With strong positive steering
        self.engine.set_steering_alpha(3.0)
        steered = self.engine.evaluate_token_activation("yes", 0, is_ground_truth_deceptive=True)

        self.assertLess(steered.deception_prob, unsteered.deception_prob)
        self.assertGreater(steered.honesty_score, unsteered.honesty_score)

    def test_streaming_generation(self):
        tokens = list(self.engine.stream_generation("Prompt", delay_ms=0))
        self.assertGreater(len(tokens), 5)
        self.assertIsNotNone(tokens[0].polygraph_waveform_y)

if __name__ == "__main__":
    unittest.main()
