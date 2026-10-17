"""
Unit tests for ProbeEvaluator and metrics computation.
"""

import numpy as np
from core.evaluator import ProbeEvaluator
from core.dataset import ActivationSampler


def test_evaluator_metrics():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_probs = np.array([0.95, 0.88, 0.75, 0.60, 0.40, 0.25, 0.15, 0.05])

    metrics = ProbeEvaluator.compute_metrics(y_true, y_probs)

    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert 0.0 <= metrics["ece"] <= 0.25


def test_layer_sweep_ablation():
    sampler = ActivationSampler(hidden_dim=64, random_seed=42)
    layers = [10, 18, 26]

    sweep_results = ProbeEvaluator.layer_sweep_ablation(layers, sampler, n_samples=60)

    assert len(sweep_results) == 3
    # Mid-layer (18) should exhibit strong ROC-AUC
    res_18 = next(r for r in sweep_results if r["layer"] == 18)
    assert res_18["roc_auc"] >= 0.70
