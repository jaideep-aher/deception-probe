"""
Unit tests for LinearProbe and ProbePrediction.
"""

import numpy as np
import pytest
from core.probe import LinearProbe
from core.dataset import ActivationSampler


def test_linear_probe_fit_and_predict():
    hidden_dim = 128
    sampler = ActivationSampler(hidden_dim=hidden_dim, random_seed=42)
    X, y = sampler.generate_benchmark_dataset(n_samples=100)

    probe = LinearProbe(hidden_dim=hidden_dim, target_layer=18)
    metrics = probe.fit(X, y, epochs=100, lr=0.08)

    assert metrics["accuracy"] > 0.80
    assert probe.is_fitted is True

    # Test single token evaluation
    sample_true = sampler.sample_activation_pair(truthful=True)
    sample_false = sampler.sample_activation_pair(truthful=False)

    p_true = probe.predict_proba(sample_true)
    p_false = probe.predict_proba(sample_false)

    assert p_true > 0.60
    assert p_false < 0.40

    pred_true = probe.evaluate_token("honest_token", sample_true)
    assert pred_true.truth_score > pred_true.deception_score
    assert 0.0 <= pred_true.uncertainty <= 1.0


def test_unfitted_probe_raises_error():
    probe = LinearProbe(hidden_dim=64)
    with pytest.raises(RuntimeError):
        probe.predict_proba(np.zeros(64))
