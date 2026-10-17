"""
Unit tests for Contrast Consistent Search (CCS).
"""

import numpy as np
from core.probe import ContrastConsistentSearch
from core.dataset import ActivationSampler


def test_ccs_optimization():
    hidden_dim = 128
    sampler = ActivationSampler(hidden_dim=hidden_dim, random_seed=42)

    # Generate contrastive pairs (x+, x-)
    n_pairs = 80
    act_pos = np.array([sampler.sample_activation_pair(truthful=True) for _ in range(n_pairs)])
    act_neg = np.array([sampler.sample_activation_pair(truthful=False) for _ in range(n_pairs)])

    ccs = ContrastConsistentSearch(hidden_dim=hidden_dim)
    loss_dict = ccs.fit(act_pos, act_neg, epochs=120, lr=0.02)

    assert "total_loss" in loss_dict
    assert loss_dict["loss_consistency"] < 0.35

    # Check truth prediction on positive vs negative
    p_pos = ccs.predict_truth_score(act_pos[0])
    p_neg = ccs.predict_truth_score(act_neg[0])

    # Consistency implies p_pos and p_neg should be on opposite sides of 0.5
    assert abs(p_pos - (1.0 - p_neg)) < 0.40
