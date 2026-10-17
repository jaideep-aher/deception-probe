"""
Unit tests for RepresentationSteering and causal interventions.
"""

import numpy as np
from core.steering import RepresentationSteering
from core.dataset import ActivationSampler


def test_steering_vector_computation():
    hidden_dim = 64
    sampler = ActivationSampler(hidden_dim=hidden_dim, random_seed=42)

    truth_acts = np.array([sampler.sample_activation_pair(truthful=True) for _ in range(50)])
    decep_acts = np.array([sampler.sample_activation_pair(truthful=False) for _ in range(50)])

    steerer = RepresentationSteering(hidden_dim=hidden_dim)
    v_truth = steerer.compute_steering_vector_from_pairs(truth_acts, decep_acts, layer=18)

    assert v_truth.shape == (hidden_dim,)
    assert abs(np.linalg.norm(v_truth) - 1.0) < 1e-5

    # Test applying steering
    sample_decep = sampler.sample_activation_pair(truthful=False)
    steered = steerer.steer_activation(sample_decep, layer=18, alpha=2.0, preserve_norm=True)

    # Cosine similarity with v_truth should increase after steering
    cos_orig = np.dot(sample_decep, v_truth) / (np.linalg.norm(sample_decep) * np.linalg.norm(v_truth))
    cos_steered = np.dot(steered, v_truth) / (np.linalg.norm(steered) * np.linalg.norm(v_truth))

    assert cos_steered > cos_orig


def test_orthogonal_concept_erasure():
    hidden_dim = 64
    steerer = RepresentationSteering(hidden_dim=hidden_dim)
    concept = np.random.randn(hidden_dim)
    concept /= np.linalg.norm(concept)

    h = np.random.randn(hidden_dim)
    h_scrubbed = steerer.orthogonal_concept_erasure(h, concept)

    # Orthogonality check: dot product with concept direction should be zero
    dot = np.dot(h_scrubbed, concept)
    assert abs(dot) < 1e-7
