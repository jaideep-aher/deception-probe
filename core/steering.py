"""
DeceptionProbe: Representation Steering & Causal Intervention
Implements additive activation steering and closed-form orthogonal projection (LEACE style).
Grounded in Zou et al. (2023) Representation Engineering and Belrose et al. (2023) LEACE.
"""

from typing import Dict, Optional, Tuple
import numpy as np


class RepresentationSteering:
    """
    Computes and applies causal steering vectors to transformer hidden states.
    Allows real-time human-in-the-loop truthfulness modulation:
        h' = h + alpha * v_truth
    """

    def __init__(self, hidden_dim: int, target_layers: Tuple[int, ...] = (16, 17, 18, 19, 20)):
        self.hidden_dim = hidden_dim
        self.target_layers = target_layers
        self.steering_vectors: Dict[int, np.ndarray] = {}  # layer -> vector
        self.layer_norms: Dict[int, float] = {}

    def compute_steering_vector_from_pairs(
        self,
        truth_activations: np.ndarray,
        deceptive_activations: np.ndarray,
        layer: int = 18,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Computes the reading/steering direction as the difference in means:
            v_truth = Mean(H_truth) - Mean(H_deceptive)
        """
        assert truth_activations.shape[1] == self.hidden_dim
        assert deceptive_activations.shape[1] == self.hidden_dim

        mu_truth = np.mean(truth_activations, axis=0)
        mu_decep = np.mean(deceptive_activations, axis=0)

        diff = mu_truth - mu_decep
        norm = np.linalg.norm(diff)

        if normalize and norm > 1e-8:
            direction = diff / norm
        else:
            direction = diff

        self.steering_vectors[layer] = direction
        self.layer_norms[layer] = float(norm)
        return direction

    def steer_activation(
        self,
        hidden_state: np.ndarray,
        layer: int,
        alpha: float = 1.0,
        preserve_norm: bool = True
    ) -> np.ndarray:
        """
        Applies additive causal intervention to a hidden state vector.
        alpha > 0 promotes truthfulness; alpha < 0 exacerbates deception/sycophancy.
        """
        if layer not in self.steering_vectors:
            return hidden_state.copy()

        v = self.steering_vectors[layer]
        original_norm = np.linalg.norm(hidden_state)

        # Causal intervention: h' = h + alpha * v
        steered = hidden_state + (alpha * v)

        if preserve_norm and original_norm > 1e-8:
            steered_norm = np.linalg.norm(steered)
            if steered_norm > 1e-8:
                steered = steered * (original_norm / steered_norm)

        return steered

    def orthogonal_concept_erasure(
        self,
        hidden_state: np.ndarray,
        concept_direction: np.ndarray
    ) -> np.ndarray:
        """
        Removes the concept subspace completely via orthogonal projection:
            h_clean = h - (h . v_norm) * v_norm
        """
        norm = np.linalg.norm(concept_direction)
        if norm < 1e-8:
            return hidden_state.copy()
        u = concept_direction / norm
        proj = np.dot(hidden_state, u) * u
        return hidden_state - proj
