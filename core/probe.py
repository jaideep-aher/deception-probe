"""
DeceptionProbe: Core Probe & Representation Engineering Engine
Implements Linear Probes, Contrast Consistent Search (CCS), and Activation Reading.
Grounded in Zou et al. (2023) and Burns et al. (2022).
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class ProbePrediction:
    token: str
    deception_score: float  # [0.0 (honest) to 1.0 (deceptive)]
    truth_score: float      # 1.0 - deception_score
    uncertainty: float      # epistemic entropy / variance
    layer_contributions: Dict[int, float] = field(default_factory=dict)
    is_anomaly: bool = False


class LinearProbe:
    """
    Supervised Linear Probe operating on Transformer Residual Stream activations.
    Computes a hyperplane separating truthful vs deceptive internal states:
        P(Truth | h) = sigma(w^T h + b)
    """

    def __init__(self, hidden_dim: int, target_layer: int = 18):
        self.hidden_dim = hidden_dim
        self.target_layer = target_layer
        self.weights: Optional[np.ndarray] = None  # shape: (hidden_dim,)
        self.bias: float = 0.0
        self.calibrated_temperature: float = 1.0
        self.is_fitted: bool = False

    def fit(self, activations: np.ndarray, labels: np.ndarray, l2_penalty: float = 1e-3, epochs: int = 100, lr: float = 0.05) -> Dict[str, float]:
        """
        Train linear probe via regularized logistic regression on hidden states.
        activations: shape (N, hidden_dim)
        labels: binary (1 = truthful, 0 = deceptive)
        """
        N, D = activations.shape
        assert D == self.hidden_dim, f"Expected hidden_dim {self.hidden_dim}, got {D}"

        # Initialize weights with Xavier scaling
        rng = np.random.RandomState(42)
        w = rng.randn(D) * np.sqrt(2.0 / D)
        b = 0.0

        for epoch in range(epochs):
            # Forward pass
            logits = np.dot(activations, w) + b
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

            # Gradient calculation with L2 regularization
            error = probs - labels
            grad_w = (np.dot(activations.T, error) / N) + (l2_penalty * w)
            grad_b = np.mean(error)

            # Gradient descent update
            w -= lr * grad_w
            b -= lr * grad_b

        self.weights = w
        self.bias = float(b)
        self.is_fitted = True

        # Compute training loss and accuracy
        final_logits = np.dot(activations, self.weights) + self.bias
        final_probs = 1.0 / (1.0 + np.exp(-np.clip(final_logits, -30.0, 30.0)))
        preds = (final_probs >= 0.5).astype(int)
        acc = float(np.mean(preds == labels))
        bce_loss = float(-np.mean(labels * np.log(final_probs + 1e-12) + (1 - labels) * np.log(1 - final_probs + 1e-12)))

        return {"accuracy": acc, "bce_loss": bce_loss}

    def predict_proba(self, activation: np.ndarray) -> float:
        """
        Returns P(Truthful | activation).
        """
        if not self.is_fitted or self.weights is None:
            raise RuntimeError("Probe has not been fitted yet.")
        logit = (np.dot(activation, self.weights) + self.bias) / self.calibrated_temperature
        prob_truth = float(1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0))))
        return prob_truth

    def evaluate_token(self, token: str, activation: np.ndarray, baseline_mean: Optional[np.ndarray] = None) -> ProbePrediction:
        """
        Evaluates a single token's residual state, producing a ProbePrediction.
        """
        prob_truth = self.predict_proba(activation)
        prob_deception = 1.0 - prob_truth

        # Entropy as epistemic uncertainty
        p = np.clip(prob_truth, 1e-6, 1.0 - 1e-6)
        uncertainty = float(- (p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p)))

        # Anomaly detection via Mahalanobis distance if baseline provided
        is_anomaly = False
        if baseline_mean is not None:
            dist = float(np.linalg.norm(activation - baseline_mean))
            is_anomaly = dist > 2.5 * np.sqrt(self.hidden_dim)

        return ProbePrediction(
            token=token,
            deception_score=prob_deception,
            truth_score=prob_truth,
            uncertainty=uncertainty,
            is_anomaly=is_anomaly
        )


class ContrastConsistentSearch:
    """
    Contrast Consistent Search (CCS) (Burns et al., 2022).
    Discovers latent truthfulness in an unsupervised manner using contrastive pairs (x+, x-).
    Enforces consistency: P(x+) + P(x-) = 1
    Enforces informativeness: min P(x+)^2 + P(x-)^2
    """

    def __init__(self, hidden_dim: int):
        self.hidden_dim = hidden_dim
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0

    def fit(self, act_pos: np.ndarray, act_neg: np.ndarray, epochs: int = 150, lr: float = 0.01) -> Dict[str, float]:
        """
        act_pos: activations for affirmative statements (N, hidden_dim)
        act_neg: activations for negated statements (N, hidden_dim)
        """
        N, D = act_pos.shape
        rng = np.random.RandomState(42)
        w = rng.randn(D) * np.sqrt(2.0 / D)
        b = 0.0

        for _ in range(epochs):
            # p_pos = P(True | x+), p_neg = P(True | x-)
            z_pos = np.dot(act_pos, w) + b
            z_neg = np.dot(act_neg, w) + b
            p_pos = 1.0 / (1.0 + np.exp(-np.clip(z_pos, -25.0, 25.0)))
            p_neg = 1.0 / (1.0 + np.exp(-np.clip(z_neg, -25.0, 25.0)))

            # Loss: Consistency Loss + Informativeness Loss
            # Consistency: (p_pos + p_neg - 1)^2
            # Informativeness: min(p_pos, p_neg)^2
            err_cons = (p_pos + p_neg - 1.0)
            grad_cons_p_pos = 2.0 * err_cons
            grad_cons_p_neg = 2.0 * err_cons

            min_mask_pos = (p_pos < p_neg).astype(float)
            min_mask_neg = (p_neg <= p_pos).astype(float)
            grad_info_p_pos = 2.0 * p_pos * min_mask_pos
            grad_info_p_neg = 2.0 * p_neg * min_mask_neg

            grad_p_pos = grad_cons_p_pos + grad_info_p_pos
            grad_p_neg = grad_cons_p_neg + grad_info_p_neg

            # Chain rule through sigmoid: dp/dz = p * (1 - p)
            dz_pos = grad_p_pos * p_pos * (1.0 - p_pos)
            dz_neg = grad_p_neg * p_neg * (1.0 - p_neg)

            grad_w = (np.dot(act_pos.T, dz_pos) + np.dot(act_neg.T, dz_neg)) / N
            grad_b = np.mean(dz_pos + dz_neg)

            w -= lr * grad_w
            b -= lr * grad_b

        self.weights = w
        self.bias = float(b)

        # Final loss evaluation
        z_pos = np.dot(act_pos, w) + b
        z_neg = np.dot(act_neg, w) + b
        p_pos = 1.0 / (1.0 + np.exp(-np.clip(z_pos, -25.0, 25.0)))
        p_neg = 1.0 / (1.0 + np.exp(-np.clip(z_neg, -25.0, 25.0)))
        loss_cons = float(np.mean((p_pos + p_neg - 1.0) ** 2))
        loss_info = float(np.mean(np.minimum(p_pos, p_neg) ** 2))

        return {"loss_consistency": loss_cons, "loss_informativeness": loss_info, "total_loss": loss_cons + loss_info}

    def predict_truth_score(self, activation: np.ndarray) -> float:
        if self.weights is None:
            raise RuntimeError("CCS probe has not been fitted.")
        logit = np.dot(activation, self.weights) + self.bias
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -25.0, 25.0))))
