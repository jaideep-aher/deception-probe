#!/usr/bin/env python3
"""
DeceptionProbe: Automated End-to-End Benchmark Runner
Trains linear probes, runs layer-depth ablation, computes ROC-AUC, PR-AUC, ECE,
and exports publication-quality benchmark results to JSON.
"""

import json
import os
import sys
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.probe import LinearProbe, ContrastConsistentSearch
from core.steering import RepresentationSteering
from core.dataset import ActivationSampler, TRUTHFULQA_BENCHMARK_PAIRS, SYCOPHANCY_BENCHMARK_PAIRS
from core.evaluator import ProbeEvaluator


def main():
    print("=" * 70)
    print(" DECEPTIONPROBE: END-TO-END BENCHMARK & LAYER ABLATION SUITE ")
    print("=" * 70)

    hidden_dim = 4096
    target_layer = 18
    sampler = ActivationSampler(hidden_dim=hidden_dim, truth_direction_layer=target_layer, random_seed=42)

    # 1. Dataset Generation
    print(f"\n[1/4] Generating contrastive evaluation dataset (D={hidden_dim}, Layer={target_layer})...")
    X_train, y_train = sampler.generate_benchmark_dataset(n_samples=250)
    X_test, y_test = sampler.generate_benchmark_dataset(n_samples=250)
    print(f"      Train split: {X_train.shape[0]} samples | Test split: {X_test.shape[0]} samples")

    # 2. Train Supervised Linear Probe
    print(f"\n[2/4] Training Supervised Linear Probe on Layer {target_layer}...")
    probe = LinearProbe(hidden_dim=hidden_dim, target_layer=target_layer)
    train_metrics = probe.fit(X_train, y_train, epochs=120, lr=0.08)
    print(f"      Train Convergence: Acc = {train_metrics['accuracy']:.4f}, Loss = {train_metrics['bce_loss']:.4f}")

    # Evaluate on Test Split
    test_probs = np.array([probe.predict_proba(x) for x in X_test])
    test_metrics = ProbeEvaluator.compute_metrics(y_test, test_probs)
    print("      Test Performance Metrics:")
    for k, v in test_metrics.items():
        print(f"        - {k.upper()}: {v}")

    # 3. Layer Depth Ablation Sweep
    layers_to_sweep = [4, 8, 12, 16, 18, 20, 24, 28, 31]
    print(f"\n[3/4] Running Layer-Depth Ablation Sweep across layers: {layers_to_sweep}...")
    sweep_results = ProbeEvaluator.layer_sweep_ablation(layers_to_sweep, sampler, n_samples=100)
    print("      Layer Sweep Summary (ROC-AUC by Depth):")
    for r in sweep_results:
        bar = "█" * int(r['roc_auc'] * 20)
        print(f"        Layer {r['layer']:02d}: AUC = {r['roc_auc']:.4f} | {bar}")

    # 4. Representation Steering Causal Test
    print("\n[4/4] Validating Causal Activation Steering (Recovery Rate)...")
    truth_acts = X_train[y_train == 1][:50]
    decep_acts = X_train[y_train == 0][:50]
    steerer = RepresentationSteering(hidden_dim=hidden_dim)
    v_truth = steerer.compute_steering_vector_from_pairs(truth_acts, decep_acts, layer=target_layer, normalize=False)

    # Test recovery on deceptive samples
    deceptive_test_samples = X_test[y_test == 0]
    recovered_count = 0
    alpha = 1.0
    for x in deceptive_test_samples:
        x_steered = steerer.steer_activation(x, layer=target_layer, alpha=alpha, preserve_norm=False)
        prob_steered = probe.predict_proba(x_steered)
        if prob_steered >= 0.5:
            recovered_count += 1

    recovery_rate = (recovered_count / len(deceptive_test_samples)) * 100.0
    print(f"      Causal Steering Recovery Rate (alpha = +{alpha}): {recovery_rate:.1f}% ({recovered_count}/{len(deceptive_test_samples)})")

    # Export benchmark report
    output_path = os.path.join(os.path.dirname(__file__), "benchmarks", "benchmark_results.json")
    results_payload = {
        "model_architecture": "Transformer Residual Stream (Llama-3-8B Config)",
        "hidden_dim": hidden_dim,
        "target_layer": target_layer,
        "test_metrics": test_metrics,
        "layer_sweep_ablation": sweep_results,
        "causal_steering_recovery_rate_pct": recovery_rate,
        "benchmark_pairs_count": len(TRUTHFULQA_BENCHMARK_PAIRS) + len(SYCOPHANCY_BENCHMARK_PAIRS)
    }

    with open(output_path, "w") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\n[+] Benchmark complete! Results saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
