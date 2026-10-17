"""
DeceptionProbe Package Initialization.
"""
from core.probe import LinearProbe, ContrastConsistentSearch, ProbePrediction
from core.steering import RepresentationSteering
from core.dataset import ActivationSampler, TRUTHFULQA_BENCHMARK_PAIRS, SYCOPHANCY_BENCHMARK_PAIRS
from core.evaluator import ProbeEvaluator

__version__ = "0.1.0"
__all__ = [
    "LinearProbe",
    "ContrastConsistentSearch",
    "ProbePrediction",
    "RepresentationSteering",
    "ActivationSampler",
    "ProbeEvaluator",
    "TRUTHFULQA_BENCHMARK_PAIRS",
    "SYCOPHANCY_BENCHMARK_PAIRS"
]
