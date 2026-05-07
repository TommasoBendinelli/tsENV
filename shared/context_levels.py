from __future__ import annotations

# Active benchmark contexts.
BENCHMARK_CONTEXT_VALUES = ("none", "low", "high", "ground_truth")

# Question payload validation may also see synthetic/derived unknown values.
QUESTION_CONTEXT_VALUES = (*BENCHMARK_CONTEXT_VALUES, "unknown")

# Active tsENV contexts accepted by current generation and validation paths.
TSENV_CONTEXT_VALUES = ("none", "low", "high", "ground_truth")

# Legacy parsing order for dataset labels and variant names.
# `medium` is retained only for backward-compatible parsing of historical labels.
PARSEABLE_CONTEXT_VALUES = ("ground_truth", "high", "medium", "low", "none")
