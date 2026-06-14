from __future__ import annotations

import json
import math
import random
from pathlib import Path


EXPLAINERS = [
    "DeepLiftshap",
    "KernelShape",
    "Lime",
    "GradShap",
    "Saliency",
    "Intgrag",
    "DeepLift",
    "GuidedBackprop",
    "GuidedGradCam",
]


def _normalise(values: list[float]) -> list[float]:
    low = min(values)
    high = max(values)
    span = high - low
    if span == 0:
        return [0.0 for _ in values]
    return [(value - low) / span for value in values]


def make_synthetic_case(size: int = 16, seed: int = 7) -> tuple[list[float], list[float], list[list[float]]]:
    """Create a tiny flattened 3D case and nine noisy explanation maps."""
    rng = random.Random(seed)
    coords = [((i / (size - 1)) * 2.0) - 1.0 for i in range(size)]
    target = []
    image = []
    for z in coords:
        for y in coords:
            for x in coords:
                signal = math.exp(-8.0 * (x * x + y * y + z * z))
                target.append(signal)
                image.append(signal + 0.05 * rng.gauss(0.0, 1.0))

    target = _normalise(target)
    image = _normalise(image)

    explanations = []
    for index, _name in enumerate(EXPLAINERS):
        noise = 0.04 + index * 0.015
        explanation = []
        cursor = 0
        for z in coords:
            for y in coords:
                for x in coords:
                    bias = 0.02 * math.sin((index + 1) * x)
                    explanation.append(target[cursor] + bias + noise * rng.gauss(0.0, 1.0))
                    cursor += 1
        explanations.append(_normalise(explanation))
    return image, target, explanations


def complexity(explanation: list[float], threshold: float = 0.25) -> float:
    """Fraction of active voxels; lower values are easier to inspect."""
    return sum(1 for value in explanation if value > threshold) / len(explanation)


def faithfulness(explanation: list[float], target: list[float]) -> float:
    """Pearson correlation with the known synthetic ground-truth explanation."""
    mean_e = sum(explanation) / len(explanation)
    mean_t = sum(target) / len(target)
    centered_e = [value - mean_e for value in explanation]
    centered_t = [value - mean_t for value in target]
    numerator = sum(e * t for e, t in zip(centered_e, centered_t))
    denom_e = math.sqrt(sum(e * e for e in centered_e))
    denom_t = math.sqrt(sum(t * t for t in centered_t))
    if denom_e == 0 or denom_t == 0:
        return 0.0
    return numerator / (denom_e * denom_t)


def run_toy_pipeline(output_dir: str | Path = "functional_testing/results") -> dict[str, object]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _image, target, explanations = make_synthetic_case()

    raw_metrics = []
    for name, explanation in zip(EXPLAINERS, explanations):
        raw_metrics.append(
            {
                "name": name,
                "faithfulness": faithfulness(explanation, target),
                "complexity": complexity(explanation),
            }
        )

    scores = [max(row["faithfulness"], 0.0) / max(row["complexity"], 1e-8) for row in raw_metrics]
    total_score = sum(scores)
    weights = [score / total_score for score in scores]
    fused = []
    for voxel_index in range(len(target)):
        fused.append(sum(weight * explanation[voxel_index] for weight, explanation in zip(weights, explanations)))
    fused = _normalise(fused)

    result = {
        "dataset": "synthetic_16x16x16_gaussian",
        "num_explainers": len(EXPLAINERS),
        "weights": {name: weight for name, weight in zip(EXPLAINERS, weights)},
        "individual_metrics": raw_metrics,
        "fused_metrics": {
            "faithfulness": faithfulness(fused, target),
            "complexity": complexity(fused),
        },
    }

    with (output_path / "toy_results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


if __name__ == "__main__":
    results = run_toy_pipeline(Path(__file__).resolve().parent / "results")
    print(json.dumps(results["fused_metrics"], indent=2))

