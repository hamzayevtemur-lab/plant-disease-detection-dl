import csv
import json
import os
import time
from typing import List, Dict, Any

import torch
import matplotlib.pyplot as plt
from src.training.trainer import evaluate_saved_model, get_device, build_model


def get_available_experiments(experiments_dir: str = "experiments") -> List[str]:
    """Finds all experiment directories containing a best_model.pt checkpoint."""
    if not os.path.exists(experiments_dir):
        return []
    exps = []
    for name in sorted(os.listdir(experiments_dir)):
        exp_path = os.path.join(experiments_dir, name)
        if os.path.isdir(exp_path) and os.path.exists(os.path.join(exp_path, "best_model.pt")):
            exps.append(name)
    return exps


def measure_inference_speed(
    checkpoint_path: str,
    device: torch.device,
    num_runs: int = 50,
    warmup: int = 10
) -> Dict[str, float]:
    """Measures inference latency (ms/image) and throughput (FPS) for a single image forward pass."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_config = checkpoint["model_config"]
        model = build_model(model_config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        dummy_input = torch.randn(1, 3, 224, 224, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_input)

        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)

        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

        elapsed = time.perf_counter() - start_time
        avg_latency_ms = (elapsed / num_runs) * 1000.0
        fps = num_runs / elapsed

        return {"latency_ms": round(avg_latency_ms, 2), "fps": round(fps, 1)}
    except Exception as e:
        return {"latency_ms": 0.0, "fps": 0.0}


def load_all_results(experiments_dir: str = "experiments") -> List[Dict[str, Any]]:
    """Loads results for all experiments, auto-evaluating any missing results.json."""
    device = get_device()
    experiments = get_available_experiments(experiments_dir)
    results = []

    for exp_name in experiments:
        results_file = os.path.join(experiments_dir, exp_name, "results.json")
        checkpoint_path = os.path.join(experiments_dir, exp_name, "best_model.pt")

        if not os.path.exists(results_file):
            print(f"results.json missing for '{exp_name}'. Auto-evaluating checkpoint...")
            evaluate_saved_model(exp_name, experiments_base_dir=experiments_dir)

        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                data = json.load(f)

            # Checkpoint file size in MB
            if os.path.exists(checkpoint_path):
                size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
                data["size_mb"] = round(size_mb, 2)
            else:
                data["size_mb"] = 0.0

            # Measure inference speed if not cached
            if "latency_ms" not in data or data["latency_ms"] == 0.0:
                speed = measure_inference_speed(checkpoint_path, device)
                data.update(speed)
                with open(results_file, "w") as f:
                    json.dump(data, f, indent=4)

            results.append(data)

    # Sort descending by test accuracy
    results.sort(key=lambda x: x.get("test_accuracy", 0.0), reverse=True)
    return results


def save_results_csv(results: List[Dict[str, Any]], csv_path: str = "experiments/results.csv") -> str:
    """Saves benchmark results to a CSV file."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = [
        "experiment",
        "parameters",
        "size_mb",
        "latency_ms",
        "fps",
        "best_epoch",
        "validation_loss",
        "validation_accuracy",
        "test_loss",
        "test_accuracy"
    ]

    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "experiment": r.get("experiment", ""),
                "parameters": r.get("parameters", 0),
                "size_mb": r.get("size_mb", 0.0),
                "latency_ms": r.get("latency_ms", 0.0),
                "fps": r.get("fps", 0.0),
                "best_epoch": r.get("best_epoch", 0),
                "validation_loss": r.get("validation_loss", 0.0),
                "validation_accuracy": r.get("validation_accuracy", 0.0),
                "test_loss": r.get("test_loss", 0.0),
                "test_accuracy": r.get("test_accuracy", 0.0)
            })

    return csv_path


def print_comparison_table(results: List[Dict[str, Any]]) -> None:
    """Prints a formatted ASCII table comparing model accuracy, size, and speed for production selection."""
    print("\n" + "=" * 90)
    print("PLANT DISEASE DETECTION - PRODUCTION MODEL BENCHMARK")
    print("=" * 90)
    print(
        f"{'Experiment':<22}"
        f"{'Parameters':>12}"
        f"{'Size (MB)':>11}"
        f"{'Latency':>11}"
        f"{'Val Acc':>10}"
        f"{'Test Acc':>10}"
        f"{'Test Loss':>11}"
    )
    print("-" * 90)

    for r in results:
        lat = f"{r.get('latency_ms', 0.0):.1f} ms" if r.get('latency_ms') else "N/A"
        print(
            f"{r.get('experiment', ''):<22}"
            f"{r.get('parameters', 0):>12,}"
            f"{r.get('size_mb', 0.0):>10.1f}MB"
            f"{lat:>11}"
            f"{r.get('validation_accuracy', 0.0):>9.2%}"
            f"{r.get('test_accuracy', 0.0):>9.2%}"
            f"{r.get('test_loss', 0.0):>11.4f}"
        )

    print("=" * 90)
    if results:
        best = results[0]
        print(f"🏆 Highest Accuracy Model: {best.get('experiment')} ({best.get('test_accuracy', 0.0):.2%} Test Accuracy)")
    print()


def plot_comparison(results: List[Dict[str, Any]], save_dir: str = "results/figures") -> None:
    """Generates and saves visual comparison charts."""
    if not results:
        return

    os.makedirs(save_dir, exist_ok=True)
    names = [r["experiment"] for r in results]
    test_accs = [r["test_accuracy"] * 100 for r in results]

    # Test Accuracy Bar Plot
    plt.figure(figsize=(10, 5))
    bars = plt.bar(names, test_accs, color="#2E7D32", width=0.45)
    plt.title("Model Test Accuracy Comparison", fontsize=14, fontweight="bold")
    plt.xlabel("Experiment", fontsize=11)
    plt.ylabel("Test Accuracy (%)", fontsize=11)
    plt.ylim(0, 105)

    for bar, acc in zip(bars, test_accs):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{acc:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold"
        )

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "model_accuracy_comparison.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved benchmark plot to: {plot_path}")


def main():
    results = load_all_results()
    print_comparison_table(results)
    csv_path = save_results_csv(results)
    print(f"Updated benchmark results at: {csv_path}")
    plot_comparison(results)


if __name__ == "__main__":
    main()
