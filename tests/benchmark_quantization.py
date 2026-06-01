import logging
import os
import time

import numpy as np
import psutil
from light_embed import TextEmbedding
# from model2vec import StaticModel

logging.getLogger("light_embed").setLevel(logging.ERROR)

MODEL_DIR = "./models/paraphrase-multilingual-MiniLM-L12-v2"
ORIGINAL_MODEL_FILE = "model.onnx"
INT8_MODEL_FILE = "model.int8.onnx"
MODEL2VEC_NAME = "minishlab/potion-base-8M"

BATCH_SIZES = [1, 10, 50]
REFERENCE_MODEL = "Original ONNX"
QUALITY_TOP_K = 3

QUALITY_TEXTS = [
    "Artificial intelligence will fundamentally change software systems.",
    "AI is transforming how software applications are designed and maintained.",
    "Machine learning models can help developers detect bugs in code.",
    "Renewable energy storage is important for modern power grids.",
    "Battery technology improvements can increase electric vehicle range.",
    "تحسن تقنيات البطاريات مدى السيارات الكهربائية.",
    "الذكاء الاصطناعي يغير طريقة بناء البرمجيات.",
    "تخزين الطاقة المتجددة مهم لاستقرار الشبكات الكهربائية.",
    "A recipe for tomato soup needs fresh tomatoes and basil.",
    "Football teams need strong defense and coordinated attacks.",
]

LABELED_PAIRS = [
    (0, 1, True),
    (0, 2, True),
    (3, 7, True),
    (4, 5, True),
    (0, 6, True),
    (3, 4, True),
    (0, 8, False),
    (1, 9, False),
    (2, 3, False),
    (5, 8, False),
    (6, 9, False),
    (7, 8, False),
]


def get_memory_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    left = normalize_rows(v1)[0]
    right = normalize_rows(v2)[0]
    return float(np.dot(left, right))


def encode_texts(model, texts: list[str]) -> np.ndarray:
    return normalize_rows(np.asarray(list(model.encode(texts)), dtype=np.float32))


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    vectors = normalize_rows(vectors)
    return vectors @ vectors.T


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    row_idx, col_idx = np.triu_indices_from(matrix, k=1)
    return matrix[row_idx, col_idx]


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)

    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1

        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end

    return ranks


def pearson_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)

    left = left - left.mean()
    right = right - right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        return 0.0

    return float(np.dot(left, right) / denominator)


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    return pearson_correlation(rankdata(left), rankdata(right))


def mean_top_k_overlap(
    baseline_matrix: np.ndarray, candidate_matrix: np.ndarray, top_k: int
) -> float:
    overlaps = []
    for row_idx in range(baseline_matrix.shape[0]):
        baseline_order = np.argsort(-baseline_matrix[row_idx])
        candidate_order = np.argsort(-candidate_matrix[row_idx])

        baseline_neighbors = [idx for idx in baseline_order if idx != row_idx][:top_k]
        candidate_neighbors = [idx for idx in candidate_order if idx != row_idx][:top_k]

        overlap = len(set(baseline_neighbors) & set(candidate_neighbors)) / top_k
        overlaps.append(overlap)

    return float(np.mean(overlaps))


def labeled_pair_accuracy(matrix: np.ndarray) -> tuple[float, float]:
    positive_scores = []
    negative_scores = []

    for left_idx, right_idx, is_positive in LABELED_PAIRS:
        score = float(matrix[left_idx, right_idx])
        if is_positive:
            positive_scores.append(score)
        else:
            negative_scores.append(score)

    correct = 0
    total = 0
    for positive_score in positive_scores:
        for negative_score in negative_scores:
            correct += positive_score > negative_score
            total += 1

    margin = float(np.mean(positive_scores) - np.mean(negative_scores))
    return correct / total, margin


def benchmark_inference(model, text: str, batch_sizes: list[int] | None = None):
    batch_sizes = batch_sizes or BATCH_SIZES
    results = {}

    for batch_size in batch_sizes:
        texts = [text] * batch_size

        start = time.perf_counter()
        _ = encode_texts(model, texts)
        elapsed = time.perf_counter() - start

        results[batch_size] = {
            "time_ms": elapsed * 1000,
            "throughput": batch_size / elapsed,
        }

    return results


def run_onnx_benchmark(model_filename: str):
    print(f"\n{'=' * 60}")
    print(f"ONNX Benchmark: {model_filename}")
    print(f"{'=' * 60}")

    filepath = os.path.join(MODEL_DIR, model_filename)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)

    print(f"Disk size: {size_mb:.2f} MB")

    mem_before = get_memory_mb()

    config = {
        "onnx_file": model_filename,
        "pooling_config_path": "1_Pooling",
        "normalize": False,
    }

    start = time.perf_counter()
    model = TextEmbedding(model_name_or_path=MODEL_DIR, model_config=config)
    load_time = time.perf_counter() - start

    mem_after = get_memory_mb()

    print(f"Load time: {load_time:.4f}s")
    print(f"RAM delta: {mem_after - mem_before:.2f} MB")

    text = "Artificial intelligence will fundamentally change software systems."

    _ = encode_texts(model, [text])
    infer_stats = benchmark_inference(model, text)
    vectors = encode_texts(model, QUALITY_TEXTS)
    single_vector = encode_texts(model, [text])[0]

    print(f"Embedding dim: {vectors.shape[1]}")

    del model

    return {
        "vectors": vectors,
        "single_vector": single_vector,
        "inference": infer_stats,
        "memory_mb": mem_after - mem_before,
        "disk_mb": size_mb,
        "dim": vectors.shape[1],
    }


def run_model2vec_benchmark():
    print(f"\n{'=' * 60}")
    print("Model2Vec Benchmark")
    print(f"{'=' * 60}")

    mem_before = get_memory_mb()

    start = time.perf_counter()
    # model = StaticModel.from_pretrained(MODEL2VEC_NAME)
    load_time = time.perf_counter() - start

    mem_after = get_memory_mb()

    print(f"Model: {MODEL2VEC_NAME}")
    print(f"Load time: {load_time:.4f}s")
    print(f"RAM delta: {mem_after - mem_before:.2f} MB")

    text = "Artificial intelligence will fundamentally change software systems."

    _ = encode_texts(model, [text])
    infer_stats = benchmark_inference(model, text)
    vectors = encode_texts(model, QUALITY_TEXTS)
    single_vector = encode_texts(model, [text])[0]

    print(f"Embedding dim: {vectors.shape[1]}")

    del model

    return {
        "vectors": vectors,
        "single_vector": single_vector,
        "inference": infer_stats,
        "memory_mb": mem_after - mem_before,
        "disk_mb": None,
        "dim": vectors.shape[1],
    }


def print_inference_stats(name: str, stats):
    print(f"\n{name} Inference Stats")
    for batch, data in stats.items():
        print(
            f"Batch {batch:>2}: "
            f"{data['time_ms']:.2f} ms | "
            f"{data['throughput']:.2f} req/sec"
        )


def print_quality_report(results):
    print(f"\n{'=' * 60}")
    print("Semantic Quality Comparison")
    print(f"{'=' * 60}")
    print(
        "Cross-model vectors are not compared directly. Each model builds its "
        "own cosine-similarity matrix over the same texts, then those rankings "
        "are compared."
    )

    matrices = {
        name: cosine_matrix(result["vectors"]) for name, result in results.items()
    }
    baseline_matrix = matrices[REFERENCE_MODEL]
    baseline_scores = upper_triangle_values(baseline_matrix)

    for name, result in results.items():
        matrix = matrices[name]
        scores = upper_triangle_values(matrix)
        pair_accuracy, labeled_margin = labeled_pair_accuracy(matrix)

        if name == REFERENCE_MODEL:
            rank_corr = 1.0
            top_k_overlap = 1.0
        else:
            rank_corr = spearman_correlation(baseline_scores, scores)
            top_k_overlap = mean_top_k_overlap(
                baseline_matrix, matrix, top_k=QUALITY_TOP_K
            )

        print(f"\n{name}")
        print(f"  dim: {result['dim']}")
        print(f"  rank correlation vs {REFERENCE_MODEL}: {rank_corr:.4f}")
        print(
            f"  mean top-{QUALITY_TOP_K} overlap vs {REFERENCE_MODEL}: {top_k_overlap:.4f}"
        )
        print(f"  labeled pair ranking accuracy: {pair_accuracy:.4f}")
        print(f"  positive-negative similarity margin: {labeled_margin:.4f}")

    same_dim = results["Original ONNX"]["dim"] == results["INT8 ONNX"]["dim"]
    if same_dim:
        same_text_cosine = cosine_similarity(
            results["Original ONNX"]["single_vector"],
            results["INT8 ONNX"]["single_vector"],
        )
        print(f"\nOriginal ONNX vs INT8 same-text cosine: {same_text_cosine:.6f}")


def print_final_analysis(results):
    print(f"\n{'=' * 60}")
    print("Final Analysis")
    print(f"{'=' * 60}")

    memory = {name: result["memory_mb"] for name, result in results.items()}
    best_model = min(memory, key=memory.get) # type: ignore
    worst_model = max(memory, key=memory.get) # type: ignore

    print(f"Lowest RAM model: {best_model} ({memory[best_model]:.2f} MB)")
    print(f"Highest RAM model: {worst_model} ({memory[worst_model]:.2f} MB)")

    original_memory = results["Original ONNX"]["memory_mb"]
    model2vec_memory = results["Model2Vec"]["memory_mb"]
    if original_memory:
        ram_reduction = (original_memory - model2vec_memory) / original_memory * 100
        print(f"RAM reduction (Original ONNX -> Model2Vec): {ram_reduction:.2f}%")

    speed_ratio = (
        results["Model2Vec"]["inference"][1]["throughput"]
        / results["Original ONNX"]["inference"][1]["throughput"]
    )

    print(f"Speed ratio, batch size 1 (Model2Vec / Original ONNX): {speed_ratio:.2f}x")


def main():
    print("Starting embedding benchmark suite")
    print(f"Baseline RAM: {get_memory_mb():.2f} MB")

    results = {
        "Original ONNX": run_onnx_benchmark(ORIGINAL_MODEL_FILE),
        "INT8 ONNX": run_onnx_benchmark(INT8_MODEL_FILE),
        "Model2Vec": run_model2vec_benchmark(),
    }

    for name, result in results.items():
        print_inference_stats(name, result["inference"])

    print_quality_report(results)
    print_final_analysis(results)


if __name__ == "__main__":
    main()
