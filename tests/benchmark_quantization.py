import os
import time
import psutil
import numpy as np
import logging
from light_embed import TextEmbedding

# Disable logging to keep the output clean
logging.getLogger("light_embed").setLevel(logging.ERROR)

MODEL_DIR = "./models/paraphrase-multilingual-MiniLM-L12-v2"
ORIGINAL_MODEL_FILE = "model.onnx"
INT8_MODEL_FILE = "model.int8.onnx"


def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def run_benchmark(model_filename):
    print(f"\n{'=' * 50}")
    print(f"Benchmarking: {model_filename}")
    print(f"{'=' * 50}")

    filepath = os.path.join(MODEL_DIR, model_filename)
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        print(f"Please ensure you placed it at {filepath}")
        return None

    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"📦 File Size on Disk: {size_mb:.2f} MB")

    mem_before = get_memory_mb()

    config = {
        "onnx_file": model_filename,
        "pooling_config_path": "1_Pooling",
        "normalize": False,
    }

    # Measure Load Time
    start = time.perf_counter()
    model = TextEmbedding(model_name_or_path=MODEL_DIR, model_config=config)
    load_time = time.perf_counter() - start

    mem_after = get_memory_mb()
    mem_used = mem_after - mem_before

    print(f"⏱️ Load Time:         {load_time:.4f} seconds")
    print(f"🧠 RAM Used by Model: {mem_used:.2f} MB")

    # Measure Inference Time
    text = "Artificial intelligence will fundamentally change how we build software systems."

    # Warmup
    _ = list(model.encode([text]))

    # Single doc
    start = time.perf_counter()
    vec1 = list(model.encode([text]))[0]
    inf_time_1 = time.perf_counter() - start
    print(f"⚡ Inference (1 doc):  {inf_time_1 * 1000:.2f} ms")

    # Batch doc
    texts = [text] * 10
    start = time.perf_counter()
    _ = list(model.encode(texts))
    inf_time_10 = time.perf_counter() - start
    print(f"⚡ Inference (10 doc): {inf_time_10 * 1000:.2f} ms")

    # Delete model from memory manually to ensure clean RAM reading for the next model
    del model

    return np.array(vec1, dtype=np.float32)


def main():
    print("🚀 Starting Benchmark...")
    print(f"Baseline Python RAM usage: {get_memory_mb():.2f} MB")

    vec_orig = run_benchmark(ORIGINAL_MODEL_FILE)
    vec_int8 = run_benchmark(INT8_MODEL_FILE)

    if vec_orig is not None and vec_int8 is not None:
        print(f"\n{'=' * 50}")
        print("🎯 Accuracy Comparison")
        print(f"{'=' * 50}")
        sim = cosine_similarity(vec_orig, vec_int8)
        print(f"Cosine Similarity (Original vs INT8): {sim:.6f}")

        if sim >= 0.99:
            print(
                "✅ Verdict: The INT8 model is practically identical to the original!"
            )
        else:
            print("⚠️ Verdict: Significant accuracy difference detected.")


if __name__ == "__main__":
    main()
