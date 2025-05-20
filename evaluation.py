import pandas as pd
import random
import numpy as np
import time

# Import Range-Tree functions
from range_tree.main import build_range_tree, execute_range_query

# Import KD-Tree functions
from kd_tree.data import build_kdtree, execute_kd_query

# Import Quad-Tree functions
from quad_tree.quad_main import build_quadtree, execute_quad_query

# Import LSH functions
from lsh import run_lsh_on_candidates, build_shingle_vocab, generate_hash_functions

# === Step 1: Load the Dataset ===
df = pd.read_csv("cars24data.csv")

df_full = df[["Model Name", "Price", "Engine capacity", "KM driven"]].dropna()
df_main = df[["Price", "Engine capacity", "KM driven"]].dropna()

points = [tuple(int(v) for v in row) for row in df_main.to_numpy()]

# For looking up model names
lookup = {
    (int(row["Price"]), int(row["Engine capacity"]), int(row["KM driven"])): row["Model Name"]
    for _, row in df_full.iterrows()
}

# Prepare LSH parameters
model_names = df_full["Model Name"].dropna().unique()
vocab = build_shingle_vocab(model_names)

num_hashes = 20
bands = 10
rows_per_band = num_hashes // bands
hash_funcs = generate_hash_functions(num_hashes, len(vocab))

# Setup Query Range
price_min, price_max = df_main["Price"].min(), df_main["Price"].max()
engine_min, engine_max = df_main["Engine capacity"].min(), df_main["Engine capacity"].max()
km_min, km_max = df_main["KM driven"].min(), df_main["KM driven"].max()

def generate_random_query():
    x1 = random.randint(price_min, price_max)
    x2 = random.randint(x1, price_max)
    y1 = random.randint(engine_min, engine_max)
    y2 = random.randint(y1, engine_max)
    z1 = random.randint(km_min, km_max)
    z2 = random.randint(z1, km_max)
    return (x1, x2), (y1, y2), (z1, z2)

# === Step 2: Modular Evaluation Functions ===

def evaluate_range_tree(points, lookup, vocab, hash_funcs, model_names, n_queries=50):
    print("\nBuilding and Evaluating Range Tree + LSH...")
    tree, build_time = build_range_tree(points)

    query_model_name = "Maruti 2014"
    query_times = []
    lsh_times = []

    for _ in range(n_queries):
        x_range, y_range, z_range = generate_random_query()
        candidates, query_time = execute_range_query(tree, x_range, y_range, z_range)
        query_times.append(query_time)

        if not candidates:
            continue

        candidate_model_names = [
            lookup[(price, engine, km)]
            for (price, engine, km) in candidates
            if (price, engine, km) in lookup
        ]

        start_lsh = time.perf_counter()
        _ = run_lsh_on_candidates(
            candidate_model_names=candidate_model_names,
            query_model_name=query_model_name,
            vocab=vocab,
            hash_funcs=hash_funcs,
            bands=bands,
            rows_per_band=rows_per_band,
            ngram=2,
            top_n=5
        )
        end_lsh = time.perf_counter()

        lsh_times.append((end_lsh - start_lsh) * 1000)  # ms

    return build_time, np.mean(query_times), np.mean(lsh_times)

def evaluate_kd_tree(points, lookup, vocab, hash_funcs, model_names, n_queries=50):
    print("\nBuilding and Evaluating KD-Tree + LSH...")
    tree, build_time = build_kdtree(points)

    query_model_name = "Maruti 2014"
    query_times = []
    lsh_times = []

    for _ in range(n_queries):
        x_range, y_range, z_range = generate_random_query()
        candidates, query_time = execute_kd_query(tree, x_range, y_range, z_range)
        query_times.append(query_time)

        if not candidates:
            continue

        candidate_model_names = [
            lookup[(price, engine, km)]
            for (price, engine, km) in candidates
            if (price, engine, km) in lookup
        ]

        start_lsh = time.perf_counter()
        _ = run_lsh_on_candidates(
            candidate_model_names=candidate_model_names,
            query_model_name=query_model_name,
            vocab=vocab,
            hash_funcs=hash_funcs,
            bands=bands,
            rows_per_band=rows_per_band,
            ngram=2,
            top_n=5
        )
        end_lsh = time.perf_counter()

        lsh_times.append((end_lsh - start_lsh) * 1000)  # ms

    return build_time, np.mean(query_times), np.mean(lsh_times)


def evaluate_quadtree(cars, lookup, vocab, hash_funcs, model_names, n_queries=50):
    print("\nBuilding and Evaluating QuadTree + LSH...")
    tree, build_time = build_quadtree(cars)

    query_model_name = "Maruti 2014"
    query_times = []
    lsh_times = []

    for _ in range(n_queries):
        x_range, y_range, z_range = generate_random_query()
        candidates, query_time = execute_quad_query(tree, x_range, y_range, z_range)
        query_times.append(query_time)

        if not candidates:
            continue

        candidate_model_names = [
            lookup[(price, engine, km)]
            for (_, price, engine, km) in candidates
            if (price, engine, km) in lookup
        ]

        start_lsh = time.perf_counter()
        _ = run_lsh_on_candidates(
            candidate_model_names=candidate_model_names,
            query_model_name=query_model_name,
            vocab=vocab,
            hash_funcs=hash_funcs,
            bands=bands,
            rows_per_band=rows_per_band,
            ngram=2,
            top_n=5
        )
        end_lsh = time.perf_counter()

        lsh_times.append((end_lsh - start_lsh) * 1000)

    return build_time, np.mean(query_times), np.mean(lsh_times)



# === Step 3: Run All Evaluations ===

results = []

# Range Tree Evaluation
build_time, avg_query_time, avg_lsh_time = evaluate_range_tree(points, lookup, vocab, hash_funcs, model_names)
results.append(("Range Tree + LSH", build_time, avg_query_time, avg_lsh_time))

# KD-Tree Evaluation
build_time, avg_query_time, avg_lsh_time = evaluate_kd_tree(points, lookup, vocab, hash_funcs, model_names)
results.append(("KD-Tree + LSH", build_time, avg_query_time, avg_lsh_time))

# QuadTree Evaluation
build_time, avg_query_time, avg_lsh_time = evaluate_quadtree(df_full.values.tolist(), lookup, vocab, hash_funcs, model_names)
results.append(("QuadTree + LSH", build_time, avg_query_time, avg_lsh_time))

# === Step 4: Print Final Comparison Table ===

print("\n=== Final Performance Comparison ===")
print(f"{'Structure':25} | {'Build Time (s)':>15} | {'Avg Query Time (ms)':>20} | {'Avg LSH Time (ms)':>17}")
print("-" * 85)

for method, build_t, query_t, lsh_t in results:
    print(f"{method:25} | {build_t:15.4f} | {query_t:20.2f} | {lsh_t:17.2f}")
