import pandas as pd
import random
import numpy as np
import time

# Import RangeTree functions
from range_tree.main import build_range_tree, execute_range_query

# Import LSH function
from lsh import run_lsh_on_candidates, build_shingle_vocab, generate_hash_functions

# === Step 1: Load the Dataset ===
df = pd.read_csv("cars24data.csv")  # Adjust path if needed

df_full = df[["Model Name", "Price", "Engine capacity", "KM driven"]].dropna()
df_main = df[["Price", "Engine capacity", "KM driven"]].dropna()

points = [tuple(int(v) for v in row) for row in df_main.to_numpy()]

# For looking up model names later
lookup = {
    (int(row["Price"]), int(row["Engine capacity"]), int(row["KM driven"])): row["Model Name"]
    for _, row in df_full.iterrows()
}

# === Step 2: Build the 3D Range Tree ===
print("Building 3D Range Tree...")
tree3D, build_time = build_range_tree(points)

# === Step 3: Prepare global LSH parameters (vocab + hash functions) ===
print("Preparing LSH parameters...")
model_names = df_full["Model Name"].dropna().unique()
vocab = build_shingle_vocab(model_names)

num_hashes = 100
bands = 20
rows_per_band = num_hashes // bands
hash_funcs = generate_hash_functions(num_hashes, len(vocab))

# === Step 4: Setup Query Generation ===

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

# === Step 5: Evaluation Loop ===

query_model_name = "Maruti"  # You can randomize or input this if you want
n_queries = 50  # Number of random queries
query_times = []
lsh_times = []
total_results = 0

print("\nRunning evaluation with Range Tree + LSH...")

for _ in range(n_queries):
    x_range, y_range, z_range = generate_random_query()

    # 1. Range Query
    candidates, query_time = execute_range_query(tree3D, x_range, y_range, z_range)
    query_times.append(query_time)

    if not candidates:
        continue  # Skip if no candidates

    # 2. Get model names
    candidate_model_names = [
        lookup[(price, engine, km)]
        for (price, engine, km) in candidates
        if (price, engine, km) in lookup
    ]

    # 3. Run LSH on candidates
    start_lsh = time.perf_counter()
    top_similar = run_lsh_on_candidates(
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

    lsh_times.append((end_lsh - start_lsh) * 1000)  # in milliseconds
    total_results += len(top_similar)

# === Step 6: Final Reporting ===

avg_query_time = np.mean(query_times)
avg_lsh_time = np.mean(lsh_times)
avg_total_results = total_results / n_queries

print("\n=== Final Evaluation Results ===")
print(f"Build Time for Range Tree: {build_time:.4f} seconds")
print(f"Average Range Query Time over {n_queries} queries: {avg_query_time:.2f} ms")
print(f"Average LSH Processing Time over {n_queries} queries: {avg_lsh_time:.2f} ms")
print(f"Average Top-N Results Found per Query: {avg_total_results:.2f}")
