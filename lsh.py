import re
import pandas as pd
import random
import numpy as np
from collections import defaultdict

def get_shingles(text, n=2):
    """
    Returns a set of n-gram shingles from a given string.
    Default is bigrams (n=2).
    """
    # Clean the text: lowercase and remove non-alphanumeric characters
    cleaned = re.sub(r'[^a-z0-9]', '', text.lower())
    return {cleaned[i:i+n] for i in range(len(cleaned) - n + 1)}


def build_shingle_vocab(model_names, n=2):
    """
    Builds the universal set of n-gram shingles (vocabulary)
    from a list of model names.
    """
    vocab = set()
    for name in model_names:
        shingles = get_shingles(name, n=n)
        vocab.update(shingles)
    return sorted(vocab)  # Optional: return in fixed order for consistent indexing



def generate_hash_functions(num_hashes, vocab_size, prime=4294967311):
    """
    Generate num_hashes hash functions of the form: h(x) = (a*x + b) % prime
    We mod again with vocab_size to keep it in range.
    """
    hash_funcs = []
    used = set()
    while len(hash_funcs) < num_hashes:
        a, b = random.randint(1, prime-1), random.randint(0, prime-1)
        if (a, b) not in used:
            hash_funcs.append((a, b))
            used.add((a, b))
    return hash_funcs



def minhash_signature(shingle_set, vocab, hash_funcs, prime=4294967311):
    """
    Create a MinHash signature vector for a single shingle set.
    """
    signature = []
    vocab_index = {shingle: idx for idx, shingle in enumerate(vocab)}
    shingles_idx = [vocab_index[s] for s in shingle_set if s in vocab]

    for (a, b) in hash_funcs:
        min_hash = float('inf')
        for idx in shingles_idx:
            h = (a * idx + b) % prime
            if h < min_hash:
                min_hash = h
        signature.append(min_hash)
    
    return signature


def build_lsh_index(signatures, bands, rows_per_band):
    """
    Create LSH index using banding technique.
    signatures: dict of {model_name: signature_vector}
    """
    assert len(next(iter(signatures.values()))) == bands * rows_per_band, \
        "Signature length must equal bands * rows_per_band"

    lsh_buckets = [defaultdict(set) for _ in range(bands)]  # One hash table per band

    for model_name, signature in signatures.items():
        for b in range(bands):
            start = b * rows_per_band
            end = start + rows_per_band
            band_slice = tuple(signature[start:end])
            bucket_key = hash(band_slice)
            lsh_buckets[b][bucket_key].add(model_name)

    return lsh_buckets


def lsh_query(query_signature, lsh_buckets, bands, rows_per_band):
    """
    Given a query MinHash signature, return a set of candidate model names
    that fall into the same buckets in at least one band.
    """
    assert len(query_signature) == bands * rows_per_band, \
        "Query signature must match LSH band configuration"

    candidates = set()

    for b in range(bands):
        start = b * rows_per_band
        end = start + rows_per_band
        band_slice = tuple(query_signature[start:end])
        bucket_key = hash(band_slice)

        # Get matching models in this band (if any)
        bucket = lsh_buckets[b].get(bucket_key, set())
        candidates.update(bucket)

    return candidates


def jaccard_similarity(set1, set2):
    """Compute Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0  # both empty, treat as identical
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union


def rank_candidates_by_jaccard(query_shingles, candidates, ngram, top_n):
    """Ranks candidate models by their Jaccard similarity with the query."""
    ranked = []

    for candidate in candidates:
        candidate_shingles = get_shingles(candidate, n=ngram)
        sim = jaccard_similarity(query_shingles, candidate_shingles)
        ranked.append((candidate, sim))

    # Sort by similarity score (highest first)
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:top_n]  # return only top-N


# Add this at the bottom of your lsh.py file:

def run_lsh_on_candidates(candidate_model_names, query_model_name, vocab, hash_funcs, bands, rows_per_band, ngram, top_n):
    """
    Runs full LSH matching on a subset of candidate models and returns the top-N most similar to the query model.
    """
    # Build signatures for candidates
    candidate_signatures = {}
    for name in candidate_model_names:
        shingles = get_shingles(name, n=ngram)
        signature = minhash_signature(shingles, vocab, hash_funcs)
        candidate_signatures[name] = signature

    # Build temporary LSH index
    candidate_lsh_index = build_lsh_index(candidate_signatures, bands=bands, rows_per_band=rows_per_band)

    # Build query signature
    query_shingles = get_shingles(query_model_name, n=ngram)
    query_signature = minhash_signature(query_shingles, vocab, hash_funcs)

    # Query LSH
    lsh_candidates = lsh_query(query_signature, candidate_lsh_index, bands=bands, rows_per_band=rows_per_band)

    # Rank by true Jaccard
    top_similar = rank_candidates_by_jaccard(query_shingles, lsh_candidates, ngram=ngram, top_n=top_n)

    return top_similar



if __name__ == "__main__":

   # Step 1: Load dataset and extract model names
    df = pd.read_csv("cars24data.csv")
    model_names = df["Model Name"].dropna().unique()

    # Step 2: Build vocabulary and hash functions for MinHashing
    vocab = build_shingle_vocab(model_names)
    num_hashes = 100
    bands = 20
    rows_per_band = num_hashes // bands
    hash_funcs = generate_hash_functions(num_hashes, len(vocab))

    # Step 3: Ask user for the query model
    query_model = input("\nEnter the model name to query (e.g., 'Hyundai i10'): ").strip()

    # Step 4: Run LSH on full dataset (all models) to find similar models
    top_similar = run_lsh_on_candidates(
        candidate_model_names=model_names,
        query_model_name=query_model,
        vocab=vocab,
        hash_funcs=hash_funcs,
        bands=bands,
        rows_per_band=rows_per_band,
        ngram=2,
        top_n=5
    )

    # Step 5: Print final Top-N similar models
    print(f"\nTop-5 models similar to '{query_model}':")
    for model, sim in top_similar:
        print(f"{model:30} → Similarity: {sim:.3f}")

