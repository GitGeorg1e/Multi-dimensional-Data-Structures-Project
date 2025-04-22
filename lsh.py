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


def rank_candidates_by_jaccard(query_shingles, candidates, ngram=2, top_n=10):
    """Ranks candidate models by their Jaccard similarity with the query."""
    ranked = []

    for candidate in candidates:
        candidate_shingles = get_shingles(candidate, n=ngram)
        sim = jaccard_similarity(query_shingles, candidate_shingles)
        ranked.append((candidate, sim))

    # Sort by similarity score (highest first)
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:top_n]  # return only top-N


if __name__ == "__main__":

   # Step 1: Load dataset and extract model names
    df = pd.read_csv("cars24data.csv")
    model_names = df["Model Name"].dropna().unique()

    # Step 2: Build vocabulary from bigram shingles
    vocab = build_shingle_vocab(model_names)

    # Step 3: Generate hash functions for MinHashing
    num_hashes = 100
    bands = 20
    rows_per_band = num_hashes // bands
    hash_funcs = generate_hash_functions(num_hashes, len(vocab))

    # Step 4: Create MinHash signatures for all model names
    signatures = {}
    for name in model_names:
        shingles = get_shingles(name)
        signature = minhash_signature(shingles, vocab, hash_funcs)
        signatures[name] = signature

    # Step 5: Build LSH index
    lsh_index = build_lsh_index(signatures, bands=bands, rows_per_band=rows_per_band)

    # Step 6: Query model
    query_model = input("\nEnter the model name to query (e.g., 'Hyundai i10'): ").strip()
    query_shingles = get_shingles(query_model)
    query_signature = minhash_signature(query_shingles, vocab, hash_funcs)

    # Step 7: LSH query to get candidate matches
    candidates = lsh_query(query_signature, lsh_index, bands=bands, rows_per_band=rows_per_band)
    print(f"\nFound {len(candidates)} candidate models through LSH.")

    # Step 8: Rank candidates using actual Jaccard similarity
    top_similar = rank_candidates_by_jaccard(query_shingles, candidates, top_n=10)

    # Step 9: Print final Top-N similar models
    print(f"\nTop-10 models similar to '{query_model}':")
    for model, sim in top_similar:
        print(f"{model:30} → Similarity: {sim:.3f}")
