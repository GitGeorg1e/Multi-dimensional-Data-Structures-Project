# Multi-dimensional-Data-Structures-Project

## 📘 Overview

This project explores and compares various **multidimensional data structures** for efficient storage and querying of a 3D dataset, using car listings from [Cars24 Kaggle dataset](https://www.kaggle.com/datasets/amanrajput16/used-car-price-data-from-cars24).  
Each car is represented using 3 features: `Price`, `Engine Capacity`, and `KM Driven`.

We implemented the following structures:

- **Octree**
- **3D Range Tree**
- **KD-Tree**
- **R-Tree**
- **LSH (Locality-Sensitive Hashing)** for string similarity on `Model Name`


## 🧱 Implemented Structures

### 🔹 Octree (Point-based)
Recursive spatial decomposition into 8 subcubes for efficient **3D point storage** and **range queries**.

### 🔹 3D Range Tree
Nested balanced BSTs for each axis (X: Price, Y: Engine, Z: KM) with **canonical subtrees** for optimized queries.

### 🔹 KD-Tree
Efficient median-splitting BST using cyclical axes (x → y → z), supporting both **range search** and **k-NN queries**.

### 🔹 R-Tree
An efficient spatial indexing structure that uses **Minimum Bounding Rectangles (MBRs)** to index multidimensional data.  


---

## 🔍 LSH for Model Name Similarity

Used for fuzzy matching of car models based on **2-shingles**, **MinHashing**, and **Locality Sensitive Hashing**. Final results are ranked using **Jaccard Similarity**.

---

## 📊 Evaluation

Using `evaluation.py`, we measured:

- Build Time
- Query Time (50 random queries)
- LSH Time & Candidate Count

### Key Insights:
- **Range Tree**: Slowest to build, fastest in query time.
- **KD-Tree & Octree**: Balanced performance, fast build and query times.
- **R-Tree**: Slightly slower due to overlapping regions.
- **LSH performance** closely depends on the filtering effectiveness of the spatial structure.

---

## 📎 Sample Execution Outputs

Screenshots and example outputs for each structure and the evaluation script are included in the report and the project files.
