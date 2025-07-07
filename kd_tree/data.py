import time
import pandas as pd
import heapq
import math


# Κλάση κόμβου
class KDNode:
    def __init__(self, point, axis, left=None, right=None):
        self.point = point
        self.axis = axis
        self.left = left
        self.right = right

# Κλάση custom KD-Tree με print και range query
class KDTreeCustom:
    def __init__(self, data, depth=0):
        if not data:
            self.node = None
            return

        k = len(data[0])  # number of dimensions
        axis = depth % k

        data.sort(key=lambda x: x[axis])
        median = len(data) // 2

        self.node = KDNode(
            point=data[median],
            axis=axis,
            left=KDTreeCustom(data[:median], depth + 1),
            right=KDTreeCustom(data[median + 1:], depth + 1)
        )

    def print_tree(self, node=None, depth=0, direction="ROOT"):
        if node is None:
            node = self.node
        if node is not None:
            indent = "  " * depth
            print(f"{indent}[{direction}] Depth {depth}, Axis {node.axis}, Point {node.point}")
            if node.left and node.left.node:
                self.print_tree(node.left.node, depth + 1, "LEFT")
            if node.right and node.right.node:
                self.print_tree(node.right.node, depth + 1, "RIGHT")

    def range_query(self, node, ranges, results):
        if node is None:
            return
        point = node.point
        axis = node.axis

        in_range = all(ranges[i][0] <= point[i] <= ranges[i][1] for i in range(len(ranges)))
        if in_range:
            results.append(point)

        if point[axis] >= ranges[axis][0]:
            self.range_query(node.left.node, ranges, results)
        if point[axis] <= ranges[axis][1]:
            self.range_query(node.right.node, ranges, results)

def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

# Συνάρτηση αναζήτησης k πλησιέστερων σημείων
def knn_search(tree, query_point, k=3):
    best = []

    def recursive_search(node):
        if node is None:
            return

        point = node.point
        axis = node.axis
        dist = euclidean_distance(point, query_point)

        if len(best) < k:
            heapq.heappush(best, (-dist, point))
        elif dist < -best[0][0]:
            heapq.heappushpop(best, (-dist, point))

        diff = query_point[axis] - point[axis]
        first, second = (node.left, node.right) if diff < 0 else (node.right, node.left)

        recursive_search(first.node)

        if len(best) < k or abs(diff) < -best[0][0]:
            recursive_search(second.node)

    recursive_search(tree.node)
    return [point for (_, point) in sorted(best, reverse=True)]    


def build_kdtree(points):
    start = time.perf_counter()
    tree = KDTreeCustom(points)
    end = time.perf_counter()
    build_time = end - start
    print(f"Build time for KD-Tree: {build_time:.4f} seconds")
    return tree, build_time     


def execute_kd_query(tree, x_range, y_range, z_range):
    ranges = [x_range, y_range, z_range]
    results = []
    start = time.perf_counter()
    tree.range_query(tree.node, ranges, results)
    end = time.perf_counter()
    query_time = (end - start) * 1000  # in ms
    return results, query_time


if __name__ == "__main__":
    
    # Load the CSV file directly
    df = pd.read_csv("../cars24data.csv")

    # Select the relevant columns
    columns = ['Price', 'Engine capacity', 'KM driven', 'Model Name']
    selected_data = df[columns].dropna()

    # Extract points and model names separately
    points_only = selected_data[['Price', 'Engine capacity', 'KM driven']].values.tolist()
    model_names = selected_data['Model Name'].values.tolist()
    point_to_model = {tuple(points_only[i]): model_names[i] for i in range(len(points_only))}

    # Δημιουργία, χρονομέτρηση και εκτύπωση του KD-Tree
    kd_points = points_only
    start_time = time.perf_counter()
    custom_tree = KDTreeCustom(kd_points)
    end_time = time.perf_counter()
    build_duration = end_time - start_time
    print("\n Εκτύπωση KD-Tree:")
    custom_tree.print_tree()
    print(f"Build time for 3D KD-Tree: {build_duration:.4f} seconds")


    # Ορισμός των ορίων για το range query
    print("Enter your 3D range query:")
    x_range = tuple(map(int, input("Price range (min max): ").split()))
    y_range = tuple(map(int, input("Engine capacity range (min max): ").split()))
    z_range = tuple(map(int, input("KM driven range (min max): ").split()))
    ranges = [x_range, y_range, z_range]

    # Εκτέλεση και χρονομέτρηση του range query 
    results = []
    start_time = time.perf_counter()
    custom_tree.range_query(custom_tree.node, ranges, results)
    end_time = time.perf_counter()
    query_time = (end_time - start_time) * 1000  # in ms
    print(f"Query time for KD-Tree: {query_time:.4f} seconds")

    # Εκτύπωση αποτελεσμάτων
    print("\n Αποτελέσματα Range Query:")
    for point in results:
        model = point_to_model.get(tuple(point), "Άγνωστο")
        print(f"Model: {model} | Price: {point[0]} | Engine: {point[1]} | KM: {point[2]}")


    # Εκτέλεση kNN query για 3 πλησιέστερα σημεία
    print("\n\nEnter your 3D point for k-NN search:")
    x_point = int(input("\nEnter price (x-axis): "))
    y_point = int(input("\nEnter engine capacity (y-axis): "))
    z_point = int(input("\nEnter KM driven (z-axis): "))
    query = [x_point, y_point, z_point]
    knn_results = knn_search(custom_tree, query, k=3)

    # Εκτύπωση αποτελεσμάτων
    print("\nΤα 3 πλησιέστερα αυτοκίνητα:")
    for p in knn_results:
        model = point_to_model.get(tuple(p), "Άγνωστο")
        print(f"Model: {model} | Price: {p[0]} | Engine: {p[1]} | KM: {p[2]}")

