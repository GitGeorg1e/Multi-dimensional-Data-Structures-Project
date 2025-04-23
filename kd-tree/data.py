# Re-import necessary modules after state reset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.spatial import KDTree
import heapq
import math
# Re-import necessary modules after state reset


# Load the CSV file directly
df = pd.read_csv("cars24data.csv")

# Select the relevant columns
columns = ['Price', 'Engine capacity', 'KM driven', 'Model Name']
selected_data = df[columns].dropna()

# Extract points and model names separately
points_only = selected_data[['Price', 'Engine capacity', 'KM driven']].values.tolist()
model_names = selected_data['Model Name'].values.tolist()
point_to_model = {tuple(points_only[i]): model_names[i] for i in range(len(points_only))}

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


# Δημιουργία και εκτύπωση του KD-Tree
kd_points = points_only
custom_tree = KDTreeCustom(kd_points)
print("\n Εκτύπωση KD-Tree:")
custom_tree.print_tree()

# Ορισμός των ορίων για το range query
ranges = [
    (500000, 700000),   # Price
    (1000, 1300),       # Engine capacity
    (20000, 60000)      # KM Driven
]

# Εκτέλεση του range query
results = []
custom_tree.range_query(custom_tree.node, ranges, results)

# Εκτύπωση αποτελεσμάτων
print("\n Αποτελέσματα Range Query:")
for point in results:
    model = point_to_model.get(tuple(point), "Άγνωστο")
    print(f"Model: {model} | Price: {point[0]} | Engine: {point[1]} | KM: {point[2]}")



# Εκτέλεση kNN query για 3 πλησιέστερα σημεία
query = [580000, 1197, 40000]
knn_results = knn_search(custom_tree, query, k=3)

# Εκτύπωση αποτελεσμάτων
print("\n 🔍 Τα 3 πλησιέστερα αυτοκίνητα:")
for p in knn_results:
    model = point_to_model.get(tuple(p), "Άγνωστο")
    print(f"Model: {model} | Price: {p[0]} | Engine: {p[1]} | KM: {p[2]}")

