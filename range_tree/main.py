import pandas as pd
import numpy as np
import time

class RangeTree1D:
    def __init__(self, points):
        self.points = sorted(points, key=lambda p: p[2]) # sort by z
        if not self.points:
            self.is_leaf = True
            self.left = self.right = self.value = None
            return

        if len(points) == 1:
            self.is_leaf = True
            self.value = points[0]
            self.left = self.right = None
        else:
            self.is_leaf = False
            mid = len(points) // 2
            self.value = self.points[mid]
            self.left = RangeTree1D(self.points[:mid])
            self.right = RangeTree1D(self.points[mid:])

class RangeTree2D:
    def __init__(self, points):
        self.tree = self._build(points)

    def _build(self, points):
        return self._build_recursive(sorted(points, key=lambda p: p[1]))  # sort by y

    def _build_recursive(self, points):
        if not points:
            return None
        if len(points) == 1:
            y = points[0][1]
            return {
                "value": points[0],
                "range": (y, y),
                "left": None,
                "right": None,
                "z_tree": RangeTree1D(points)
            }

        min_y = min(p[1] for p in points)
        max_y = max(p[1] for p in points)
        mid = len(points) // 2
        node = {
            "value": points[mid],
            "range": (min_y, max_y),
            "left": self._build_recursive(points[:mid]),
            "right": self._build_recursive(points[mid+1:]),
            "z_tree": RangeTree1D(points)
        }
        return node

class RangeTree3D:
    def __init__(self, points):
        self.tree = self._build(points)

    def _build(self, points):
        return self._build_recursive(sorted(points, key=lambda p: p[0]))  # sort by x

    def _build_recursive(self, points):
        if not points:
            return None
        if len(points) == 1:
            x = points[0][0]
            return {
                "value": points[0],
                "range": (x, x),
                "left": None,
                "right": None,
                "yz_tree": RangeTree2D(points)
            }

        min_x = min(p[0] for p in points)
        max_x = max(p[0] for p in points)
        mid = len(points) // 2
        node = {
            "value": points[mid],
            "range": (min_x, max_x),
            "left": self._build_recursive(points[:mid]),
            "right": self._build_recursive(points[mid+1:]),
            "yz_tree": RangeTree2D(points)  # the full subtree
        }
        return node
    

def range_query_1D(points, z_range):
    return [p for p in points if z_range[0] <= p[2] <= z_range[1]]


def range_query_2D(node, y_range, z_range):
    if node is None:
        return []

    y_min, y_max = node["range"]

    # Case 1: Canonical y-subtree — fully within y_range
    if y_range[0] <= y_min and y_max <= y_range[1]:
        return range_query_1D(node["z_tree"].points, z_range)

    # Case 2: No overlap with y_range — skip
    if y_range[1] < y_min or y_max < y_range[0]:
        return []

    # Case 3: Partial overlap — check value and recurse
    x, y, z = node["value"]
    results = []

    if y_range[0] <= y <= y_range[1] and z_range[0] <= z <= z_range[1]:
        results.append((x, y, z))

    results += range_query_2D(node["left"], y_range, z_range)
    results += range_query_2D(node["right"], y_range, z_range)

    return results



def range_query_3D(node, x_range, y_range, z_range):
    if node is None:
        return []

    x_min, x_max = node["range"]

    # Case 1: Canonical x-subtree → entire range inside x_range
    if x_range[0] <= x_min and x_max <= x_range[1]:
        return range_query_2D(node["yz_tree"].tree, y_range, z_range)

    # Case 2: No overlap with x_range
    if x_range[1] < x_min or x_max < x_range[0]:
        return []

    # Case 3: Partial overlap — check current node and recurse
    x, y, z = node["value"]
    results = []

    if x_range[0] <= x <= x_range[1] and y_range[0] <= y <= y_range[1] and z_range[0] <= z <= z_range[1]:
        results.append((x, y, z))

    results += range_query_3D(node["left"], x_range, y_range, z_range)
    results += range_query_3D(node["right"], x_range, y_range, z_range)

    return results


def build_range_tree(points):
    start_time = time.perf_counter()
    tree = RangeTree3D(points)
    end_time = time.perf_counter()
    build_duration = end_time - start_time
    print(f"Build time for 3D Range Tree: {build_duration:.4f} seconds")
    return tree, build_duration


def execute_range_query(tree, x_range, y_range, z_range):
    start_time = time.perf_counter()
    results = range_query_3D(tree.tree, x_range, y_range, z_range)
    end_time = time.perf_counter()
    query_duration = (end_time - start_time) * 1000  # milliseconds
    print(f"Query time: {query_duration:.2f} ms")
    return results, query_duration



if __name__ == "__main__":

    df = pd.read_csv("../cars24data.csv")

    df_full = df[["Model Name", "Price", "Engine capacity", "KM driven"]].dropna()
    columns = ["Price", "Engine capacity", "KM driven"]
    df_main = df[columns].dropna()

    # Convert to tuple list: [(year, price, km), ...]
    points = [tuple(int(v) for v in row) for row in df_main.to_numpy()]

    print("Building 3D range tree...")
    tree3D, build_time = build_range_tree(points)

    print("Enter your 3D range query:")
    x_range = tuple(map(int, input("Price range (min max): ").split()))
    y_range = tuple(map(int, input("Engine capacity range (min max): ").split()))
    z_range = tuple(map(int, input("KM driven range (min max): ").split()))


    print("Running 3D query...")
    results, query_time = execute_range_query(tree3D, x_range, y_range, z_range)

    lookup = {
    (int(row["Price"]), int(row["Engine capacity"]), int(row["KM driven"])): row["Model Name"]
    for _, row in df_full.iterrows() }

    print(f"Found {len(results)} matching cars.\nPrinting the first 10:")
    for r in results[:10]:
        model = lookup.get(r, "Unknown Model")
        print(f"{model:35} | Price: {r[0]}, Engine: {r[1]}, KM: {r[2]}")