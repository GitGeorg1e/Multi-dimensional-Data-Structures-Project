import pandas as pd
import numpy as np
from collections import defaultdict
import time


class MBR:
    def __init__(self, x_range, y_range, z_range):
        self.x_range = x_range  # [x_min, x_max]
        self.y_range = y_range  # [y_min, y_max]
        self.z_range = z_range  # [z_min, z_max]

    def volume(self):
        dx = self.x_range[1] - self.x_range[0]
        dy = self.y_range[1] - self.y_range[0]
        dz = self.z_range[1] - self.z_range[0]
        return dx * dy * dz

    def enlarged_volume(self, other):
        x_min = min(self.x_range[0], other.x_range[0])
        x_max = max(self.x_range[1], other.x_range[1])
        y_min = min(self.y_range[0], other.y_range[0])
        y_max = max(self.y_range[1], other.y_range[1])
        z_min = min(self.z_range[0], other.z_range[0])
        z_max = max(self.z_range[1], other.z_range[1])
        return (x_max - x_min) * (y_max - y_min) * (z_max - z_min)

    def intersects(self, other):
        return (
            self.x_range[0] <= other.x_range[1] and self.x_range[1] >= other.x_range[0] and
            self.y_range[0] <= other.y_range[1] and self.y_range[1] >= other.y_range[0] and
            self.z_range[0] <= other.z_range[1] and self.z_range[1] >= other.z_range[0]
        )

def create_mbr_from_point(x, y, z):
    return MBR([x, x], [y, y], [z, z])


def create_mbr_from_mbrs(mbrs):
    x_min = min(m.x_range[0] for m in mbrs)
    x_max = max(m.x_range[1] for m in mbrs)
    y_min = min(m.y_range[0] for m in mbrs)
    y_max = max(m.y_range[1] for m in mbrs)
    z_min = min(m.z_range[0] for m in mbrs)
    z_max = max(m.z_range[1] for m in mbrs)
    return MBR([x_min, x_max], [y_min, y_max], [z_min, z_max])

class RTreeNode:
    def __init__(self, is_leaf=True):
        self.entries = []     # List of (MBR, object) for leaf; (MBR, child_node) for internal
        self.is_leaf = is_leaf
        self.parent = None    # Optional, could help with upward updates
        self.mbr = None       # Will be computed as entries are added

    def update_mbr(self):
        if not self.entries:
            self.mbr = None
            return
        # Extract MBRs from entries and create the bounding MBR
        mbrs = [entry[0] for entry in self.entries]
        self.mbr = create_mbr_from_mbrs(mbrs)

    def add_entry(self, mbr, obj_or_child):
        self.entries.append((mbr, obj_or_child))
        self.update_mbr()
         # Set parent pointer if it's a child node
        if isinstance(obj_or_child, RTreeNode):
            obj_or_child.parent = self


class RTree:
    def __init__(self, max_entries=4):
        self.max_entries = max_entries
        self.root = RTreeNode(is_leaf=True)

        
    def insert(self, point):
        mbr = create_mbr_from_point(point["price"], point["engine"], point["km"])  # (x, y, z)
        leaf = self.choose_leaf(self.root, mbr)
        leaf.add_entry(mbr, point)

        if len(leaf.entries) > self.max_entries:
            self.split_node(leaf)

    
    def choose_leaf(self, node, mbr):
        if node.is_leaf:
            return node

        # Choose child with least enlargement
        best_child = None
        min_enlargement = float('inf')

        for child_mbr, child_node in node.entries:
            enlargement = child_mbr.enlarged_volume(mbr) - child_mbr.volume()
            if enlargement < min_enlargement:
                min_enlargement = enlargement
                best_child = child_node

        return self.choose_leaf(best_child, mbr)
    

    def split_node(self, node):
        # Step 1: Pick seeds using max normalized separation
        def get_seeds(entries):
            best_axis = None
            best_separation = -1
            best_pair = (0, 1)

            for dim in ['x_range', 'y_range', 'z_range']:
                lows = [getattr(mbr, dim)[0] for mbr, _ in entries]
                highs = [getattr(mbr, dim)[1] for mbr, _ in entries]
                min_low = min(lows)
                max_high = max(highs)
                span = max_high - min_low
                if span == 0:
                    continue

                sorted_entries = sorted(enumerate(entries), key=lambda e: getattr(e[1][0], dim)[0])
                i_min = sorted_entries[0][0]
                i_max = sorted_entries[-1][0]

                separation = abs(lows[i_max] - highs[i_min]) / span
                if separation > best_separation:
                    best_separation = separation
                    best_pair = (i_min, i_max)
                    best_axis = dim

            return best_pair

        # Step 2: Prepare entries
        entries = node.entries.copy()
        i1, i2 = get_seeds(entries)
        group1_entries = [entries[i1]]
        group2_entries = [entries[i2]]

        # Remove seeds from list
        for i in sorted([i1, i2], reverse=True):
            entries.pop(i)

        group1_node = RTreeNode(is_leaf=node.is_leaf)
        group2_node = RTreeNode(is_leaf=node.is_leaf)
        group1_node.add_entry(*group1_entries[0])
        group2_node.add_entry(*group2_entries[0])

        # Step 3: Distribute remaining
        for mbr, obj in entries:
            v1 = group1_node.mbr.enlarged_volume(mbr) - group1_node.mbr.volume()
            v2 = group2_node.mbr.enlarged_volume(mbr) - group2_node.mbr.volume()
            if v1 < v2:
                group1_node.add_entry(mbr, obj)
            else:
                group2_node.add_entry(mbr, obj)

        # Step 4: Handle parent
        if node == self.root:
            # Create new root
            new_root = RTreeNode(is_leaf=False)
            new_root.add_entry(group1_node.mbr, group1_node)
            new_root.add_entry(group2_node.mbr, group2_node)
            self.root = new_root
        else:
            parent = node.parent

            # Safely remove the old child entry
            for i, (mbr, child) in enumerate(parent.entries):
                if child is node:
                    del parent.entries[i]
                    break



            parent.add_entry(group1_node.mbr, group1_node)
            parent.add_entry(group2_node.mbr, group2_node)
            if len(parent.entries) > self.max_entries:
                self.split_node(parent)


# Functions for "evaluation.py"
def build_rtree_from_df(df):
    """
    Accepts a pandas DataFrame with columns: Model Name, Price, Engine capacity, KM driven
    and builds an R-tree with the proper format.
    """
    tree = RTree(max_entries=4)
    start = time.perf_counter()

    for _, row in df.iterrows():
        point = {
            "model": row["Model Name"],
            "price": int(row["Price"]),
            "engine": int(row["Engine capacity"]),
            "km": int(row["KM driven"])
        }
        tree.insert(point)

    end = time.perf_counter()
    build_duration = end - start
    print(f"Build time for R-Tree: {build_duration:.4f} seconds")
    return tree, end - start


def execute_rtree_query(tree, x_range, y_range, z_range):
    """
    Executes a range query on an existing R-tree.
    Returns the matching points and query time in milliseconds.
    """
    results = []
    query_mbr = MBR(x_range, y_range, z_range)

    def search(node):
        if not node.mbr or not node.mbr.intersects(query_mbr):
            return
        if node.is_leaf:
            for mbr, point in node.entries:
                x, y, z = point["price"], point["engine"], point["km"]
                if (x_range[0] <= x <= x_range[1] and
                    y_range[0] <= y <= y_range[1] and
                    z_range[0] <= z <= z_range[1]):
                    results.append(point)
        else:
            for _, child in node.entries:
                search(child)

    start = time.perf_counter()
    search(tree.root)
    end = time.perf_counter()
    return results, (end - start) * 1000                


if __name__ == "__main__":

    # Load the cars24data.csv file
    df = pd.read_csv("../cars24data.csv")

    # Extract only necessary columns (drop missing values)
    df = df[['Model Name', 'Price', 'Engine capacity', 'KM driven']].dropna()
    df[['Price', 'Engine capacity', 'KM driven']] = df[['Price', 'Engine capacity', 'KM driven']].astype(int)

    # Initialize the R-tree
    tree = RTree(max_entries=4)

    # Insert each row as a 3D point into the R-tree
    for _, row in df.iterrows():
        point = {
            "model": row["Model Name"],
            "price": row["Price"],
            "engine": row["Engine capacity"],
            "km": row["KM driven"]
   }

        tree.insert(point)

    print("Inserted", len(df), "points into the R-tree.")

    # Diagnostic functions
    def print_tree_stats(tree):
        def count_nodes(node):
            if node.is_leaf:
                return 1
            return 1 + sum(count_nodes(child) for _, child in node.entries)

        def max_depth(node):
            if node.is_leaf:
                return 1
            return 1 + max(max_depth(child) for _, child in node.entries)

        total_nodes = count_nodes(tree.root)
        depth = max_depth(tree.root)
        print(f"Tree depth: {depth}")
        print(f"Total nodes: {total_nodes}")
        print(f"Root has {len(tree.root.entries)} entries")

    # Run the checks
    print_tree_stats(tree)
    def get_range_input(label):
        raw = input(f"Enter {label} range: ").strip().split()
        if len(raw) != 2:
            raise ValueError("You must enter two numbers separated by space.")
        return [int(raw[0]), int(raw[1])]


    print("\nDefine your 3D range query:")

    query_x = get_range_input("Price (X)")
    query_y = get_range_input("Engine (Y)")
    query_z = get_range_input("KM driven (Z)")


    def range_query(node, x_range, y_range, z_range):
        results = []

        query_mbr = MBR(x_range, y_range, z_range)

        def search(n):
            if not n.mbr.intersects(query_mbr):
                return  # Prune this branch

            if n.is_leaf:
                for mbr, point in n.entries:
                    x = point["price"]
                    y = point["engine"]
                    z = point["km"]

                    if x_range[0] <= x <= x_range[1] and \
                    y_range[0] <= y <= y_range[1] and \
                    z_range[0] <= z <= z_range[1]:
                        results.append(point)
            else:
                for _, child in n.entries:
                    search(child)

        search(node)
        return results


    matching_points = range_query(tree.root, query_x, query_y, query_z)

    print(f"\n🔎 Found {len(matching_points)} matching points:")
    for pt in matching_points[:10]:  # Show first 10
        print(f"  → {pt['model']}: [{pt['price']}, {pt['engine']}, {pt['km']}]")




'''def print_some_leaf_points(tree, limit=10):
    count = 0
    def visit(node):
        nonlocal count
        if node.is_leaf:
            for mbr, point in node.entries:
                print("Point:", point)
                count += 1
                if count >= limit:
                    return
            return
        for _, child in node.entries:
            visit(child)
            if count >= limit:
                return

    visit(tree.root)'''


'''def print_tree_structure(tree):
    def visit(node, level):
        indent = "  " * level
        print(f"{indent}Level {level} | Leaf: {node.is_leaf} | Entries: {len(node.entries)}")
        print(f"{indent}MBR: {node.mbr.x_range}, {node.mbr.y_range}, {node.mbr.z_range}")
        if not node.is_leaf:
            for mbr, child in node.entries:
                visit(child, level + 1)
    visit(tree.root, 0)'''



'''print_some_leaf_points(tree)'''
'''print_tree_structure(tree)'''

'''def count_nodes_by_level(tree):
    level_counts = defaultdict(int)

    def visit(node, level):
        level_counts[level] += 1
        if not node.is_leaf:
            for _, child in node.entries:
                visit(child, level + 1)

    visit(tree.root, 0)

    print("Node counts by level:")
    for level in sorted(level_counts):
        print(f"Level {level}: {level_counts[level]} node(s)")


count_nodes_by_level(tree)'''



'''def print_mbrs_at_levels(tree, levels=(0, 1)):
    def visit(node, level):
        if level in levels:
            print(f"\nLevel {level} | Leaf: {node.is_leaf} | Entries: {len(node.entries)}")
            print(f"  MBR of this node: X={node.mbr.x_range}, Y={node.mbr.y_range}, Z={node.mbr.z_range}")
        if level + 1 in levels and not node.is_leaf:
            for mbr, child in node.entries:
                print(f"  ├─ Child MBR: X={mbr.x_range}, Y={mbr.y_range}, Z={mbr.z_range}")
        if not node.is_leaf:
            for _, child in node.entries:
                visit(child, level + 1)

    visit(tree.root, 0)

print_mbrs_at_levels(tree, levels=(0, 1))'''

'''def print_leaf_contents(tree, limit=5):
    count = 0
    def visit(node, level):
        nonlocal count
        if node.is_leaf:
            print(f"\nLevel {level} | Leaf: {node.is_leaf} | Entries: {len(node.entries)}")
            print(f"MBR: X={node.mbr.x_range}, Y={node.mbr.y_range}, Z={node.mbr.z_range}")
            for mbr, point in node.entries:
                print("  ↳ Point:", point)
            count += 1
            if count >= limit:
                return
        else:
            for _, child in node.entries:
                if count >= limit:
                    return
                visit(child, level + 1)

    visit(tree.root, 0)

print_leaf_contents(tree, limit=5) '''

'''def show_level1_children_targets(tree):
    root = tree.root
    print(f"🔝 Root Node MBR:\n  X={root.mbr.x_range}, Y={root.mbr.y_range}, Z={root.mbr.z_range}")
    print(f"Root has {len(root.entries)} children:\n")

    for i, (mbr, level1_node) in enumerate(root.entries):
        print(f"🔷 Level 1 Node {i+1}")
        print(f"  MBR: X={mbr.x_range}, Y={mbr.y_range}, Z={mbr.z_range}")
        print(f"  Is Leaf: {level1_node.is_leaf} | Entries: {len(level1_node.entries)}")
        print("  └── Points to:")

        for j, (child_mbr, child_obj) in enumerate(level1_node.entries):
            print(f"      • Entry {j+1}:")
            print(f"         MBR: X={child_mbr.x_range}, Y={child_mbr.y_range}, Z={child_mbr.z_range}")
            if isinstance(child_obj, RTreeNode):
                print(f"         → Node | Leaf: {child_obj.is_leaf} | Entries: {len(child_obj.entries)}")
            else:
                print(f"         → Point: {child_obj}")  # In case it's a leaf with data points
        print()

show_level1_children_targets(tree)'''



'''query_x = [200000, 500000]
query_y = [1000, 1300]
query_z = [0, 60000]'''