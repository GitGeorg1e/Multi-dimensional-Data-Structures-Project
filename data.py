# Re-import necessary modules after state reset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.spatial import KDTree
# import ace_tools as tools  # Commented out as the module is not available

# Load the CSV file directly

df = pd.read_csv("cars24data.csv")


# Select the relevant columns
columns = ['Price', 'Engine capacity', 'KM driven']
selected_data = df[columns].dropna()
kd_points = selected_data.to_numpy().tolist()

# Convert to numpy array for KDTree
#kd_data = selected_data.to_numpy()

# Build the KDTree
#kd_tree = KDTree(kd_data)

# Display the processed DataFrame
# tools.display_dataframe_to_user(name="KDTree Input Data", dataframe=selected_data)  # Commented out as ace_tools is not available
#print("KDTree Input Data:")
#print(selected_data)  # Displaying the DataFrame using print as an alternative
#print(kd_data)


class KDNode:
    def __init__(self, point, axis, left=None, right=None):
        self.point = point
        self.axis = axis
        self.left = left
        self.right = right


# Fix the print_tree function to properly access KDNode attributes
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

    def print_tree(self, node=None, depth=0):
        if node is None:
            node = self.node
        if node is not None:
            print("  " * depth + f"Depth {depth}, Axis {node.axis}, Point {node.point}")
            if node.left and node.left.node:
                self.print_tree(node.left.node, depth + 1)
            if node.right and node.right.node:
                self.print_tree(node.right.node, depth + 1)


# Rebuild the custom k-d tree and print its structure
custom_tree = KDTreeCustom(kd_points)
custom_tree.print_tree()

