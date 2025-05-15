import pandas as pd

class Node:
    # Constructor to create a new node
    def __init__(self, attributes):
        self.attributes = attributes
        self.OOO = None
        self.OOI = None
        self.OIO = None
        self.OII = None
        self.IOO = None
        self.IOI = None
        self.II0 = None
        self.III = None

csv_path = "../cars24data.csv"

# Load the CSV
df = pd.read_csv(csv_path)

# Keep only the four columns of interest
selected = df[['Model Name', 'Price', 'Engine capacity', 'KM driven']].dropna()

# Convert each row to a Python list and gather them in `cars`
cars = selected.values.tolist()


