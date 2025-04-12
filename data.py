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


# Convert to numpy array for KDTree
kd_data = selected_data.to_numpy()

# Build the KDTree
kd_tree = KDTree(kd_data)

# Display the processed DataFrame
# tools.display_dataframe_to_user(name="KDTree Input Data", dataframe=selected_data)  # Commented out as ace_tools is not available
print("KDTree Input Data:")
print(selected_data)  # Displaying the DataFrame using print as an alternative
