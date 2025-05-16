import pandas as pd

class Node:
    # Constructor to create a new node
    def __init__(self, attributes):
        self.attributes = [attributes[1], attributes[2], attributes[3]]
        self.model_name = attributes[0]
        self.OOO = None
        self.OOI = None
        self.OIO = None
        self.OII = None
        self.IOO = None
        self.IOI = None
        self.IIO = None
        self.III = None


# The function to insert a car in the Octree
def InsertCar(node, attributes):

    # If the tree is empty, set the root node
    if node is None:
        return Node(attributes)

    # Otherwise, recur down the tree

    # Compare the attributes to find the right slot in the tree
    if ( (attributes[1] < node.attributes[1]) and (attributes[2] < node.attributes[2]) and (attributes[3] < node.attributs[3]) ) :
        node.OOO = InsertCar(node.OOO, attributes)

    elif ( (attributes[1] < node.attributes[1]) and (attributes[2] < node.attributes[2]) and (attributes[3] >= node.attributs[3]) ) :
        node.OOI = InsertCar(node.OOI, attributes)

    elif ( (attributes[1] < node.attributes[1]) and (attributes[2] >= node.attributes[2]) and (attributes[3] < node.attributs[3]) ) :
        node.OIO = InsertCar(node.OIO, attributes)

    elif ( (attributes[1] < node.attributes[1]) and (attributes[2] >= node.attributes[2]) and (attributes[3] >= node.attributs[3]) ) :
        node.OII = InsertCar(node.OII, attributes)

    elif ( (attributes[1] >= node.attributes[1]) and (attributes[2] < node.attributes[2]) and (attributes[3] < node.attributs[3]) ) :
        node.IOO = InsertCar(node.IOO, attributes)

    elif ( (attributes[1] >= node.attributes[1]) and (attributes[2] < node.attributes[2]) and (attributes[3] >= node.attributs[3]) ) :
        node.IOI = InsertCar(node.IOI, attributes)

    elif ( (attributes[1] >= node.attributes[1]) and (attributes[2] >= node.attributes[2]) and (attributes[3] < node.attributs[3]) ) :
        node.IIO = InsertCar(node.IIO, attributes)

    elif ( (attributes[1] >= node.attributes[1]) and (attributes[2] >= node.attributes[2]) and (attributes[3] >= node.attributs[3]) ) :
        node.III = InsertCar(node.III, attributes)

    return node


# The function for the range search in the 3-D tree
def RangeSearchKD(node, depth, left_l,right_l,awards_th,left_db,right_db):


    if node is None:
        return None

     # Calculate the current dimension (curr_dim) of comparison. We have three dimensions: 0,1,2.

    curr_dim = depth % 3


       #Check if the node's/scientist's attributes are in range
    if (node.attributes[0][0] >= left_l) and (node.attributes[0][0] <= right_l) and (int(node.attributes[1]) > awards_th) and (int(node.attributes[2]) >= left_db) and (int(node.attributes[2]) <= right_db):
        ScientistsInRange.append(node.attributes)


    # If we are on the first dimension (curr_dim = 0) we check the surname to see if the searching should continue
    if (curr_dim == 0):
            if left_l < node.attributes[0][0]:
               RangeSearchKD(node.left,depth+1,left_l,right_l,awards_th,left_db,right_db)

            if right_l >= node.attributes[0][0]:
                RangeSearchKD(node.right, depth + 1, left_l, right_l, awards_th, left_db, right_db)


    # If we are on the second dimension (curr_dim = 1) we check the number of awards to see if the searching should continue
    if (curr_dim == 1):
         if node.attributes[1] > awards_th + 1 :
             RangeSearchKD(node.left, depth + 1, left_l, right_l, awards_th, left_db, right_db)

         #In this case, the searching always continues at the right sub-tree.
         RangeSearchKD(node.right, depth + 1, left_l, right_l, awards_th, left_db, right_db)


    # If we are on the third dimension (curr_dim = 2) we check the DBLP record to see if the searching should continue
    if (curr_dim == 2):
        if left_db < node.attributes[2]:
            RangeSearchKD(node.left, depth + 1, left_l, right_l, awards_th, left_db, right_db)

        if right_db >= node.attributes[2]:
            RangeSearchKD(node.right, depth + 1, left_l, right_l, awards_th, left_db, right_db)


    return None


# Main function
if __name__ == '__main__':

 csv_path = "../cars24data.csv"

 # Load the CSV
 df = pd.read_csv(csv_path)

 # Keep only the four columns of interest
 selected = df[['Model Name', 'Price', 'Engine capacity', 'KM driven']].dropna()

 # Convert each row to a Python list and gather them in `cars`
 cars = selected.values.tolist()

