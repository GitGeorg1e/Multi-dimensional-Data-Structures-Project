import pandas as pd
import time


# The function to get the query from the user
def get_query(): 
  print('\n')
  print('-' * 60)
  print('User Query')
  left_pr = int(input('Give the minimum price: '))
  right_pr = int(input('Give the maximum price: '))
  left_ec = int(input('Give the minimum engine capacity: '))
  right_ec = int(input('Give the maximum engine capacity: '))
  left_km = int(input('Give the minimum kilometers driven: '))
  right_km = int(input('Give the maximum kilometers driven: '))
  print('-' * 60)
  print('\n')
  return left_pr, right_pr, left_ec, right_ec, left_km, right_km 



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
        self.IIO = None
        self.III = None



# The function to insert a car in the Octree
def InsertCar(node, attributes):

    # If the tree is empty, set the root node
    if node is None:
        return Node(attributes)

    # Otherwise, recur down the tree

    # Compare the attributes to find the right slot in the tree
    if ( (attributes[1] < node.attributes[1]) and (attributes[2] < node.attributes[2]) and (attributes[3] < node.attributes[3]) ) :
        node.OOO = InsertCar(node.OOO, attributes)

    elif ( (attributes[1] < node.attributes[1]) and (attributes[2] < node.attributes[2]) and (attributes[3] >= node.attributes[3]) ) :
        node.OOI = InsertCar(node.OOI, attributes)

    elif ( (attributes[1] < node.attributes[1]) and (attributes[2] >= node.attributes[2]) and (attributes[3] < node.attributes[3]) ) :
        node.OIO = InsertCar(node.OIO, attributes)

    elif ( (attributes[1] < node.attributes[1]) and (attributes[2] >= node.attributes[2]) and (attributes[3] >= node.attributes[3]) ) :
        node.OII = InsertCar(node.OII, attributes)

    elif ( (attributes[1] >= node.attributes[1]) and (attributes[2] < node.attributes[2]) and (attributes[3] < node.attributes[3]) ) :
        node.IOO = InsertCar(node.IOO, attributes)

    elif ( (attributes[1] >= node.attributes[1]) and (attributes[2] < node.attributes[2]) and (attributes[3] >= node.attributes[3]) ) :
        node.IOI = InsertCar(node.IOI, attributes)

    elif ( (attributes[1] >= node.attributes[1]) and (attributes[2] >= node.attributes[2]) and (attributes[3] < node.attributes[3]) ) :
        node.IIO = InsertCar(node.IIO, attributes)

    elif ( (attributes[1] >= node.attributes[1]) and (attributes[2] >= node.attributes[2]) and (attributes[3] >= node.attributes[3]) ) :
        node.III = InsertCar(node.III, attributes)

    return node




# The function for the range search in the tree
def RangeSearch(node, left_pr, right_pr, left_ec, right_ec, left_km, right_km):


    if node is None:
        return None

     
    #Check if the node's/cars's attributes are in range
    if (node.attributes[1] >= left_pr) and (node.attributes[1] <= right_pr) and (node.attributes[2] >= left_ec) and (node.attributes[2] <= right_ec) and (node.attributes[3] >= left_km) and (node.attributes[3] <= right_km):
        CarsInRange.append(node.attributes)
    
    #Check if there is no point to keep searching in a subtree
    if((left_pr < node.attributes[1]) and (left_ec < node.attributes[2]) and (left_km < node.attributes[3])):
         RangeSearch(node.OOO, left_pr, right_pr, left_ec, right_ec, left_km, right_km)

    if((left_pr < node.attributes[1]) and (left_ec < node.attributes[2]) and (right_km >= node.attributes[3])):
         RangeSearch(node.OOI, left_pr, right_pr, left_ec, right_ec, left_km, right_km)

    if((left_pr < node.attributes[1]) and (right_ec >= node.attributes[2]) and (left_km < node.attributes[3])):
         RangeSearch(node.OIO, left_pr, right_pr, left_ec, right_ec, left_km, right_km)

    if((left_pr < node.attributes[1]) and (right_ec >= node.attributes[2]) and (right_km >= node.attributes[3])):
         RangeSearch(node.OII, left_pr, right_pr, left_ec, right_ec, left_km, right_km)

    if((right_pr >= node.attributes[1]) and (left_ec < node.attributes[2]) and (left_km < node.attributes[3])):
         RangeSearch(node.IOO, left_pr, right_pr, left_ec, right_ec, left_km, right_km)

    if((right_pr >= node.attributes[1]) and (left_ec < node.attributes[2]) and (right_km >= node.attributes[3])):
         RangeSearch(node.IOI, left_pr, right_pr, left_ec, right_ec, left_km, right_km)

    if((right_pr >= node.attributes[1]) and (right_ec >= node.attributes[2]) and (left_km < node.attributes[3])):
         RangeSearch(node.IIO, left_pr, right_pr, left_ec, right_ec, left_km, right_km)

    if((right_pr >= node.attributes[1]) and (right_ec >= node.attributes[2]) and (right_km >= node.attributes[3])):
         RangeSearch(node.III, left_pr, right_pr, left_ec, right_ec, left_km, right_km)

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

creation_time=time.time() # Record the start time of the creation of the tree
# Insert the first car in the tree and set this object as the root node
root = None
root = InsertCar(root, cars[0])

# Insert the other cars in the tree
for car in cars[1:]:
    InsertCar(root, car)

creation_time_end=time.time()-creation_time
print(f"\nTotal construction time: {creation_time_end} seconds\n")

#User query
left_pr, right_pr, left_ec, right_ec, left_km, right_km = get_query()

start_time = time.time()  # Record the start time

# This array contains all the cars whose attributes are included in the given range
CarsInRange = []
RangeSearch(root, left_pr, right_pr, left_ec, right_ec, left_km, right_km)

end_time = time.time()  # Record the end time
search_time = end_time - start_time  # Calculate the total time for the search

print('\nRange search finished. Results:\n')

# Print the cars in range
for c in CarsInRange:
        print(c)

print(f"\nTotal search time: {search_time} seconds\n")


if len(CarsInRange) > 1:
        
       # Execute the LSH algorithm
        print('EKTELESI LSH')
        
        
        
else:
        print("\n We have only one or zero results. LSH was not executed! \n")
        print("RESULTS: \n")
        print(CarsInRange)
