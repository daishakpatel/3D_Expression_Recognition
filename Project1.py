import sys
import numpy as np
from classify import evaluate
from DataCleaning import readData
import random
import matplotlib.pyplot as plot


# Function to run the experiment
def runExperiment(classifier, data_type, data_directory):
    # Printing a separator for clarity
    print("------------------------------------------------------")
    # Print the classifier and data type for experiment
    print(classifier, data_type)

    # Read and transform the dataset
    X, y, subjects = readData(data_directory, data_type)
    # Convert the lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)
    # Convert the subjects list to a NumPy array
    subjects = np.array(subjects)

    # Print the shape of the dataset
    # Evaluating the model based on data type
    if data_type == "Original" or data_type == "Translated":
        # evaluate the model
        evaluate(X, y, classifier, data_type, subjects)

    # If the data type is Rotated
    elif data_type == "Rotated":
        X_rotated_x, X_rotated_y, X_rotated_z = X[:, 0], X[:, 1], X[:, 2]
        # for RotatedX
        evaluate(X_rotated_x, y, classifier, data_type+"X", subjects)
        # for RotatedY
        evaluate(X_rotated_y, y, classifier, data_type+"Y", subjects)
        # for RotatedZ
        evaluate(X_rotated_z, y, classifier, data_type+"Z", subjects)

    print("------------------------------------------------------")


if __name__ == "__main__":
    # Command Line Arguments 
    classifier = sys.argv[1]
    #classifier = "TREE"
    #data_type = "Translated"
    #data_directory = "./BU4DFE_BND_V1.1"
    data_type = sys.argv[2]
    data_directory = sys.argv[3]

    # Check if classifier argument is valid
    if classifier not in ["SVM", "RF", "TREE"]:
        print("Invalid classifier") # Print an error message
        sys.exit()

    # Check if data type argument is valid
    if data_type not in ["Original", "Translated", "Rotated"]:
        print("Invalid Datatype")
        sys.exit()

    # Run experiment
    runExperiment(classifier, data_type, data_directory)