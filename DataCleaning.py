import zipfile
import os
import numpy as np
from math import acos

# defining the class labels and assigning them to a number
def getClassLabel(label):
    class_labels = {
        'Angry': 0,
        'Disgust': 1,
        'Fear': 2,
        'Happy': 3,
        'Sad': 4,
        'Surprise': 5
    }
    return class_labels.get(label)

def readData(path, data_type):
    # Initialize empty lists to store features, classes, and subjects
    features = []
    classes = []
    subjects = []

    # Check if path is a zip file
    if path.endswith('.zip'):
        # create a ZipFile object and extract all files to memory
        with zipfile.ZipFile(path) as myzip: #opening the zip file
            # Loop over all subject directories, expressions, and bnd files in the zip file
            for subject_dir in sorted(myzip.namelist()): # Iterating over subject directories
                for expression_dir in sorted(myzip.namelist()): # Iterating over expressions
                    for bnd_file in sorted(myzip.namelist()): # Iterating over bnd files
                        # Check if file has .bnd extension
                        if bnd_file.endswith('.bnd'): 
                            # Extract the class label i.e Angry, Sad, Happy etc
                            label = os.path.basename(os.path.dirname(bnd_file)) 
                            # Get the class label value
                            label_val = getClassLabel(label)
                            # Extract the subject
                            subject = os.path.basename(
                                os.path.dirname(os.path.dirname(bnd_file)))
                            # Extract the file to memory
                            bnd_data = myzip.read(bnd_file)
                            # Convert the bytes to a string
                            bnd_str = bnd_data.decode('utf-8')
                            # Initialize an empty list to store the landmarks
                            landmarks = []
                            # Extract the coordinates in string i.e x,y,z
                            # Splitting the string by new line and iterating over each line in BND File
                            for line in bnd_str.split("\n"): 
                                if len(line) > 0: # Checking if the line is not empty
                                    x, y, z = line.split()[1:] # Extracting x, y, z coordinates
                                    # Appending the coordinates to the landmarks list
                                    landmarks.append([float(x), float(y), float(z)])

                            # Check the data type and apply the necessary transformation
                            if data_type == "Original":
                                landmarks = np.array(landmarks).flatten() # Flattening the landmarks
                            elif data_type == "Translated":
                                landmarks = translateLandmarks(landmarks) # Translating the landmarks
                            elif data_type == "Rotated":
                                # Rotating the landmarks
                                landmarksRotatedX, landmarksRotatedY, landmarksRotatedZ = rotatedLandmarks(
                                    landmarks)
                                # assigning rotated landmarks in a list
                                landmarks = [landmarksRotatedX,
                                             landmarksRotatedY, landmarksRotatedZ]

                            # Append the features, class labels, and subject to their respective lists
                            features.append(landmarks)
                            classes.append(label_val)
                            subjects.append(subject)
    else:
        # Loop over all subject directories and bnd files in the given path
        for subdir, dirs, files in os.walk(path):
            for file in files: # Iterating over files in the directory
                # Check if file has .bnd extension
                if file.endswith('.bnd'):
                    # Extract the class label i.e Angry, Sad, Happy etc
                    label = subdir.split(os.path.sep)[-1]
                    # Get the class label value
                    label_val = getClassLabel(label)
                    # Extract the subject
                    subject = os.path.join(subdir, file)
                    # File path
                    # generating the file path
                    filepath = os.path.join(subdir, file)
                    # Open the file to read
                    with open(filepath, 'r') as f:
                        landmarks = [] # Initialize an empty list to store the landmarks
                        # Extract the lines from the file
                        bnd_data = f.readlines() # Reading the lines from the file
                        # Extract the coordinates in string i.e x,y,z
                        for line in bnd_data[:84]: # Iterating over the lines in the BND File
                            x, y, z = line.split()[1:] # Extracting x, y, z coordinates
                            landmarks.append([float(x), float(y), float(z)]) # Appending the coordinates to the landmarks list

                        # Check the data type and apply the necessary transformation
                        if data_type == "Original":
                            landmarks = np.array(landmarks).flatten() # Flattening the landmarks
                        elif data_type == "Translated":
                            landmarks = translateLandmarks(landmarks) # Translating the landmarks
                        elif data_type == "Rotated":
                            # Rotating the landmarks
                            landmarksRotatedX, landmarksRotatedY, landmarksRotatedZ = rotatedLandmarks(
                                landmarks)
                            # assigning rotated landmarks in a list
                            landmarks = [landmarksRotatedX,
                                         landmarksRotatedY, landmarksRotatedZ]

                        # Append the features, class labels, and subject to their respective lists
                        features.append(landmarks)
                        classes.append(label_val)
                        subjects.append(subject)
    # return the features, class labels, and subjects as NumPy arrays
    return features, classes, subjects

# Function to rotate the landmarks using numpy and the data read and stored in landmarks list using readData function
def rotatedLandmarks(landmarks):
    # Convert the list to a NumPy array
    landmarks = np.array(landmarks)
    # Calculate the value of pi using the arccosine function
    pi = round(2 * acos(0.0), 3)

    cos = np.cos(pi) # Calculate the cosine of pi
    sine = np.sin(pi) # Calculate the sine of pi

    # Respective Axis data for rotating the coordinates
    x_axis = np.array([[1, 0, 0], [0, cos, sine], [0, -sine, cos]]) # Rotation matrix around x-axis
    y_axis = np.array([[cos, 0, -sine], [0, 1, 0], [sine, 0, cos]]) # Rotation matrix around y-axis
    z_axis = np.array([[cos, sine, 0], [-sine, cos, 0], [0, 0, 1]]) # Rotation matrix around z-axis

    # Rotate the points around each axis
    rotated_x = x_axis.dot(landmarks.T).T # Rotating the landmarks around x-axis
    rotated_y = y_axis.dot(landmarks.T).T # Rotating the landmarks around y-axis
    rotated_z = z_axis.dot(landmarks.T).T # Rotating the landmarks around z-axis

    # Returning rotated landmarks
    return rotated_x.flatten(), rotated_y.flatten(), rotated_z.flatten()


def translateLandmarks(landmarks):
    # Convert landmarks to a NumPy array
    landmarks = np.array(landmarks)

    # Calculate the mean of each x, y, z
    mean_x = np.mean(landmarks[:, 0])
    mean_y = np.mean(landmarks[:, 1])
    mean_z = np.mean(landmarks[:, 2])

    # Subtract the mean from each x, y, z coordinate
    landmarks[:, 0] -= mean_x # Subtracting the mean from x coordinate
    landmarks[:, 1] -= mean_y # Subtracting the mean from y coordinate
    landmarks[:, 2] -= mean_z # Subtracting the mean from z coordinate

    # Return the translated flattened landmarks
    return landmarks.flatten()
