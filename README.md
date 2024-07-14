# 3D Expression Recognition Project

This project leverages advanced machine learning techniques to analyze 3D facial landmarks, utilizing three classic classifiers: Random Forest (RF), Support Vector Machine (SVM), and Decision Tree (TREE). Our goal is to evaluate the effectiveness of these classifiers across various data manipulations—original, translated, and rotated 3D landmarks—to produce detailed reports on their performance metrics.

## Overview

The 3D Expression Recognition Project is designed to understand and interpret human expressions through 3D facial landmark analysis. By employing machine learning classifiers on different manipulations of 3D data, we aim to uncover the nuances in facial expressions that differentiate one emotion from another.

## Dependencies

Ensure you have Python 3.8 or later installed, along with the following packages:

- `NumPy`: A fundamental package for numerical computation in Python. It is used for handling arrays and matrices, which are essential for storing and processing the 3D landmark data.
- `SciPy`: A Python library used for scientific and technical computing. It offers modules for optimization, integration, interpolation, and other tasks useful in data processing and analysis.
- `scikit-learn`: A powerful tool for machine learning in Python. It provides simple and efficient tools for data mining and data analysis, including the classifiers used in this project.
- `matplotlib`: A plotting library for the Python programming language and its numerical mathematics extension, NumPy. It is used here to generate 3D scatter plots of the facial landmarks for visual analysis.

Install these using pip:

```bash
pip install numpy scipy scikit-learn matplotlib
```

## Project Structure

- `classify.py`: Implements cross-validation and prints evaluation metrics.
- `evaluate.py`: Evaluates classifiers with cross-validation, handling classifier initialization and metric calculations.
- `landmarks.py`: Provides functions for translating and rotating landmarks in the dataset.
- `plot.py`: Utilizes matplotlib for plotting data in 3D scatter plots.
- `process_data.py`: Processes data from the CK+ dataset, preparing it for classification by transforming landmarks based on input data type.
- `Project1.py`: Main entry point, orchestrating the data processing and evaluation pipeline.
- `README.md`: This file, providing an overview and instructions for the project.

## Setup and Usage

1. Clone or download this repository.
2. Install the required dependencies as listed.
3. To run the project, navigate to the project directory and execute:

```bash
python Project1.py <classifier> <data_type> <data_directory>
```

- `<classifier>`: Choose between `RF`, `SVM`, or `TREE`.
- `<data_type>`: Specify `Original`, `Translated`, or `Rotated`.
- `<data_directory>`: Path to the dataset.

Example command:

```bash
python Project1.py TREE Original ./data_directory
```

## Experiments
The project conducts three main experiments to assess classifier performance:

- **Raw 3D Landmarks**: Utilizes the original 3D landmarks without any manipulation.
- **Translated 3D Landmarks**: Translates the landmarks to the origin, centralizing the data.
- **Rotated 3D Landmarks**: Rotates the landmarks around the x, y, and z axes to test classifier robustness to orientation changes.

Each experiment utilizes 10-fold cross-validation to ensure the reliability of the performance metrics—accuracy, precision, recall, and confusion matrices. The insights derived from these experiments are crucial for understanding the strengths and limitations of each classifier in the context of 3D expression recognition.

## Performance Insights

Based on extensive experimentation, the Random Forest classifier consistently outperforms SVM and Decision Trees across different data types due to its ability to manage high-dimensional data and reduce overfitting through ensemble learning.

For each data type, here's a brief insight into the classifier performance:
- **Original Data**: Best handled by Random Forest, benefiting from the complexity and dimensionality of the data.
- **Translated Data**: Slight performance drop for all classifiers, with Random Forest still leading.
- **Rotated Data**: Random Forest excels, demonstrating its robustness to variations in data orientation.

Detailed metrics such as accuracy, precision, and recall for each classifier and data type are provided in the `Project Report.pdf`.

## Additional Notes

It's recommended to use a virtual environment for installing dependencies to avoid conflicts with system-wide packages.

For more detailed insights into the project's background, methodology, and results, please refer to the included `Project Report.pdf`.
