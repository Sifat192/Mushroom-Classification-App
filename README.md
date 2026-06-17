# Mushroom Classification Web App

## Project Overview

This project is a Streamlit-based Machine Learning web application that predicts whether a mushroom is edible or poisonous using classification algorithms.

The application allows users to compare the performance of three machine learning models:

* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest

Users can tune model hyperparameters and visualize performance metrics through interactive plots.

---

## Features

* Interactive Streamlit interface
* Model selection and comparison
* Hyperparameter tuning
* Accuracy, Precision, and Recall evaluation
* Confusion Matrix visualization
* ROC Curve visualization
* Precision-Recall Curve visualization

---

## Dataset

The project uses the Mushroom Dataset containing various mushroom characteristics such as:

* Cap shape
* Cap color
* Odor
* Gill size
* Stalk characteristics
* Habitat

Target Variable:

* Edible
* Poisonous

All categorical features are encoded using Label Encoding before model training.

---

## Technologies Used

* Python
* Streamlit
* NumPy
* Pandas
* Scikit-Learn
* Matplotlib

---

## Project Structure

Mushroom-Classification-App/

├── app.py

├── mushrooms.csv

├── requirements.txt

├── README.md

├── screenshots/

│ ├── home.png

│ ├── svm.png

│ ├── rf.png

│ └── metrics.png

└── notebook/

└── mushroom_classification.ipynb

---

## Installation

Clone the repository:

git clone https://github.com/yourusername/Mushroom-Classification-App.git

Navigate to project folder:

cd Mushroom-Classification-App

Install dependencies:

pip install -r requirements.txt

Run application:

streamlit run app.py

---

## Machine Learning Models

### Logistic Regression

A linear classification algorithm suitable for binary classification tasks.

### Support Vector Machine

Creates an optimal decision boundary for classification.

### Random Forest

An ensemble learning technique that combines multiple decision trees for improved prediction performance.

---

## Evaluation Metrics

The following metrics are used:

* Accuracy Score
* Precision Score
* Recall Score
* Confusion Matrix
* ROC Curve
* Precision-Recall Curve

---

## Results

Among the implemented models, Random Forest generally achieves the highest classification accuracy on the mushroom dataset due to its ability to capture complex feature interactions.

---

## Future Improvements

* Cross-validation
* Feature importance analysis
* Model export and deployment
* Additional classification algorithms
* Cloud deployment using Streamlit Community Cloud

---

## Author

Sifat Bhatia

Computer Science Engineering Student

Machine Learning Enthusiast
