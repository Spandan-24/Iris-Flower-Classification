# Iris-Flower-Classification
This project implements a machine learning–based Iris flower classification system using the K-Nearest Neighbors (KNN) algorithm. The model classifies Iris flowers into Setosa, Versicolor, and Virginica based on four physical features: sepal length, sepal width, petal length, and petal width.

🌸 Iris Flower Classification using KNN | Streamlit Web App
📌 Project Description

This project develops an Iris flower classification model using supervised machine learning. The algorithm applied is K-Nearest Neighbors (KNN), which predicts the species—Setosa, Versicolor, or Virginica—based on four morphological attributes: sepal length, sepal width, petal length, and petal width.
The project follows a complete end-to-end machine learning workflow, including data preprocessing, model training, evaluation, and deployment as an interactive Streamlit web application.


🚀 Key Features

Uses the Iris dataset for supervised classification
Implements KNN algorithm with feature scaling
80:20 train-test split for model evaluation
Achieves ~97% classification accuracy
Displays confusion matrix for performance analysis
Provides real-time prediction using slider-based inputs
Deployed as a Streamlit web application
User-friendly and interactive interface
Cloud-deployable and shareable via URL


🧠 Machine Learning Workflow

Load and explore the Iris dataset
Split data into training and testing sets
Apply feature scaling using StandardScaler
Train KNN classifier with optimal K value
Evaluate model using accuracy and confusion matrix
Save trained model and scaler
Deploy model using Streamlit for real-time predictions

📊 Dataset Information

Dataset: Iris Flower Dataset
Total Samples: 150

Features:
Sepal Length (cm)
Sepal Width (cm)
Petal Length (cm)
Petal Width (cm)

Classes :
Setosa
Versicolor
Virginica


🌐 Web Application

The Streamlit application allows users to:
Adjust flower measurements using sliders
Instantly view predicted Iris species
See prediction probabilities
Visualize the confusion matrix

🛠️ Technologies Used

Python
Scikit-learn
NumPy
Pandas
Matplotlib
Seaborn
Streamlit
Joblib

📁 Project Structure
IRIS FLOWER PROJECT
│
├── app.py
├── knn_iris_model.pkl
├── scaler.pkl
├── requirements.txt
└── README.md

▶️ How to Run Locally
pip install -r requirements.txt
streamlit run app.py


👨‍💻 Author

Developed by SPANDAN KARFA
Machine Learning & Data Science Enthusiast
