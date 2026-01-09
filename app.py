import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix

# Load model and scaler
knn = joblib.load("knn_iris_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load dataset
iris = load_iris()

# App title
st.title("🌸 Iris Flower Classification using KNN")
st.write("Streamlit Web Application")

# Sidebar inputs
st.sidebar.header("Enter Flower Measurements")

sl = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sw = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
pl = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
pw = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Prepare input
sample = np.array([[sl, sw, pl, pw]])
sample_scaled = scaler.transform(sample)

# Prediction
prediction = knn.predict(sample_scaled)
probability = knn.predict_proba(sample_scaled)

# Output
st.subheader("🔍 Prediction Result")
st.success(f"Predicted Iris Flower: **{iris.target_names[prediction[0]]}**")

st.subheader("📊 Prediction Probability")
for i, flower in enumerate(iris.target_names):
    st.write(f"{flower}: {probability[0][i]:.2f}")

# Confusion matrix
st.subheader("📉 Confusion Matrix")
y_pred = knn.predict(scaler.transform(iris.data))
cm = confusion_matrix(iris.target, y_pred)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, cmap="Blues",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

st.markdown("---")
st.caption("KNN Iris Classification | Streamlit App")
