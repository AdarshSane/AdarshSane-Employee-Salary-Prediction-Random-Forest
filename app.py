import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("Salary Data.csv")
df = df.dropna()

# Label encode categorical columns
df_encoded = df.copy()
le_gender = LabelEncoder()
le_edu = LabelEncoder()
le_job = LabelEncoder()

df_encoded["Gender"] = le_gender.fit_transform(df_encoded["Gender"])
df_encoded["Education Level"] = le_edu.fit_transform(df_encoded["Education Level"])
df_encoded["Job Title"] = le_job.fit_transform(df_encoded["Job Title"])

X = df_encoded.drop("Salary", axis=1)
y = df_encoded["Salary"]

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X, y)

# Streamlit App UI
st.title("Employee Salary Prediction")

st.markdown("Enter employee details to estimate their salary:")

age = st.slider("Age", 18, 65, 30)
gender = st.selectbox("Gender", df["Gender"].unique())
education = st.selectbox("Education Level", df["Education Level"].unique())
job = st.selectbox("Job Title", df["Job Title"].unique())
experience = st.slider("Years of Experience", 0, 40, 5)

# Encode user input
input_data = pd.DataFrame({
    "Age": [age],
    "Gender": le_gender.transform([gender]),
    "Education Level": le_edu.transform([education]),
    "Job Title": le_job.transform([job]),
    "Years of Experience": [experience]
})

# Predict
predicted_salary = model.predict(input_data)[0]

st.subheader("Predicted Salary:")
st.success(f"${predicted_salary:,.2f}")
