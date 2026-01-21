import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Employee Attrition Prediction System",
    layout="wide"
)

st.title("📉 Employee Attrition Prediction System")
st.caption("Internship Final Project – HR Analytics & Machine Learning")

# -------------------------------------------------
# DATASET UPLOAD
# -------------------------------------------------
st.markdown("### 📂 Upload HR Dataset")

uploaded_file = st.file_uploader(
    "Upload IBM HR Analytics Dataset (CSV format)",
    type="csv"
)

if uploaded_file is None:
    st.info("Please upload the HR Attrition dataset to proceed.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.markdown("### 🔍 Dataset Preview")
st.dataframe(df.head())

# -------------------------------------------------
# TARGET COLUMN
# -------------------------------------------------
TARGET_COLUMN = "Attrition"

if TARGET_COLUMN not in df.columns:
    st.error("❌ Dataset must contain an 'Attrition' column.")
    st.stop()

df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})

# -------------------------------------------------
# FEATURE SELECTION (HR RELEVANT ONLY)
# -------------------------------------------------
FEATURES = [
    "Age",
    "MonthlyIncome",
    "JobSatisfaction",
    "WorkLifeBalance",
    "YearsAtCompany",
    "OverTime",
    "Gender",
    "Department"
]

missing_features = [f for f in FEATURES if f not in df.columns]
if missing_features:
    st.error(f"❌ Missing required columns: {missing_features}")
    st.stop()

df = df[FEATURES + [TARGET_COLUMN]]

# -------------------------------------------------
# ENCODING
# -------------------------------------------------
df["OverTime"] = df["OverTime"].map({"Yes": 1, "No": 0})
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
df["Department"] = df["Department"].map({
    "Sales": 0,
    "Research & Development": 1,
    "Human Resources": 2
})

# -------------------------------------------------
# MODEL PIPELINE
# -------------------------------------------------
X = df[FEATURES]
y = df[TARGET_COLUMN]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

# -------------------------------------------------
# MODEL PERFORMANCE
# -------------------------------------------------
preds = model.predict(X_test)

st.markdown("### 📊 Model Performance")

col1, col2 = st.columns(2)
with col1:
    st.metric("Accuracy", round(accuracy_score(y_test, preds), 2))
with col2:
    st.metric("Recall", round(recall_score(y_test, preds), 2))

st.markdown(
    "> **Recall is prioritized** because missing an employee likely to leave is costlier than a false alert."
)

# -------------------------------------------------
# PREDICTION PANEL
# -------------------------------------------------
st.markdown("### 🔮 Predict Employee Attrition Risk")

c1, c2 = st.columns(2)

with c1:
    age = st.slider("Employee Age", 18, 60, 30)
    income = st.number_input("Monthly Income (₹)", 1000, 200000, 30000)
    years_company = st.slider("Years at Company", 0, 40, 5)
    job_sat = st.slider("Job Satisfaction (1 = Low, 4 = High)", 1, 4, 3)

with c2:
    work_life = st.slider("Work-Life Balance (1 = Poor, 4 = Excellent)", 1, 4, 3)
    overtime = st.selectbox("OverTime", ["Yes", "No"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    department = st.selectbox(
        "Department",
        ["Sales", "Research & Development", "Human Resources"]
    )

# Encode input
overtime_val = 1 if overtime == "Yes" else 0
gender_val = 1 if gender == "Male" else 0
dept_val = {
    "Sales": 0,
    "Research & Development": 1,
    "Human Resources": 2
}[department]

input_data = np.array([[
    age,
    income,
    job_sat,
    work_life,
    years_company,
    overtime_val,
    gender_val,
    dept_val
]])

input_scaled = scaler.transform(input_data)

if st.button("🚀 Predict Attrition Risk"):
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("⚠️ High Risk: Employee is likely to leave the organization.")
    else:
        st.success("✅ Low Risk: Employee is likely to stay with the organization.")
