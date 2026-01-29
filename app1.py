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
st.set_page_config(page_title="Employee Attrition Prediction System", layout="wide")

st.title("📉 Employee Attrition Prediction System")
st.caption("Internship / Final Year / Placement Project – HR Analytics & ML")

# -------------------------------------------------
# DATA UPLOAD
# -------------------------------------------------
uploaded_file = st.file_uploader("Upload IBM HR Attrition Dataset (CSV)", type="csv")

if uploaded_file is None:
    st.stop()

df = pd.read_csv(uploaded_file)

TARGET = "Attrition"
df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})

FEATURES = [
    "Age","MonthlyIncome","JobSatisfaction","WorkLifeBalance",
    "YearsAtCompany","OverTime","Gender","Department"
]

df = df[FEATURES + [TARGET]]

df["OverTime"] = df["OverTime"].map({"Yes": 1, "No": 0})
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
df["Department"] = df["Department"].map({
    "Sales": 0, "Research & Development": 1, "Human Resources": 2
})

X = df[FEATURES]
y = df[TARGET]

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURES)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# -------------------------------------------------
# MODEL METRICS
# -------------------------------------------------
preds = model.predict(X_test)

st.subheader("📊 Model Performance")
c1, c2 = st.columns(2)
c1.metric("Accuracy", round(accuracy_score(y_test, preds), 2))
c2.metric("Recall", round(recall_score(y_test, preds), 2))

# -------------------------------------------------
# USER INPUT
# -------------------------------------------------
st.subheader("🔮 Employee Details")

c1, c2 = st.columns(2)

with c1:
    age = st.slider("Age", 18, 60, 30)
    income = st.number_input("Monthly Income", 1000, 200000, 30000)
    years = st.slider("Years at Company", 0, 40, 5)
    job_sat = st.slider("Job Satisfaction", 1, 4, 3)

with c2:
    wlb = st.slider("Work Life Balance", 1, 4, 3)
    overtime = st.selectbox("OverTime", ["Yes","No"])
    gender = st.selectbox("Gender", ["Male","Female"])
    dept = st.selectbox("Department", ["Sales","Research & Development","Human Resources"])

base_input = pd.DataFrame([{
    "Age": age,
    "MonthlyIncome": income,
    "JobSatisfaction": job_sat,
    "WorkLifeBalance": wlb,
    "YearsAtCompany": years,
    "OverTime": 1 if overtime=="Yes" else 0,
    "Gender": 1 if gender=="Male" else 0,
    "Department": {"Sales":0,"Research & Development":1,"Human Resources":2}[dept]
}])

base_scaled = scaler.transform(base_input)

# -------------------------------------------------
# PREDICTION + SCENARIO ANALYSIS
# -------------------------------------------------
if st.button("🚀 Predict & Analyze"):

    base_prob = model.predict_proba(base_scaled)[0][1]

    st.markdown("---")
    st.subheader("🎯 Final Prediction")

    st.metric("Attrition Probability", f"{base_prob:.2%}")

    # ----------------------------
    # WHAT-IF SCENARIOS
    # ----------------------------
    scenarios = {
        "Base Case": base_prob,
        "Low Job Satisfaction": model.predict_proba(
            scaler.transform(base_input.assign(JobSatisfaction=1))
        )[0][1],
        "Poor Work-Life Balance": model.predict_proba(
            scaler.transform(base_input.assign(WorkLifeBalance=1))
        )[0][1],
        "High Income Increase": model.predict_proba(
            scaler.transform(base_input.assign(MonthlyIncome=income+20000))
        )[0][1],
        "Overtime Increased": model.predict_proba(
            scaler.transform(base_input.assign(OverTime=1))
        )[0][1],
        "More Years at Company": model.predict_proba(
            scaler.transform(base_input.assign(YearsAtCompany=years+5))
        )[0][1],
        "Department Change (R&D)": model.predict_proba(
            scaler.transform(base_input.assign(Department=1))
        )[0][1]
    }

    scenario_df = pd.DataFrame({
        "Scenario": scenarios.keys(),
        "Attrition Probability": scenarios.values()
    }).set_index("Scenario")

    # ----------------------------
    # VISUALIZATION 1: BAR
    # ----------------------------
    st.subheader("📊 Scenario-wise Attrition Probability")
    st.bar_chart(scenario_df)

    # ----------------------------
    # VISUALIZATION 2: LINE TREND
    # ----------------------------
    st.subheader("📈 Risk Trend Across Scenarios")
    st.line_chart(scenario_df)

    # ----------------------------
    # VISUALIZATION 3: TABLE
    # ----------------------------
    st.subheader("🧾 Scenario Comparison Table")
    st.dataframe(scenario_df.style.format("{:.2%}"))

    # ----------------------------
    # VISUALIZATION 4: FEATURE IMPORTANCE
    # ----------------------------
    st.subheader("📌 Feature Importance")
    fi = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.bar_chart(fi.set_index("Feature"))

    # ----------------------------
    # VISUALIZATION 5: RISK LEVEL
    # ----------------------------
    st.subheader("🚦 Risk Category")

    if base_prob > 0.65:
        st.error("🔴 HIGH RISK – Immediate HR intervention needed")
    elif base_prob > 0.35:
        st.warning("🟠 MEDIUM RISK – Monitor closely")
    else:
        st.success("🟢 LOW RISK – Employee likely to stay")
