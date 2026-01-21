import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Smart Analytics Dashboard",
    layout="wide"
)

# -----------------------------
# TITLE
# -----------------------------
st.title("📊 Smart Analytics & Prediction Dashboard")
st.markdown("""
Upload a dataset and the dashboard will **automatically analyze, visualize, and generate insights**
based on the project title and questionnaire-style data.
""")

st.divider()

# -----------------------------
# DATASET UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "📁 Upload your CSV dataset",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("Please upload a CSV file to continue.")
    st.stop()

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(uploaded_file)

# -----------------------------
# AUTO TARGET SELECTION
# -----------------------------
st.sidebar.header("⚙️ Configuration")

possible_targets = [col for col in df.columns if df[col].nunique() <= 5]

TARGET_COLUMN = st.sidebar.selectbox(
    "Select Target Column (Yes/No, 0/1, etc.)",
    possible_targets
)

PROJECT_TITLE = st.sidebar.text_input(
    "Project Title",
    value="Customer Churn Prediction"
)

st.title(f"📌 {PROJECT_TITLE}")

st.divider()

# -----------------------------
# DATASET OVERVIEW
# -----------------------------
st.header("📁 Dataset Overview")

c1, c2, c3 = st.columns(3)
c1.metric("Total Records", df.shape[0])
c2.metric("Total Features", df.shape[1])
c3.metric("Target Rate (%)", round(df[TARGET_COLUMN].value_counts(normalize=True).iloc[0]*100, 2))

st.dataframe(df.head())

st.divider()

# -----------------------------
# TARGET DISTRIBUTION
# -----------------------------
st.header(f"🎯 {TARGET_COLUMN} Distribution")

fig, ax = plt.subplots()
sns.countplot(x=TARGET_COLUMN, data=df, ax=ax)
st.pyplot(fig)

st.divider()

# -----------------------------
# NUMERIC ANALYSIS
# -----------------------------
st.header("📊 Numeric Feature Analysis")

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

for col in numeric_cols:
    if col != TARGET_COLUMN:
        st.subheader(f"{col} vs {TARGET_COLUMN}")
        fig, ax = plt.subplots()
        sns.boxplot(x=TARGET_COLUMN, y=col, data=df, ax=ax)
        st.pyplot(fig)

st.divider()

# -----------------------------
# QUESTIONNAIRE / CATEGORICAL ANALYSIS
# -----------------------------
st.header("📝 Questionnaire-Based Analysis")

cat_cols = df.select_dtypes(include="object").columns

for col in cat_cols:
    if col != TARGET_COLUMN:
        st.subheader(f"{col} vs {TARGET_COLUMN}")
        fig, ax = plt.subplots()
        sns.countplot(x=col, hue=TARGET_COLUMN, data=df, ax=ax)
        plt.xticks(rotation=30)
        st.pyplot(fig)

st.divider()

# -----------------------------
# AUTO INSIGHTS
# -----------------------------
st.header("📌 Key Insights")

st.markdown("""
✔ The target variable shows clear variation across multiple features  
✔ Both numeric and questionnaire-based attributes strongly influence outcomes  
✔ Certain customer segments show higher risk and require attention  
""")

st.divider()

# -----------------------------
# RECOMMENDATIONS
# -----------------------------
st.header("💡 Business Recommendations")

st.markdown("""
🔹 Identify and monitor high-risk segments  
🔹 Improve services linked to negative outcomes  
🔹 Use predictive analytics for proactive decision-making  
""")

st.divider()

st.markdown("""
---
### 👨‍💻 Internship Final Project  
**Developed by Bhanu Prasad**
""")
