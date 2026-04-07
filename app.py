import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("StudentsPerformance.csv")
    return df

df = load_data()

st.title("📊 Student Performance EDA Project")

# ---------------- 1. RAW DATA ----------------
st.header("1. Raw Dataset")
st.write(df)

# ---------------- 2. DATA INFO ----------------
st.header("2. Dataset Info")

st.subheader("Shape of Dataset")
st.write(df.shape)

st.subheader("Data Types")
st.write(df.dtypes)

st.subheader("Missing Values")
st.write(df.isnull().sum())

st.subheader("Duplicate Rows")
st.write(df.duplicated().sum())

# ---------------- 3. DATA CLEANING ----------------
st.header("3. Data Cleaning")

clean_df = df.copy()

# Fix age column (convert text to numeric)
clean_df['age'] = pd.to_numeric(clean_df['age'], errors='coerce')

# Fix gender format
clean_df['gender'] = clean_df['gender'].str.capitalize()

# Fill missing numeric values with mean
for col in clean_df.select_dtypes(include=np.number).columns:
    clean_df[col].fillna(clean_df[col].mean(), inplace=True)

# Fill missing categorical values with mode
for col in clean_df.select_dtypes(include='object').columns:
    clean_df[col].fillna(clean_df[col].mode()[0], inplace=True)

# Remove duplicates
clean_df.drop_duplicates(inplace=True)

st.subheader("Cleaned Dataset")
st.write(clean_df)

# ---------------- 4. OUTLIER DETECTION ----------------
st.header("4. Outlier Detection")

numeric_cols = clean_df.select_dtypes(include=np.number).columns

method = st.selectbox("Select Method", ["Z-Score", "IQR"])

outliers = pd.DataFrame()

if method == "Z-Score":
    z = (clean_df[numeric_cols] - clean_df[numeric_cols].mean()) / clean_df[numeric_cols].std()
    outliers = clean_df[(np.abs(z) > 3).any(axis=1)]

elif method == "IQR":
    Q1 = clean_df[numeric_cols].quantile(0.25)
    Q3 = clean_df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1

    outliers = clean_df[((clean_df[numeric_cols] < (Q1 - 1.5 * IQR)) |
                         (clean_df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

st.subheader("Outliers Found")
st.write(outliers)

# ---------------- 5. NORMALIZATION ----------------
st.header("5. Normalization")

norm = st.selectbox("Normalization Type", ["Min-Max", "Z-Score"])

scaled_df = clean_df.copy()

if norm == "Min-Max":
    for col in numeric_cols:
        scaled_df[col] = (clean_df[col] - clean_df[col].min()) / (clean_df[col].max() - clean_df[col].min())
else:
    for col in numeric_cols:
        scaled_df[col] = (clean_df[col] - clean_df[col].mean()) / clean_df[col].std()

st.subheader("Normalized Data")
st.write(scaled_df.head())

# ---------------- 6. GROUPED ANALYSIS ----------------
st.header("6. Grouped Analysis")

group_col = st.selectbox("Group By", clean_df.select_dtypes(include='object').columns)

st.subheader("Mean Values")
st.write(clean_df.groupby(group_col).mean(numeric_only=True))

st.subheader("Detailed Stats")
st.write(clean_df.groupby(group_col).describe())

# ---------------- 7. VISUALIZATION ----------------
st.header("7. Data Visualization")

plot = st.selectbox("Select Plot", ["Histogram", "Boxplot", "Scatter", "Heatmap"])

if plot == "Histogram":
    col = st.selectbox("Select Column", numeric_cols)
    fig, ax = plt.subplots()
    sns.histplot(clean_df[col], kde=True, ax=ax)
    st.pyplot(fig)

elif plot == "Boxplot":
    col = st.selectbox("Select Column", numeric_cols)
    fig, ax = plt.subplots()
    sns.boxplot(x=clean_df[col], ax=ax)
    st.pyplot(fig)

elif plot == "Scatter":
    col1 = st.selectbox("X-axis", numeric_cols)
    col2 = st.selectbox("Y-axis", numeric_cols)
    fig, ax = plt.subplots()
    sns.scatterplot(x=clean_df[col1], y=clean_df[col2], hue=clean_df[group_col], ax=ax)
    st.pyplot(fig)

elif plot == "Heatmap":
    fig, ax = plt.subplots()
    sns.heatmap(clean_df.corr(numeric_only=True), annot=True, ax=ax)
    st.pyplot(fig)

# ---------------- 8. FINAL MESSAGE ----------------
st.success("✅ EDA Completed Successfully!")
