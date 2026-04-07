import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Student Performance EDA", layout="wide", page_icon="📊")

# ─────────────────────────────────────────────
# LOAD & CACHE RAW DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("StudentsPerformance_large.csv")
    return df


# ─────────────────────────────────────────────
# CLEAN & CACHE CLEANED DATA  (performance fix)
# Running this every widget interaction was slow
# ─────────────────────────────────────────────
@st.cache_data
def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    clean = df.copy()

    # 🔹 Convert columns to best possible types (important for pandas 2.x)
    clean = clean.infer_objects(copy=False)

    # 🔹 Convert age to numeric (handle "twenty", "", etc.)
    if "age" in clean.columns:
        clean["age"] = pd.to_numeric(clean["age"], errors="coerce")

    # 🔹 Standardize gender values
    if "gender" in clean.columns:
        clean["gender"] = clean["gender"].astype(str).str.strip().str.capitalize()

    # 🔹 Remove duplicates
    before = len(clean)

    # If roll_no exists → best way
    if "roll_no" in clean.columns:
        clean = clean.drop_duplicates(subset=["roll_no"], keep="last")
    else:
        clean = clean.drop_duplicates()

    removed_dups = before - len(clean)

    # 🔹 Handle missing values safely
    for col in clean.columns:
        try:
            if pd.api.types.is_numeric_dtype(clean[col]):  # ✅ safe numeric check
                skew_val = clean[col].skew()

                if pd.notna(skew_val) and skew_val > 1:
                    clean[col] = clean[col].fillna(clean[col].median())
                else:
                    clean[col] = clean[col].fillna(clean[col].mean())
            else:
                mode = clean[col].mode()
                clean[col] = clean[col].fillna(mode[0] if not mode.empty else "")
        except Exception:
            # fallback safety
            clean[col] = clean[col].fillna("")

    # 🔹 Create derived columns (scores)
    score_cols = [c for c in clean.columns if c.lower() in ["math", "physics", "chemistry", "english"]]

    if score_cols:
        clean["total_score"] = clean[score_cols].sum(axis=1)
        clean["avg_score"] = clean[score_cols].mean(axis=1).round(2)

    return clean, removed_dups


# ─────────────────────────────────────────────
# OUTLIER DETECTION (bug fix: per-column MAD)
# ─────────────────────────────────────────────
def detect_outliers(df: pd.DataFrame, numeric_cols, method: str) -> pd.DataFrame:
    if not numeric_cols or df.empty:
        return pd.DataFrame()
    
    # We only want to check columns that actually have variance
    subset = df[numeric_cols].dropna()
    
    if method == "Z-Score":
        # Standard: Mean +/- 3 Std Dev
        z = (subset - subset.mean()) / subset.std()
        mask = (np.abs(z) > 3).any(axis=1)

    elif method == "Modified Z-Score":
        # Robust: Median +/- 3.5 MAD
        median = subset.median()
        mad = (subset - median).abs().median()
        # Avoid division by zero if MAD is 0
        mad = mad.replace(0, np.nan)
        mod_z = 0.6745 * (subset - median) / mad
        mask = (np.abs(mod_z) > 3.5).any(axis=1)

    else:  # IQR Method
        Q1 = subset.quantile(0.25)
        Q3 = subset.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        mask = ((subset < lower_bound) | (subset > upper_bound)).any(axis=1)

    # Return the rows from the original dataframe
    return df.loc[subset.index[mask]]


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ `StudentsPerformance.csv` not found. Place it in the same directory as this script.")
    st.stop()

clean_df, removed_dups = clean_data(df)
numeric_cols = clean_df.select_dtypes(include=np.number).columns.tolist()
cat_cols = clean_df.select_dtypes(include="object").columns.tolist()

with st.sidebar:
    st.title("⚙️ Controls")

    st.markdown("### Outlier Detection")
    outlier_method = st.selectbox(
        "Method", ["IQR", "Z-Score", "Modified Z-Score"], key="outlier_method"
    )
   

    # Chart palette removed as requested - Defaulting to 'Set2' in code
    chart_palette = "Set2" 

  

    st.markdown("---")
    csv_bytes = clean_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download cleaned CSV",
        data=csv_bytes,
        file_name="cleaned_students.csv",
        mime="text/csv",
    )
# ── Header ────────────────────────────────────
st.title("📊 Student Performance — Advanced EDA")
st.caption("Cleaned • Interactive • Reproducible")

# Quick metrics
m_cols = st.columns(5)
m_cols[0].metric("Total rows", f"{len(df):,}")
m_cols[1].metric("After cleaning", f"{len(clean_df):,}")
m_cols[2].metric("Duplicates removed", removed_dups)
m_cols[3].metric("Numeric features", len(numeric_cols))
m_cols[4].metric("Categorical features", len(cat_cols))

st.divider()

# ── Tabs ──────────────────────────────────────
tabs = st.tabs([
    "1 · Preview",
    "2 · Raw Info",
    "3 · Cleaning",
    "4 · Stats",
    "5 · Outliers",
    "6 · Categorical",
    "7 · Distributions",
    "8 · Normalisation",
    "9 · Advanced Viz",
])

# ── Tab 1: Preview ────────────────────────────
with tabs[0]:
    st.header("Dataset Preview")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Head (5 rows)")
        st.dataframe(df.head(), use_container_width=True)
    with c2:
        st.subheader("Tail (5 rows)")
        st.dataframe(df.tail(), use_container_width=True)

# ── Tab 2: Raw Info ───────────────────────────
with tabs[1]:
    st.header("Dataset Info (Before Cleaning)")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Shape:**", df.shape)
        st.write("**Columns:**", df.columns.tolist())
        st.subheader("Data types")
        st.dataframe(df.dtypes.rename("dtype").reset_index(), use_container_width=True)
    with c2:
        st.subheader("Null value counts")
        st.dataframe(df.isnull().sum().rename("nulls").reset_index(), use_container_width=True)
        st.metric("Duplicate rows", df.duplicated().sum())

# ── Tab 3: Cleaning ───────────────────────────
with tabs[2]:
    st.header("Data Cleaning")
    st.success(f"✅ Duplicates removed: **{removed_dups}**")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Null values after cleaning")
        null_after = clean_df.isnull().sum()
        st.dataframe(null_after.rename("nulls").reset_index(), use_container_width=True)
    with c2:
        if "total_score" in clean_df.columns:
            st.subheader("Derived columns added")
            st.info("🆕 `total_score` and `avg_score` computed from score columns")

    st.subheader("Cleaned dataset")
    st.dataframe(clean_df, use_container_width=True)

# ── Tab 4: Descriptive Stats ──────────────────
with tabs[3]:
    st.header("Descriptive Statistics")
    st.dataframe(clean_df.describe().T.style.format("{:.2f}"), use_container_width=True)

    st.subheader("Skewness & Kurtosis")
    sk_df = pd.DataFrame({
        "Skewness": clean_df[numeric_cols].skew(),
        "Kurtosis": clean_df[numeric_cols].kurt(),
    })
    st.dataframe(sk_df.style.background_gradient(cmap="RdYlGn_r"), use_container_width=True)

# ── Tab 5: Outliers ───────────────────────────
with tabs[4]:
    st.header(f"Outlier Detection — {outlier_method}")
    outliers = detect_outliers(clean_df, numeric_cols, outlier_method)
    st.metric("Outlier rows detected", len(outliers))
    st.dataframe(outliers, use_container_width=True)

    st.subheader("Boxplots")
    cols_per_row = 3
    col_chunks = [numeric_cols[i:i+cols_per_row] for i in range(0, len(numeric_cols), cols_per_row)]
    for chunk in col_chunks:
        row_cols = st.columns(len(chunk))
        for ax_col, col in zip(row_cols, chunk):
            with ax_col:
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.boxplot(x=clean_df[col], ax=ax, palette=chart_palette)
                ax.set_title(col)
                st.pyplot(fig)   # Bug fix: always pass fig
                plt.close(fig)

# ── Tab 6: Categorical Analysis (new section) ─
with tabs[5]:
    st.header("Categorical Feature Analysis")

    if not cat_cols:
        st.info("No categorical columns found in the dataset.")
    else:
        for cat in cat_cols:
            st.subheader(f"Distribution: `{cat}`")
            vc = clean_df[cat].value_counts().reset_index()
            vc.columns = [cat, "count"]

            c1, c2 = st.columns([1, 2])
            with c1:
                st.dataframe(vc, use_container_width=True)
            with c2:
                fig, ax = plt.subplots(figsize=(5, 3))
                sns.countplot(data=clean_df, x=cat, palette=chart_palette, ax=ax,
                              order=vc[cat])
                ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
                st.pyplot(fig)    # Bug fix: always pass fig
                plt.close(fig)

        if numeric_cols:
            st.subheader("Grouped Means by Category")
            group_cat = st.selectbox("Group by", cat_cols, key="group_cat")
            group_num = st.selectbox("Numeric column", numeric_cols, key="group_num")
            grouped = clean_df.groupby(group_cat)[group_num].mean().reset_index().sort_values(group_num, ascending=False)
            st.dataframe(grouped.style.format({group_num: "{:.2f}"}), use_container_width=True)

            fig, ax = plt.subplots(figsize=(6, 3))
            sns.barplot(data=grouped, x=group_cat, y=group_num, palette=chart_palette, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
            st.pyplot(fig)
            plt.close(fig)

# ── Tab 7: Distributions ─────────────────────
with tabs[6]:
    st.header("Histograms & KDE")
    # Bug fix: sections 5 and 7 in original both did histplots — merged here
    col_chunks = [numeric_cols[i:i+2] for i in range(0, len(numeric_cols), 2)]
    for chunk in col_chunks:
        row_cols = st.columns(len(chunk))
        for ax_col, col in zip(row_cols, chunk):
            with ax_col:
                fig, ax = plt.subplots(figsize=(5, 3))
                sns.histplot(clean_df[col], kde=True, ax=ax, color=sns.color_palette(chart_palette)[0])
                ax.set_title(f"{col}  (skew: {clean_df[col].skew():.2f})")
                st.pyplot(fig)   # Bug fix: always pass fig
                plt.close(fig)

# ── Tab 8: Normalisation ──────────────────────
with tabs[7]:
    st.header("Min-Max Normalisation")
    scaled_df = clean_df.copy()
    for col in numeric_cols:
        mn, mx = clean_df[col].min(), clean_df[col].max()
        if mx > mn:
            scaled_df[col] = (clean_df[col] - mn) / (mx - mn)
        # If range is 0, leave column as-is rather than producing NaN

    st.dataframe(scaled_df[numeric_cols].head(10).style.format("{:.4f}"), use_container_width=True)

# ── Tab 9: Advanced Viz ───────────────────────
with tabs[8]:
    st.header("Advanced Visualisation")

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(clean_df.corr(numeric_only=True), annot=True, fmt=".2f",
                cmap="coolwarm", ax=ax)
    st.pyplot(fig)   # Bug fix: always pass fig
    plt.close(fig)

    # Pairplot (bug fix: capture returned figure from pairplot)
    st.subheader("Pairplot")
    with st.spinner("Rendering pairplot…"):
        pp_hue = hue if (hue and hue in clean_df.columns) else None
        pair_fig = sns.pairplot(clean_df[numeric_cols[:4] + ([pp_hue] if pp_hue else [])],
                                hue=pp_hue, palette=chart_palette)
        st.pyplot(pair_fig.figure)   # Bug fix: pass pair_fig.figure, not bare st.pyplot()
        plt.close("all")

    st.divider()

    # KDE plot (bug fix: unique key)
    st.subheader("KDE Plot")
    kde_col = st.selectbox("Column", numeric_cols, key="kde_col_select")
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.kdeplot(clean_df[kde_col], fill=True, ax=ax,
                color=sns.color_palette(chart_palette)[2])
    st.pyplot(fig)
    plt.close(fig)

    # Violin plot (bug fix: unique key)
    st.subheader("Violin Plot")
    vio_col = st.selectbox("Column", numeric_cols, key="violin_col_select")
    fig, ax = plt.subplots(figsize=(6, 3))
    vio_hue = hue if (hue and clean_df[hue].nunique() <= 6) else None
    sns.violinplot(data=clean_df, x=vio_col, hue=vio_hue,
                   palette=chart_palette, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    # Scatter plot (bug fix: unique keys)
    st.subheader("Scatter Plot")
    c1, c2 = st.columns(2)
    with c1:
        scatter_x = st.selectbox("X axis", numeric_cols, key="scatter_x")
    with c2:
        scatter_y = st.selectbox("Y axis", numeric_cols,
                                 index=min(1, len(numeric_cols) - 1), key="scatter_y")
    fig, ax = plt.subplots(figsize=(6, 4))
    sc_hue = hue if (hue and hue in clean_df.columns) else None
    sns.scatterplot(data=clean_df, x=scatter_x, y=scatter_y,
                    hue=sc_hue, palette=chart_palette, alpha=0.7, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

st.divider()
st.success("✅ Full Advanced EDA Completed!")