"""Streamlit dashboard for the Medical Insurance dataset.

Run with:
    streamlit run streamlit_app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Medical Insurance Dashboard", layout="wide")

REGION_COORDS = {
    "northeast": (54.9783, -1.6178),   # Newcastle
    "northwest": (53.4808, -2.2426),   # Manchester
    "southeast": (51.2362, -0.5704),   # Guildford
    "southwest": (50.8225, -3.5339),   # Exeter
}


@st.cache_data
def load_and_clean_data(path: str = "insurance_data.xlsx") -> pd.DataFrame:
    df = pd.read_excel(path)
    df_clean = df.copy()

    for col in ["gender", "smoker", "region"]:
        df_clean[col] = df_clean[col].astype("string").str.strip().str.lower()

    df_clean["region"] = df_clean["region"].replace({"northteast": "northeast"})
    df_clean.loc[df_clean["children"] > 5, "children"] = 5

    for col in ["age", "bmi", "children", "charges"]:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    for col in ["gender", "smoker", "region"]:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

    df_clean["smoker_binary"] = df_clean["smoker"].map({"yes": 1, "no": 0})
    df_clean["bmi_category"] = pd.cut(
        df_clean["bmi"],
        bins=[0, 18.5, 25, 30, np.inf],
        labels=["underweight", "healthy", "overweight", "obese"],
        right=False,
    )
    return df_clean


df_clean = load_and_clean_data()

st.title("Medical Insurance Charges Dashboard")

# --- Sidebar filters ---
st.sidebar.header("Filters")
regions = st.sidebar.multiselect(
    "Region", sorted(df_clean["region"].unique()), default=sorted(df_clean["region"].unique())
)
smoker_options = st.sidebar.multiselect(
    "Smoker status", sorted(df_clean["smoker"].unique()), default=sorted(df_clean["smoker"].unique())
)
age_range = st.sidebar.slider(
    "Age range", int(df_clean["age"].min()), int(df_clean["age"].max()),
    (int(df_clean["age"].min()), int(df_clean["age"].max()))
)
bmi_range = st.sidebar.slider(
    "BMI range", float(df_clean["bmi"].min()), float(df_clean["bmi"].max()),
    (float(df_clean["bmi"].min()), float(df_clean["bmi"].max()))
)

filtered = df_clean[
    df_clean["region"].isin(regions)
    & df_clean["smoker"].isin(smoker_options)
    & df_clean["age"].between(*age_range)
    & df_clean["bmi"].between(*bmi_range)
]

if filtered.empty:
    st.warning("No records match the current filters.")
    st.stop()

# --- KPIs ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Policyholders", f"{len(filtered):,}")
col2.metric("Average Charges", f"£{filtered['charges'].mean():,.0f}")
col3.metric("Average BMI", f"{filtered['bmi'].mean():.1f}")
col4.metric("Smoker %", f"{(filtered['smoker'].eq('yes').mean() * 100):.1f}%")

st.divider()

# --- Demographic bubble chart ---
st.subheader("Demographic Patterns: Age vs Charges (Size = BMI, Colour = Smoker)")
fig_demo = px.scatter(
    filtered, x="age", y="charges", size="bmi", color="smoker",
    color_discrete_map={"yes": "crimson", "no": "steelblue"},
    hover_data=["gender", "children", "region", "bmi_category"],
    size_max=40, opacity=0.7,
)
st.plotly_chart(fig_demo, use_container_width=True)

# --- Regional disparities ---
left, right = st.columns(2)
with left:
    st.subheader("Regional Disparities: Charges by Region")
    fig_region_box = px.box(filtered, x="region", y="charges", color="region", points="outliers")
    st.plotly_chart(fig_region_box, use_container_width=True)

with right:
    st.subheader("No-Claims Bonus Impact")
    fig_ncb = px.scatter(
        filtered, x="NoClaimsBonus", y="charges", color="smoker",
        color_discrete_map={"yes": "crimson", "no": "steelblue"},
        opacity=0.5, trendline="ols",
    )
    st.plotly_chart(fig_ncb, use_container_width=True)

# --- Children multiplier ---
st.subheader("The Dependents Multiplier: Children vs Charges")
children_summary = filtered.groupby("children")["charges"].mean()
fig_children = px.box(filtered, x="children", y="charges", points=False)
fig_children.add_scatter(
    x=children_summary.index, y=children_summary.values,
    mode="lines+markers", name="Mean Charges", line=dict(color="crimson", width=3),
)
st.plotly_chart(fig_children, use_container_width=True)

# --- Regional map ---
st.subheader("Regional Disparities on a Map (Approximate Locations)")
map_df = filtered.copy()
rng = np.random.default_rng(42)
map_df["lat"] = map_df["region"].map(lambda r: REGION_COORDS[r][0]) + rng.uniform(-0.4, 0.4, len(map_df))
map_df["lon"] = map_df["region"].map(lambda r: REGION_COORDS[r][1]) + rng.uniform(-0.4, 0.4, len(map_df))
fig_map = px.scatter_map(
    map_df, lat="lat", lon="lon", color="charges", size="charges",
    color_continuous_scale="Reds", hover_data=["region", "smoker", "age", "bmi", "NoClaimsBonus"],
    zoom=4.5,
)
fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0))
st.plotly_chart(fig_map, use_container_width=True)

st.divider()
st.header("Interaction, Regional and Dimensionality-Reduction Insights")

# --- Multivariate profile (parallel coordinates) ---
st.subheader("Multivariate Medical Insurance Profiles (Parallel Coordinates)")
df_para = filtered.copy()
df_para["smoker_code"] = df_para["smoker"].map({"yes": 1, "no": 0})
df_para["gender_code"] = df_para["gender"].map({"male": 1, "female": 0})
region_cats = df_para["region"].astype("category").cat.categories
df_para["region_code"] = df_para["region"].astype("category").cat.codes
bmi_cats = df_para["bmi_category"].cat.categories
df_para["bmi_category_code"] = df_para["bmi_category"].cat.codes

fig_parallel = px.parallel_coordinates(
    df_para,
    dimensions=[
        "age", "gender_code", "bmi", "bmi_category_code", "children",
        "smoker_code", "region_code", "NoClaimsBonus", "charges",
    ],
    color="charges",
    color_continuous_scale=px.colors.sequential.Inferno,
    labels={
        "age": "Age", "gender_code": "Gender", "bmi": "BMI Index",
        "bmi_category_code": "BMI Category", "children": "Children",
        "smoker_code": "Smoking Status", "region_code": "UK Region",
        "NoClaimsBonus": "No Claims Bonus", "charges": "Total Charges",
    },
)
axis_labels_map = {
    "gender_code": {"tickvals": [0, 1], "ticktext": ["Female", "Male"]},
    "smoker_code": {"tickvals": [0, 1], "ticktext": ["Non-Smoker", "Smoker"]},
    "region_code": {"tickvals": list(range(len(region_cats))), "ticktext": list(region_cats)},
    "bmi_category_code": {"tickvals": list(range(len(bmi_cats))), "ticktext": list(bmi_cats)},
}
for dim in fig_parallel.data[0].dimensions:
    if dim.name in axis_labels_map:
        dim.update(axis_labels_map[dim.name])
fig_parallel.update_traces(unselected=dict(line=dict(opacity=0.05)))
fig_parallel.update_layout(
    coloraxis_colorbar=dict(title="Charges (£)"),
    margin=dict(l=80, r=80, t=40, b=20),
)
st.plotly_chart(fig_parallel, use_container_width=True)
st.caption("Drag along any axis to brush/highlight matching profiles across all dimensions. Chart updates with the sidebar filters.")

# --- Obese-smoker financial cliff ---
st.subheader("The Obese-Smoker Interaction: A Hidden Financial Cliff")
fig_cliff = px.scatter(
    filtered, x="bmi", y="charges", color="smoker", size="age",
    color_discrete_map={"yes": "crimson", "no": "steelblue"},
    opacity=0.7, size_max=20,
)
fig_cliff.add_vline(x=30, line_dash="dash", line_color="black", annotation_text="Obesity Threshold (BMI 30)")
st.plotly_chart(fig_cliff, use_container_width=True)

# --- Regional risk index ---
st.subheader("Regional Risk Index (% Smokers × Avg BMI / Avg NoClaimsBonus)")
regional_summary = filtered.groupby("region").agg(
    avg_charges=("charges", "mean"),
    smoker_rate=("smoker_binary", "mean"),
    avg_bmi=("bmi", "mean"),
    avg_ncb=("NoClaimsBonus", "mean"),
)
regional_summary["Risk_Index"] = (
    regional_summary["smoker_rate"] * regional_summary["avg_bmi"]
) / regional_summary["avg_ncb"]
regional_plot = regional_summary.sort_values("avg_charges", ascending=False).reset_index()
fig_risk_index = px.bar(
    regional_plot, x="region", y="Risk_Index", color="avg_charges",
    color_continuous_scale="Reds", text_auto=".2f",
    hover_data=["avg_charges", "smoker_rate", "avg_bmi", "avg_ncb"],
)
st.plotly_chart(fig_risk_index, use_container_width=True)

# --- NCB degraded cushion by age group ---
st.subheader("The Degraded Cushion of the No-Claims Bonus, by Age Group")
age_bins = [17, 30, 45, 60, filtered["age"].max() + 1]
age_labels = ["18-30", "31-45", "46-60", "60+"]
age_grouped = filtered.copy()
age_grouped["age_group"] = pd.cut(age_grouped["age"], bins=age_bins, labels=age_labels, right=True)
fig_ncb_facet = px.scatter(
    age_grouped, x="NoClaimsBonus", y="charges", color="smoker",
    color_discrete_map={"yes": "crimson", "no": "steelblue"},
    facet_col="age_group", facet_col_wrap=2, opacity=0.5, trendline="ols",
)
st.plotly_chart(fig_ncb_facet, use_container_width=True)

# --- PCA vs LDA silhouette comparison ---
st.subheader("Class Separation: PCA (Unsupervised) vs LDA (Supervised)")
pca_lda_cols = ["age", "gender", "bmi", "bmi_category", "children", "region", "NoClaimsBonus"]
pca_lda_categorical = ["gender", "bmi_category", "region"]
X_pca_lda = pd.get_dummies(filtered[pca_lda_cols], columns=pca_lda_categorical, drop_first=True, dtype=int)
y_pca_lda = filtered["smoker_binary"]

if y_pca_lda.nunique() < 2:
    st.info("Both smoker classes must be present in the current filter to compute PCA/LDA silhouette scores.")
else:
    X_pca_lda_scaled = StandardScaler().fit_transform(X_pca_lda)
    X_pca_proj = PCA(n_components=2).fit_transform(X_pca_lda_scaled)
    X_lda_proj = LDA(n_components=1).fit_transform(X_pca_lda_scaled, y_pca_lda)

    silhouette_comparison = pd.DataFrame({
        "Method": ["PCA", "LDA"],
        "Silhouette_Score": [
            silhouette_score(X_pca_proj, y_pca_lda),
            silhouette_score(X_lda_proj, y_pca_lda),
        ],
    })
    fig_silhouette = px.bar(
        silhouette_comparison, x="Method", y="Silhouette_Score", color="Method",
        color_discrete_map={"PCA": "steelblue", "LDA": "crimson"}, text_auto=".3f",
    )
    st.plotly_chart(fig_silhouette, use_container_width=True)

# --- Raw data ---
with st.expander("View filtered data"):
    st.dataframe(filtered)
