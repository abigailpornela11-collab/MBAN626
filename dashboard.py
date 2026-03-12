import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------
# Global Chart Settings
# ---------------------------------
plt.rcParams["figure.figsize"] = (4,2)
plt.rcParams["axes.titlesize"] = 8
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["xtick.labelsize"] = 5
plt.rcParams["ytick.labelsize"] = 5
plt.rcParams["figure.autolayout"] = True

# --------------------------------------------
# Cache Dataset Loading (Improves Performance)
# --------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Predictive_Maintenance_Dataset.csv")

# ------------------------------------------------
# Page Configuration
# ------------------------------------------------
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

# ------------------------------------------------
# Header Banner
# ------------------------------------------------
st.markdown("""
<style>
.header {
background-color:#0e1117;
padding:20px;
border-radius:10px;
color:white;
text-align:center;
}
button[data-baseweb="tab"] {
font-size:18px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
<h1>🔧 Predictive Maintenance Analytics Dashboard</h1>
<p>Machine sensor analysis integrated with environmental context</p>
</div>
""", unsafe_allow_html=True)

st.write("")

st.markdown("""
<style>

.kpi-card {
background-color:#f5f7fb;
padding:20px;
border-radius:12px;
text-align:center;
box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
}

.kpi-title {
font-size:16px;
color:#555;
}

.kpi-value {
font-size:36px;
font-weight:bold;
color:#0e1117;
}

</style>
""", unsafe_allow_html=True)    

# ------------------------------------------------
# Load Dataset
# ------------------------------------------------
df = load_data()

df.columns = df.columns.str.replace(" ", "_")
df = df.drop(columns=["UDI","Product_ID"])

# ------------------------------------------------
# Sidebar Filters
# ------------------------------------------------
st.sidebar.header("Dashboard Filters")

machine_types = ["All"] + sorted(df["Type"].unique().tolist())

machine_type = st.sidebar.selectbox(
    "Select Machine Type",
    machine_types,
    index=0
)

# Apply filter
if machine_type == "All":
    filtered_df = df.copy()
else:
    filtered_df = df[df["Type"] == machine_type]

# ------------------------------------------------
# Dashboard Summary
# ------------------------------------------------

st.markdown("### 📊 Dashboard Summary")

summary_container = st.container()

with summary_container:

    col1, col2 = st.columns(2)

    with col1:

        st.info(
        f"""
        **Operational Overview**

        • The dataset contains **{len(df)} machine observations**.  
        • Overall machine failure rate is approximately **{round(df["Machine_failure"].mean()*100,2)}%**.  
        • Most machines operate normally, indicating failures are relatively rare but critical events.
        """
        )

    with col2:

        st.info(
        """
        **Key Insights**

        • Air temperature strongly correlates with process temperature.  
        • Rotational speed and torque show an inverse mechanical relationship.  
        • Higher tool wear levels are associated with increased failure risk.  
        """
        )

# ------------------------------------------------
# KPI Metrics
# ------------------------------------------------
total_machines = len(filtered_df)
failure_count = filtered_df["Machine_failure"].sum()
failure_rate = (failure_count / total_machines) * 100
avg_tool_wear = filtered_df["Tool_wear_[min]"].mean()

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
    <div class="kpi-card">
    <div class="kpi-title">Total Machines</div>
    <div class="kpi-value">{total_machines}</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card">
    <div class="kpi-title">Failure Count</div>
    <div class="kpi-value">{failure_count}</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card">
    <div class="kpi-title">Failure Rate (%)</div>
    <div class="kpi-value">{round(failure_rate,2)}</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="kpi-card">
    <div class="kpi-title">Avg Tool Wear (min)</div>
    <div class="kpi-value">{round(avg_tool_wear,1)}</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ------------------------------------------------
# Tabs
# ------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Failure Analysis",
    "Sensor Relationships",
    "Machine Behavior",
    "Environmental Context"
])

# =================================================
# FAILURE ANALYSIS
# =================================================
with tab1:

    st.subheader("Failure Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Machine Failure Distribution**")

        fig, ax = plt.subplots(figsize=(4,3))
        sns.countplot(x="Machine_failure", data=filtered_df, ax=ax)

        st.pyplot(fig)

        st.caption(
        "Most machines operate without failure. Failures represent a smaller "
        "portion of observations, highlighting the importance of predictive maintenance."
        )

    with col2:

        st.markdown("**Failure Type Frequency**")

        failure_types = ["TWF","HDF","PWF","OSF","RNF"]

        fig, ax = plt.subplots(figsize=(4,3))
        filtered_df[failure_types].sum().plot(kind="bar", ax=ax)

        st.pyplot(fig)

        st.caption(
        "Heat Dissipation Failure and Overstrain Failure appear frequently, "
        "suggesting thermal and mechanical stress contribute to machine breakdown."
        )

# =================================================
# SENSOR RELATIONSHIPS
# =================================================
with tab2:

    st.subheader("Sensor Relationships")

    st.markdown("**Correlation Heatmap**")

    fig, ax = plt.subplots(figsize=(6,4))  # override global size

    sns.heatmap(
        filtered_df.corr(numeric_only=True),
        cmap="coolwarm",
        annot=False,
        ax=ax
    )

    st.pyplot(fig)

    st.caption(
    "Air temperature and process temperature show strong positive correlation, "
    "indicating that environmental conditions influence machine operating temperature."
    )

    st.markdown("**Pairwise Relationships Between Sensors**")

    pair_vars = [
        "Air_temperature_[K]",
        "Process_temperature_[K]",
        "Rotational_speed_[rpm]",
        "Torque_[Nm]",
        "Tool_wear_[min]",
        "Machine_failure"
    ]

    sample_df = filtered_df[pair_vars].sample(800)

    pairplot = sns.pairplot(
        sample_df,
        hue="Machine_failure",
        diag_kind="hist"
    )

    st.pyplot(pairplot)

    st.caption(
    "The pairwise plot highlights strong relationships between temperature variables "
    "and shows the inverse relationship between rotational speed and torque."
    )

# =================================================
# MACHINE BEHAVIOR
# =================================================
with tab3:

    st.subheader("Machine Behavior Analysis")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("**Rotational Speed vs Torque**")

        fig, ax = plt.subplots(figsize=(4,3))

        sns.scatterplot(
            x="Rotational_speed_[rpm]",
            y="Torque_[Nm]",
            hue="Machine_failure",
            data=filtered_df,
            alpha=0.6,
            ax=ax
        )

        st.pyplot(fig)

        st.caption(
        "A clear inverse relationship exists between rotational speed and torque, "
        "reflecting the mechanical characteristics of rotating systems."
        )

    with col2:

        st.markdown("**Tool Wear vs Machine Failure**")

        fig, ax = plt.subplots(figsize=(4,3))

        sns.boxplot(
            x="Machine_failure",
            y="Tool_wear_[min]",
            data=filtered_df,
            ax=ax
        )

        st.pyplot(fig)

        st.caption(
        "Machines that experience failure generally exhibit higher tool wear, "
        "suggesting accumulated wear contributes to breakdown risk."
        )

# =================================================
# ENVIRONMENTAL CONTEXT
# =================================================
with tab4:

    st.subheader("Environmental Context")

    st.caption(
    "External environmental conditions may influence machine operating temperatures."
    )

    API_KEY = "9c7bf36f5228c50f4d642ffb4f0e5d1c"

    city = "Manila"

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": city,
        "appid": API_KEY
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:

        data = response.json()

        temperature_celsius = data["main"]["temp"] - 273.15
        humidity = data["main"]["humidity"]
        pressure = data["main"]["pressure"]
        wind_speed = data["wind"]["speed"]

    else:

        st.warning("Weather API request failed.")
        temperature_celsius = 0
        humidity = 0
        pressure = 0
        wind_speed = 0

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{'type':'indicator'}, {'type':'indicator'}],
               [{'type':'indicator'}, {'type':'indicator'}]],
        subplot_titles=("Temperature (°C)", "Humidity (%)",
                        "Pressure (hPa)", "Wind Speed (m/s)")
    )

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=temperature_celsius,
        gauge={'axis': {'range':[0,50]}}
    ), row=1, col=1)

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=humidity,
        gauge={'axis': {'range':[0,100]}}
    ), row=1, col=2)

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=pressure,
        gauge={'axis': {'range':[950,1050]}}
    ), row=2, col=1)

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=wind_speed,
        gauge={'axis': {'range':[0,20]}}
    ), row=2, col=2)

    fig.update_layout(height=600)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Environmental Temperature vs Machine Air Temperature**")

    machine_air_temp = filtered_df["Air_temperature_[K]"] - 273.15

    fig, ax = plt.subplots(figsize=(6,3))

    sns.histplot(machine_air_temp, bins=30, kde=True, ax=ax)

    ax.axvline(temperature_celsius, color="red", linestyle="--")

    st.pyplot(fig)

    st.caption(
    "The red dashed line represents the current environmental temperature "
    "retrieved from the weather API and provides context for machine air temperatures."
    )