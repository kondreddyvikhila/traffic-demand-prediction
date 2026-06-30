import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ==========================================================
# CONFIG
# ==========================================================

st.set_page_config(
    page_title="Smart Traffic AI",
    page_icon="🚦",
    layout="wide"
)

# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_resource
def load_model():
    return joblib.load("models/traffic_model.pkl")

@st.cache_resource
def load_features():
    return joblib.load("models/features.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/traffic_processed.csv")

@st.cache_resource
def load_metrics():
    return joblib.load("models/model_metrics.pkl")

model = load_model()
features = load_features()
df = load_data()
metrics = load_metrics()
# ==========================================================
# HEADER
# ==========================================================

st.markdown("""
<h1 style='text-align:center; color:#1565C0;'>
🚦 Smart Traffic AI System
</h1>
<p style='text-align:center; color:gray;'>
AI-powered Traffic Demand Prediction Engine
</p>
""", unsafe_allow_html=True)

st.divider()



#
# ==========================================================
# SIDEBAR (UPGRADED TRAFFIC CONTROL PANEL)
# ==========================================================

st.sidebar.markdown(
    """
    <div style='text-align:center;'>
        <h2 style='color:#1565C0;'>🚦 Traffic Control</h2>
        <p style='color:gray; font-size:13px;'>
            Traffic Intelligence Dashboard
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.image(
    "https://img.icons8.com/color/96/traffic-jam.png",
    
    width=80   # change this value to reduce/increase size
)


st.sidebar.markdown("---")

# ==========================================================
# NAVIGATION
# ==========================================================

page = st.sidebar.radio(
    "📌 Navigation",
    ["🏠 Dashboard", "📊 Analytics", "🚀 Predict", "ℹ About"]
)

st.sidebar.markdown("---")

# ==========================================================
# SYSTEM STATUS CARD
# ==========================================================

st.sidebar.markdown(
    """
    <div style='background:#f1f5f9; padding:10px; border-radius:10px; text-align:center;'>
        <p style='margin:0; font-size:13px; color:gray;'>System Status</p>
        <p style='margin:0; font-size:16px; color:green; font-weight:bold;'>
            ● Active
        </p>
    </div>
    """,
    unsafe_allow_html=True
)





st.sidebar.markdown("---")

st.sidebar.markdown("**Version 1.0**")
st.sidebar.text("© 2026 All Rights Reserved")
# ==========================================================
# DASHBOARD
# ==========================================================

if page == "🏠 Dashboard":

    



    # ======================================================
    # KPI CARDS (MORE PROFESSIONAL)
    # ======================================================

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
    "📊 Records Analyzed",
    f"{len(df):,}"
)

    c2.metric(
    "🧩 Features Used",
    len(features)
)

    c3.metric(
    "🤖 Final Model",
    "XGBoost"
)

    c4.metric(
    "🎯 Target",
    "Traffic Volume"
)

    st.divider()

    # ======================================================
    # INSIGHT SECTION (NEW)
    # ======================================================

    st.subheader("✨ Application Capabilities")

    col1, col2 = st.columns(2)

    with col1:
      st.success("""
🚀 **Prediction**

• Predict traffic volume

• Support multiple road types

• Weather-aware forecasting
""")

    with col2:
       st.info("""
📊 **Analytics**

• Interactive visualizations

• Traffic trend analysis

• Weather impact insights
""")

    st.divider()
    # ======================================================
    # MINI VISUAL SNAPSHOT (NEW FEATURE)
    # ======================================================

    st.subheader("📈 Quick Traffic Overview")

    hourly = df.groupby("Hour")["traffic_volume"].mean().reset_index()

    fig = px.line(
        hourly,
        x="Hour",
        y="traffic_volume",
        markers=True,
        title="Traffic Volume Throughout the Day"
    )

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

    

# ==========================================================
# ANALYTICS (FIXED + RELIABLE CHARTS)
# ==========================================================
elif page == "📊 Analytics":

    st.subheader("📈 Traffic Analytics")
    st.caption("Explore traffic patterns and factors influencing traffic volume.")
    analysis = st.selectbox(
    "📊 Select Analysis",
    [
        "Hourly Traffic Pattern",
        "Weather Impact",
        "Traffic by Road Type",
        "Weekend vs Weekday",
        "Traffic Level Distribution",
        "Feature Correlation"
    ]
)
    st.divider()
    # ======================================================
    # HOURLY TREND (SAFE)
    # ======================================================

    if analysis == "Hourly Traffic Pattern":

        data = df.groupby("Hour")["traffic_volume"].mean().reset_index()

        fig = px.line(data, x="Hour", y="traffic_volume", markers=True,
                      title="Hourly Traffic Pattern")

        st.plotly_chart(fig, use_container_width=True)

    # ======================================================
    # WEATHER (FIXED - NO EMPTY GRAPH ISSUE)
    # ======================================================

    elif analysis == "Weather Impact":

       weather_df = (
        df.groupby("Weather_Condition")["traffic_volume"]
        .mean()
        .reset_index()
    )

       fig = px.bar(
        weather_df,
        x="Weather_Condition",
        y="traffic_volume",
        title="Average Traffic Volume by Weather Condition",
        color="traffic_volume"
    )

       fig.update_layout(
        xaxis_title="Weather Condition",
        yaxis_title="Average Traffic Volume",
        template="plotly_white"
    )

       st.plotly_chart(fig, use_container_width=True)

    # ======================================================
    # ROAD TYPE (FIXED)
    # ======================================================

    elif analysis == "Traffic by Road Type":

       road_df = (
        df.groupby("Road_Type")["traffic_volume"]
        .mean()
        .reset_index()
    )

       fig = px.bar(
        road_df,
        x="Road_Type",
        y="traffic_volume",
        title="Average Traffic Volume by Road Type",
        color="traffic_volume"
    )

       fig.update_layout(
        xaxis_title="Road Type",
        yaxis_title="Average Traffic Volume",
        template="plotly_white"
    )

       st.plotly_chart(fig, use_container_width=True)
    # ======================================================
    # WEEKEND PATTERN (SAFE)
    # ======================================================

    elif analysis == "Weekend vs Weekday":

        if "Weekend" in df.columns:

            wk = df.groupby("Weekend")["traffic_volume"].mean().reset_index()
            wk["Weekend"] = wk["Weekend"].map({0: "Weekday", 1: "Weekend"})

            fig = px.bar(wk, x="Weekend", y="traffic_volume",
                         title="Weekend vs Weekday Traffic")

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Weekend column not found in dataset")
 
    # ======================================================
# TRAFFIC LEVEL DISTRIBUTION
# ======================================================

    elif analysis == "Traffic Level Distribution":

      level_df = (
        df["Traffic_Level"]
        .value_counts()
        .reset_index()
    )

      level_df.columns = ["Traffic Level", "Count"]

      fig = px.pie(
        level_df,
        names="Traffic Level",
        values="Count",
        title="Traffic Level Distribution"
    )

      fig.update_layout(
        template="plotly_white",
        height=500
    )

      st.plotly_chart(fig, use_container_width=True)
    # ======================================================
    # CORRELATION (SAFE)
    # ======================================================

    elif analysis == "Feature Correlation":

        numeric_df = df.select_dtypes(include=np.number)

        if numeric_df.shape[1] > 1:

            corr = numeric_df.corr()

            fig = px.imshow(corr, color_continuous_scale="RdBu_r",
                            title="Feature Correlation Heatmap")

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Not enough numeric columns for correlation")

# ==========================================================
# PREDICTION
# ==========================================================

elif page == "🚀 Predict":

    from datetime import date

    st.subheader("🚦 Predict Traffic Volume")


    col1, col2 = st.columns(2)

    with col1:

        hour = st.slider("Hour", 0, 23, 8)

        temp = st.slider(
            "Temperature (°C)",
            -20.0,
            45.0,
            25.0
        )

        humidity = st.slider(
            "Humidity (%)",
            0,
            100,
            60
        )

        rain = st.number_input(
            "Rain (mm)",
            min_value=0.0,
            value=0.0
        )

        snow = st.number_input(
            "Snow (mm)",
            min_value=0.0,
            value=0.0
        )

        clouds = st.slider(
            "Cloud Cover (%)",
            0,
            100,
            40
        )

    with col2:

        road = st.selectbox(
            "Road Type",
            [
                "Highway",
                "Urban",
                "Residential"
            ]
        )

        weather = st.selectbox(
            "Weather",
            [
                "Clouds",
                "Drizzle",
                "Fog",
                "Haze",
                "Mist",
                "Rain",
                "Smoke",
                "Snow",
                "Squall",
                "Thunderstorm"
            ]
        )

        rush = st.selectbox(
            "Rush Hour",
            [
                "No",
                "Yes"
            ]
        )

        lanes = st.selectbox(
            "Number of Lanes",
            [
                2,
                3,
                4,
                5,
                6
            ]
        )

        signal = st.selectbox(
            "Traffic Signal",
            [
                "Red",
                "Yellow"
            ]
        )

        event = st.selectbox(
            "Special Event",
            [
                "No",
                "Yes"
            ]
        )

        large_vehicle = st.slider(
            "Large Vehicle Count",
            0,
            300,
            200
        )

        holiday = st.selectbox(
            "Holiday",
            [
                "None",
                "Columbus Day",
                "Independence Day",
                "Labor Day",
                "Martin Luther King Jr Day",
                "Memorial Day",
                "New Years Day",
                "State Fair",
                "Thanksgiving Day",
                "Veterans Day",
                "Washingtons Birthday"
            ]
        )

    if st.button("🚦 Predict Traffic"):

        # -----------------------------
        # Date Features
        # -----------------------------
    

        from datetime import date

        today = date.today()

        day = today.day
        month = today.month
        weekend = 1 if today.weekday() >= 5 else 0

        days = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

        weekday = days[today.weekday()]

        # -----------------------------
        # Create Feature DataFrame
        # -----------------------------
        input_df = pd.DataFrame(
            np.zeros((1, len(features))),
            columns=features
        )

        # Numerical Features
        input_df.loc[0, "Hour"] = hour
        input_df.loc[0, "Day"] = day
        input_df.loc[0, "Month"] = month
        input_df.loc[0, "Weekend"] = weekend

        input_df.loc[0, "Temperature"] = temp
        input_df.loc[0, "Humidity"] = humidity
        input_df.loc[0, "rain_1h"] = rain
        input_df.loc[0, "snow_1h"] = snow
        input_df.loc[0, "clouds_all"] = clouds

        input_df.loc[0, "Rush_Hour"] = 1 if rush == "Yes" else 0
        input_df.loc[0, "Number_of_Lanes"] = lanes
        input_df.loc[0, "Event_Indicator"] = 1 if event == "Yes" else 0
        input_df.loc[0, "Large_Vehicle_Count"] = large_vehicle

        # Road Type
        road_col = f"Road_Type_{road}"
        if road_col in input_df.columns:
            input_df.loc[0, road_col] = 1

        # Weather
        weather_col = f"Weather_Condition_{weather}"
        if weather_col in input_df.columns:
            input_df.loc[0, weather_col] = 1

        # Holiday
        holiday_col = f"holiday_{holiday}"
        if holiday_col in input_df.columns:
            input_df.loc[0, holiday_col] = 1

        # Day Of Week
        day_col = f"DayOfWeek_{weekday}"
        if day_col in input_df.columns:
            input_df.loc[0, day_col] = 1

        # Traffic Signal
        signal_col = f"Traffic_Signals_{signal}"
        if signal_col in input_df.columns:
            input_df.loc[0, signal_col] = 1

        # -----------------------------
        # Prediction
        # -----------------------------
        prediction = float(model.predict(input_df)[0])

        st.metric(
            "🚗 Predicted Traffic Volume",
            f"{prediction:.0f}",
            "Vehicles / Hour"
        )

        
        # ======================
        # Traffic Status
        # ======================

        if prediction < 1500:
            st.success("🟢 Low Traffic")
            traffic_level = "Low Traffic"

        elif prediction < 3500:
            st.info("🟡 Moderate Traffic")
            traffic_level = "Moderate Traffic"

        else:
            st.error("🔴 High Traffic")
            traffic_level = "High Traffic"
        

        st.divider()

        st.subheader("Prediction Summary")

        st.write(f"""
**Estimated Traffic Volume:** **{prediction:.0f} Vehicles / Hour**


- 🛣️ Road Type: **{road}**
- 🌦️ Weather: **{weather}**
- 🌡️ Temperature: **{temp} °C**
- 💧 Humidity: **{humidity}%**
- ☁️ Cloud Cover: **{clouds}%**
- 🌧️ Rainfall: **{rain} mm**
- ❄️ Snowfall: **{snow} mm**
- 🚛 Large Vehicle Count: **{large_vehicle}**
- 🚥 Traffic Signal: **{signal}**
- 🚨 Rush Hour: **{rush}**
- 🎪 Special Event: **{event}**
- 🎉 Holiday: **{holiday}**
""")
# ==========================================================
# ABOUT (WITH REAL EVALUATION BOX)
# ==========================================================
elif page == "ℹ About":

    st.subheader("ℹ About Smart Traffic AI")

    st.write("""
Smart Traffic AI predicts real-time traffic volume using machine learning
based on weather, road, and temporal features.
    """)

    st.divider()

    # ======================================================
    # REAL EVALUATION BOX (NEW)
    # ======================================================

    st.subheader("Model Evaluation")

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", f"{metrics['MAE']:.2f}")
    col2.metric("RMSE", f"{metrics['RMSE']:.2f}")
    col3.metric("R² Score", f"{metrics['R2 Score']:.4f}")

    st.caption("Evaluation based on test dataset performance")

    st.divider()

    st.write("""
This system is optimized for real-world traffic prediction and supports
decision-making in smart city traffic management systems.
    """)

    st.markdown("---")

st.markdown(
    """
    <div style='text-align:center; color:gray; font-size:14px; padding:10px;'>
        🚦 Smart Traffic AI System

Powered by Streamlit & XGBoost

© 2026
    </div>
    """,
    unsafe_allow_html=True
)
    
        