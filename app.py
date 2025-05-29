import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# --- Load CSV ---
st.title("MEFOCDeviation vs SpeedOG (Binned + Polynomial Fit)")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    required_cols = ["VesselId", "WindSpeedUsed", "MeanDraft", "SpeedOG", "MEFOCDeviation", "IsDeltaFOCMEValid", "IsSpeedDropValid"]
    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV must include: {', '.join(required_cols)}")
    else:
        st.sidebar.header("Filters")

        vessel_names = {
        1023: "MH Perseus",
        1005: "PISCES",
        1007: "CAPELLA*",
        1017: "CETUS",
        1004: "CASSIOPEIA",
        1021: "PYXIS",
        1032: "Cenataurus",
        1016: "CHARA",
        1018: "CARINA*"
        }

        name_to_id = {v: k for k, v in vessel_names.items()}

        # Convert vessel IDs to names for UI
        available_ids = df["VesselId"].unique()
        available_names = [vessel_names.get(vid, f"Unknown ({vid})") for vid in available_ids]
        selected_names = st.sidebar.multiselect("Select Vessels", available_names, default=available_names)

        # Map names back to IDs for filtering
        selected_ids = [name_to_id[name] for name in selected_names]

        st.sidebar.subheader("Wind Speed Range")
        wind_min = st.sidebar.number_input("Min Wind Speed", value=float(df["WindSpeedUsed"].min()))
        wind_max = st.sidebar.number_input("Max Wind Speed", value=float(df["WindSpeedUsed"].max()))

        st.sidebar.subheader("Draft Range")
        draft_min = st.sidebar.number_input("Min Draft", value=float(df["MeanDraft"].min()))
        draft_max = st.sidebar.number_input("Max Draft", value=float(df["MeanDraft"].max()))

        st.sidebar.subheader("Draft Range")
        speedOG_min = st.sidebar.number_input("Min SpeedOG", value=float(df["SpeedOG"].min()))
        speedOG_max = st.sidebar.number_input("Max SpeedOG", value=float(df["SpeedOG"].max()))

        degree = st.sidebar.slider("Polynomial Degree", 1, 5, 2)
        bin_width = 1



        df = df[
    (df["MEFOCDeviation"] >= 0) &
    (df["MEFOCDeviation"] <= 100) &
    (df["IsDeltaFOCMEValid"] == 1) & 
    (df["IsSpeedDropValid"] == 1)
]

        # Apply filter
        filtered_df = df[
            (df["VesselId"].isin(selected_ids)) &
            (df["WindSpeedUsed"] > wind_min) & (df["WindSpeedUsed"] <= wind_max) &
            (df["MeanDraft"] >draft_min) & (df["MeanDraft"] <= draft_max) & 
            (df["SpeedOG"] >speedOG_min) & (df["SpeedOG"] <= speedOG_max)
        ]

        # st.write(f"Filtered data points: {len(filtered_df)}")
        # st.dataframe(filtered_df)

        # Plot Setup
        fig, ax = plt.subplots(figsize=(10, 6))

        for vessel in selected_ids:
            sub_df = filtered_df[filtered_df["VesselId"] == vessel]

            if sub_df.empty:
                continue  # Skip if no data for this vessel after filtering

            # Bin SpeedOG
            min_speed = sub_df["SpeedOG"].min()
            max_speed = sub_df["SpeedOG"].max()
            speed_col = "SpeedOG"
            sub_df["speed_bin"] = pd.cut(df[speed_col],
                         bins=np.arange(df[speed_col].min() - bin_width,
                                        df[speed_col].max() + bin_width,
                                        bin_width))

            # Group by bin and calculate average MEFOCDeviation
            grouped = sub_df.groupby("speed_bin").agg({
                "MEFOCDeviation": "mean"
            }).dropna()

            # Compute midpoints of bins for X-axis
            midpoints = [interval.left + bin_width / 2 for interval in grouped.index]
            x = np.array(midpoints).reshape(-1, 1)
            y = grouped["MEFOCDeviation"].values


            # Fit polynomial
            poly = PolynomialFeatures(degree=degree)
            x_poly = poly.fit_transform(x)
            model = LinearRegression().fit(x_poly, y)
            x_sorted = np.sort(x, axis=0)
            y_pred = model.predict(poly.transform(x_sorted))

            # Plot
            ax.scatter(x, y, alpha=0.7) #, label=f"{vessel} avg points"
            ax.plot(x_sorted, y_pred, label=f"{vessel_names[vessel]}", linewidth=2) #fit (deg {degree})

        ax.set_xlabel("SpeedOG (binned avg)")
        ax.set_ylabel("MEFOCDeviation (avg)")
        ax.set_title("Binned SpeedOG vs MEFOCDeviation with Vessel-wise Polynomial Fit")
        ax.legend()
        st.pyplot(fig)
