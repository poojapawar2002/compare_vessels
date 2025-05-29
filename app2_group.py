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
        1004: "CASSIOPEIA*",
        1021: "PYXIS",
        1032: "Cenataurus",
        1016: "CHARA",
        1018: "CARINA*"
        }

        vessel_with_deflectors = [1007, 1018, 1004]
        

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
                # Plot Setup
        fig, ax = plt.subplots(figsize=(10, 6))

        # Bin SpeedOG globally to ensure consistency across vessels
        filtered_df["speed_bin"] = pd.cut(filtered_df["SpeedOG"],
                                          bins=np.arange(filtered_df["SpeedOG"].min() - bin_width,
                                                         filtered_df["SpeedOG"].max() + bin_width,
                                                         bin_width))

        # Prepare containers for both groups
        deflector_group = filtered_df[filtered_df["VesselId"].isin(vessel_with_deflectors)]
        non_deflector_group = filtered_df[~filtered_df["VesselId"].isin(vessel_with_deflectors)]

        def compute_weighted_avg(group, label, color, degree):
            if group.empty:
                return

            vessel_ids = group["VesselId"].unique()
            bin_data = {}

            for vessel in vessel_ids:
                vessel_df = group[group["VesselId"] == vessel]
                vessel_grouped = vessel_df.groupby("speed_bin").agg(
                    mean=("MEFOCDeviation", "mean"),
                    count=("MEFOCDeviation", "count")
                ).dropna()

                for bin_interval, row in vessel_grouped.iterrows():
                    if bin_interval not in bin_data:
                        bin_data[bin_interval] = {"weighted_sum": 0, "total_count": 0}
                    bin_data[bin_interval]["weighted_sum"] += row["mean"] * row["count"]
                    bin_data[bin_interval]["total_count"] += row["count"]

            # Prepare final points
            bin_midpoints = []
            weighted_avgs = []
            for bin_interval, values in bin_data.items():
                if values["total_count"] > 0:
                    midpoint = bin_interval.left + bin_width / 2
                    weighted_avg = values["weighted_sum"] / values["total_count"]
                    bin_midpoints.append(midpoint)
                    weighted_avgs.append(weighted_avg)

            if not bin_midpoints:
                return

            # Scatter plot
            ax.scatter(bin_midpoints, weighted_avgs, label=label, s=60, alpha=0.8, color=color)

            # Polynomial fit
            x = np.array(bin_midpoints).reshape(-1, 1)
            y = np.array(weighted_avgs)

            poly = PolynomialFeatures(degree=degree)
            x_poly = poly.fit_transform(x)
            model = LinearRegression().fit(x_poly, y)

            x_sorted = np.sort(x, axis=0)
            y_pred = model.predict(poly.transform(x_sorted))

            # Plot trend line
            ax.plot(x_sorted, y_pred, color=color, linewidth=2, label=f"{label} (trend)")

        compute_weighted_avg(deflector_group, "With Deflectors", color="green", degree=degree)
        compute_weighted_avg(non_deflector_group, "Without Deflectors", color="blue", degree=degree)


        ax.set_xlabel("SpeedOG (binned)")
        ax.set_ylabel("MEFOCDeviation (weighted avg)")
        ax.set_title("Weighted Avg MEFOCDeviation by SpeedOG Bin with Polynomial Trend")
        ax.legend()
        st.pyplot(fig)

