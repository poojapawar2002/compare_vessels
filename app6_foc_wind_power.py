import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import math

# --- Load CSV ---
st.title("FOCWindPower vs SpeedOG")

df = pd.read_csv("final_combined_output.csv")

required_cols = ["VesselId", "WindSpeedUsed", "MeanDraft", "SpeedOG", "MEFOCDeviation", "IsDeltaFOCMEValid", "IsSpeedDropValid","FOCWindPower"]
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

    st.sidebar.subheader("Wind Speed Range (m/s)")
    wind_min = st.sidebar.number_input("Min Wind Speed", value=float(df["WindSpeedUsed"].min()), step=0.5)
    wind_max = st.sidebar.number_input("Max Wind Speed", value=float(df["WindSpeedUsed"].max()), step=0.5)

    st.sidebar.subheader("Draft Range (m)")
    draft_min = st.sidebar.number_input("Min Draft", value=float(df["MeanDraft"].min()), step=0.5)
    draft_max = st.sidebar.number_input("Max Draft", value=float(df["MeanDraft"].max()), step=0.5)

    st.sidebar.subheader("SpeedOG Range (knots)")
    speedOG_min = st.sidebar.number_input("Min SpeedOG", value=float(df["SpeedOG"].min()), step=0.5)
    speedOG_max = st.sidebar.number_input("Max SpeedOG", value=float(df["SpeedOG"].max()), step=0.5)

    st.sidebar.subheader("Relative Wind Direction Range (degrees)")
    rel_wind_dir_min = st.sidebar.number_input("Min Relative Wind Direction", value=float(df["RelativeWindDirection"].min()), step=0.5)
    rel_wind_dir_max = st.sidebar.number_input("Max Relative Wind Direction", value=float(df["RelativeWindDirection"].max()), step=0.5)



    degree = 2 #st.sidebar.slider("Polynomial Degree", 1, 5, 2)
    
    # NEW: Add option for manual speed ranges
    # st.sidebar.subheader("FOCWindPower over SpeedOG Ranges")
    range_width = 1 #st.sidebar.number_input("Speed Range Width", min_value=0.5, max_value=5.0, value=1.0, step=0.5)

    # Apply initial data quality filters
    df = df[
        (df["MEFOCDeviation"] >= 0) &
        (df["MEFOCDeviation"] <= 100) &
        (df["IsDeltaFOCMEValid"] == 1) & 
        (df["IsSpeedDropValid"] == 1)
    ]

    df["FOCWindPower"] = (df["FOCWindPower"])*(1440/df["ME1RunningHoursMinute"])

    

    # Apply user filters
    filtered_df = df[
        (df["VesselId"].isin(selected_ids)) &
        (df["WindSpeedUsed"] > wind_min) & (df["WindSpeedUsed"] <= wind_max) &
        (df["MeanDraft"] > draft_min) & (df["MeanDraft"] <= draft_max) & 
        (df["SpeedOG"] >= speedOG_min) & (df["SpeedOG"] <= speedOG_max) &
        (df["RelativeWindDirection"] >= rel_wind_dir_min) & (df["RelativeWindDirection"] <= rel_wind_dir_max)
    ]

    # NEW APPROACH: Create manual speed ranges instead of using bins
    def create_speed_ranges(min_speed, max_speed, width):
        ranges = []
        current = min_speed
        while current < max_speed:
            range_end = min(current + width, max_speed)
            ranges.append((current, range_end))
            current += width
        return ranges

    # Create speed ranges based on user input
    speed_ranges = create_speed_ranges(speedOG_min, speedOG_max, range_width)
    
    # Function to calculate statistics for each speed range
    def calculate_range_stats(df, speed_ranges, deflector_vessel_ids):
        results = []
        
        for range_start, range_end in speed_ranges:
            # Filter data for this speed range
            range_data = df[
                (df["SpeedOG"] > range_start) & 
                (df["SpeedOG"] <= range_end)
            ]
            
            if range_data.empty:
                results.append({
                    'SpeedRange': f"({range_start:.1f}, {range_end:.1f}]",
                    'WithDeflectors_Mean': None,
                    'WithDeflectors_Count': 0,
                    'WithoutDeflectors_Mean': None,
                    'WithoutDeflectors_Count': 0
                })
                continue
            
            # Split by deflector status
            with_deflectors = range_data[range_data["VesselId"].isin(deflector_vessel_ids)]
            without_deflectors = range_data[~range_data["VesselId"].isin(deflector_vessel_ids)]
            
            # Calculate means
            with_mean = with_deflectors["FOCWindPower"].mean() if not with_deflectors.empty else None
            without_mean = without_deflectors["FOCWindPower"].mean() if not without_deflectors.empty else None
            
            results.append({
                'SpeedRange': f"({range_start:.1f}, {range_end:.1f}]",
                'WithDeflectors_Mean': with_mean,
                'WithDeflectors_Count': len(with_deflectors),
                'WithoutDeflectors_Mean': without_mean,
                'WithoutDeflectors_Count': len(without_deflectors)
            })
        
        return pd.DataFrame(results)

    # Calculate statistics
    stats_df = calculate_range_stats(filtered_df, speed_ranges, vessel_with_deflectors)
    
    # Format for display
    display_df = stats_df.copy()
    display_df['WithDeflectors_Mean'] = display_df['WithDeflectors_Mean'].apply(
        lambda x: f"{x:.4f}" if pd.notna(x) else "None"
    )
    display_df['WithoutDeflectors_Mean'] = display_df['WithoutDeflectors_Mean'].apply(
        lambda x: f"{x:.4f}" if pd.notna(x) else "None"
    )
    
    # # Calculate percentage difference
    # def calculate_percentage_diff(with_val, without_val):
    #     if pd.isna(with_val) or pd.isna(without_val) or without_val == 0:
    #         return None
    #     return ((with_val - without_val) / without_val) * 100
    
    # display_df['Percentage_Difference'] = [
    #     calculate_percentage_diff(row['WithDeflectors_Mean'], row['WithoutDeflectors_Mean']) 
    #     for _, row in stats_df.iterrows()
    # ]
    
    # # Format percentage difference
    # display_df['Percentage_Difference'] = display_df['Percentage_Difference'].apply(
    #     lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
    # )
    
    # Rename columns for better display
    display_df = display_df.rename(columns={
        'SpeedRange': 'Speed Range',
        'WithDeflectors_Mean': 'Weighted Avg FOCWindPower With Deflector',
        # 'WithDeflectors_Count': 'Count (With Deflectors)',
        'WithoutDeflectors_Mean': 'Weighted Avg FOCWindPower Without Deflector',
        # 'WithoutDeflectors_Count': 'Count (Without Deflectors)',
        # 'Percentage_Difference': 'Percentage Difference'
    })
    
    # Select columns to display (hiding count columns)
    display_columns = ['Speed Range', 'Weighted Avg FOCWindPower With Deflector', 'Weighted Avg FOCWindPower Without Deflector']
    
    # Show the table
    st.subheader("FOCWindPower Over SpeedOG Ranges")
    st.dataframe(display_df[display_columns])
    
    # # Show summary statistics
    # st.subheader("Summary Statistics")
    # col1, col2 = st.columns(2)
    
    # with col1:
    #     st.write("**With Deflectors:**")
    #     deflector_data = filtered_df[filtered_df["VesselId"].isin(vessel_with_deflectors)]
    #     if not deflector_data.empty:
    #         st.write(f"Total Records: {len(deflector_data)}")
    #         st.write(f"Mean FOCWindPower: {deflector_data['FOCWindPower'].mean():.4f}")
    #         st.write(f"Std FOCWindPower: {deflector_data['FOCWindPower'].std():.4f}")
    #     else:
    #         st.write("No data available")
    
    # with col2:
    #     st.write("**Without Deflectors:**")
    #     non_deflector_data = filtered_df[~filtered_df["VesselId"].isin(vessel_with_deflectors)]
    #     if not non_deflector_data.empty:
    #         st.write(f"Total Records: {len(non_deflector_data)}")
    #         st.write(f"Mean FOCWindPower: {non_deflector_data['FOCWindPower'].mean():.4f}")
    #         st.write(f"Std FOCWindPower: {non_deflector_data['FOCWindPower'].std():.4f}")
    #     else:
    #         st.write("No data available")

    # Plot Setup
    fig, ax = plt.subplots(figsize=(12, 8))

    def plot_scatter_and_fit(group, label, color1, color2,degree):
        if group.empty:
            return

        # Scatter plot of raw data
        ax.scatter(group["SpeedOG"], group["FOCWindPower"], label=label, color=color1, alpha=0.6, s=30)

        # Polynomial Fit
        x = group["SpeedOG"].values.reshape(-1, 1)
        y = group["FOCWindPower"].values

        poly = PolynomialFeatures(degree=degree)
        x_poly = poly.fit_transform(x)
        model = LinearRegression().fit(x_poly, y)

        # Predict over sorted x for smooth line
        x_range = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)
        y_pred = model.predict(poly.transform(x_range))

        ax.plot(x_range, y_pred, color=color2, linewidth=3, label=f"{label} (trend)")

    # Split the groups for plotting
    deflector_group_plot = filtered_df[filtered_df["VesselId"].isin(vessel_with_deflectors)]
    non_deflector_group_plot = filtered_df[~filtered_df["VesselId"].isin(vessel_with_deflectors)]

    plot_scatter_and_fit(deflector_group_plot, "With Deflector", "green","yellow", degree)  
    plot_scatter_and_fit(non_deflector_group_plot, "Without Deflector", "blue", "orange",degree)

    # Add vertical lines to show speed ranges
    for range_start, range_end in speed_ranges:
        ax.axvline(x=range_start, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=range_end, color='gray', linestyle='--', alpha=0.3)

    # Labels and Legend
    ax.set_xlabel("SpeedOG(knots)", fontsize=12)
    ax.set_ylabel("FOCWindPower(MT/day)", fontsize=12)
    ax.set_title("FOCWindPower vs SpeedOG", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    