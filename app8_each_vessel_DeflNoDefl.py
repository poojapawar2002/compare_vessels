import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pymongo
import math

# --- Load CSV ---


mongo_url = st.secrets["mongo"]["uri"]
client = pymongo.MongoClient(mongo_url)
db = client["seaker_data"]
collection = db["final_combined_output_new"]
df = pd.DataFrame(list(collection.find()))
df.drop(columns=["_id"], inplace=True)

# df = pd.read_csv("final_combined_output_new.csv")

df["WindSpeedUsed"] = df["WindSpeedUsed"] * 1.94384  # Convert m/s to knots

required_cols = ["VesselId", "WindSpeedUsed", "MeanDraft", "SpeedOG", "MEFOCDeviation", "IsDeltaFOCMEValid", "IsSpeedDropValid", "FOCWindPowerDeflector", "FOCWindPowerNoDeflector", "RelativeWindDirection", "ME1RunningHoursMinute", "StartDateUTC", "EndDateUTC"]
df = df[required_cols]
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

    # Convert vessel IDs to names for UI - Single selection
    available_ids = df["VesselId"].unique()
    available_names = [vessel_names.get(vid, f"Unknown ({vid})") for vid in available_ids]
    selected_vessel_name = st.sidebar.selectbox("Select Vessel", available_names)

    st.title(f"FOCWindPower vs SpeedOG for {selected_vessel_name}")

    # Map name back to ID for filtering
    name_to_id = {v: k for k, v in vessel_names.items()}
    selected_vessel_id = name_to_id[selected_vessel_name]

    st.sidebar.subheader("Wind Speed Range (knots)")
    wind_min = st.sidebar.number_input("Min Wind Speed", value=float(df["WindSpeedUsed"].min()), step=0.5)
    wind_max = st.sidebar.number_input("Max Wind Speed", value=float(df["WindSpeedUsed"].max()), step=0.5)

    st.sidebar.subheader("Draft Range (m)")
    draft_min = st.sidebar.number_input("Min Draft", value=float(df["MeanDraft"].min()), step=0.5)
    draft_max = st.sidebar.number_input("Max Draft", value=float(df["MeanDraft"].max()), step=0.5)

    st.sidebar.subheader("SpeedOG Range (knots)")
    speedOG_min = st.sidebar.number_input("Min SpeedOG", value=float(df["SpeedOG"].min()), step=0.5)
    speedOG_max = st.sidebar.number_input("Max SpeedOG", value=float(df["SpeedOG"].max()), step=0.5)

    degree = 2
    range_width = 1

    # Apply initial data quality filters
    df = df[
        (df["MEFOCDeviation"] >= 0) &
        (df["MEFOCDeviation"] <= 100) &
        (df["IsDeltaFOCMEValid"] == 1) & 
        (df["IsSpeedDropValid"] == 1) &
        (df["FOCWindPowerDeflector"] >= 0) &
        (df["FOCWindPowerNoDeflector"] >= 0) 
    ]

    

    # Convert date columns to datetime if they're not already
    df["StartDateUTC"] = pd.to_datetime(df["StartDateUTC"])
    df["EndDateUTC"] = pd.to_datetime(df["EndDateUTC"])

    df = df.dropna(subset=["VesselId", "WindSpeedUsed", "MeanDraft", "SpeedOG", "FOCWindPowerDeflector", "FOCWindPowerNoDeflector", "RelativeWindDirection", "StartDateUTC", "EndDateUTC"])

    # Apply user filters for selected vessel
    filtered_df = df[
        (df["VesselId"] == selected_vessel_id) &
        (df["WindSpeedUsed"] > wind_min) & (df["WindSpeedUsed"] <= wind_max) &
        (df["MeanDraft"] > draft_min) & (df["MeanDraft"] <= draft_max) & 
        (df["SpeedOG"] >= speedOG_min) & (df["SpeedOG"] <= speedOG_max) 
    ]

    # Create speed ranges based on user input
    def create_speed_ranges(min_speed, max_speed, width):
        ranges = []
        current = min_speed
        while current < max_speed:
            range_end = min(current + width, max_speed)
            ranges.append((current, range_end))
            current += width
        return ranges

    if not filtered_df.empty:
        speed_ranges = create_speed_ranges(math.floor(filtered_df["SpeedOG"].min()), math.ceil(filtered_df["SpeedOG"].max()), range_width)
    else:
        speed_ranges = []
    
    # Function to calculate statistics for each speed range
    def calculate_range_stats(df, speed_ranges):
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
                    'WithDeflector_Mean': None,
                    'WithDeflector_Count': 0,
                    'WithoutDeflector_Mean': None,
                    'WithoutDeflector_Count': 0
                })
                continue
            
            # Calculate means for both deflector scenarios
            deflector_mean = range_data["FOCWindPowerDeflector"].mean()
            no_deflector_mean = range_data["FOCWindPowerNoDeflector"].mean()
            
            results.append({
                'SpeedRange': f"({range_start:.1f}, {range_end:.1f}]",
                'WithDeflector_Mean': deflector_mean,
                'WithDeflector_Count': len(range_data),
                'WithoutDeflector_Mean': no_deflector_mean,
                'WithoutDeflector_Count': len(range_data)
            })
        
        return pd.DataFrame(results)

    # Calculate statistics
    if not filtered_df.empty:
        # --- ADD SAVINGS CALCULATION SECTION HERE ---
        # Calculate total savings for the filtered data
        total_without_deflector = filtered_df["FOCWindPowerNoDeflector"].sum()
        total_with_deflector = filtered_df["FOCWindPowerDeflector"].sum()
        total_savings = total_without_deflector - total_with_deflector
        
        # Calculate overall date range and total running hours for display
        overall_start_date = filtered_df["StartDateUTC"].min().strftime("%Y-%m-%d")
        overall_end_date = filtered_df["EndDateUTC"].max().strftime("%Y-%m-%d")
        total_running_hours = filtered_df["ME1RunningHoursMinute"].sum() / 60  # Convert minutes to hours
        total_savings_mt_day = total_savings / (total_running_hours / 24) if total_running_hours > 0 else 0
        
        # Conditional rendering based on vessel ID
        if selected_vessel_id in [1004, 1007, 1018]:
            title_text = f"Total Fuel Savings for {selected_vessel_name}"
        else:
            title_text = f"Total Fuel Savings for {selected_vessel_name} <i>if Deflector Was Installed</i>"
        
        # Collapsible Fuel Savings Section
        with st.expander("🔋 **Fuel Savings Summary** - Click to view details", expanded=False):
            # Display Total Savings with improved styling
            st.markdown("""
                <style>
                    .savings-card {
                        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
                        color: white;
                        padding: 25px;
                        border-radius: 12px;
                        text-align: center;
                        font-family: 'Segoe UI', sans-serif;
                        margin: 15px 0;
                        box-shadow: 0 4px 15px rgba(0, 184, 148, 0.3);
                    }
                    .savings-subtitle {
                        font-size: 18px;
                        font-weight: 500;
                        margin: 8px 0;
                    }
                    .savings-value {
                        font-size: 48px;
                        font-weight: 700;
                        margin: 20px 0;
                        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                    }
                    .savings-details {
                        font-size: 16px;
                        font-weight: 400;
                        margin-top: 5px;
                    }
                </style>
            """, unsafe_allow_html=True)
            
            # Display the savings card
            st.markdown(f"""
                <div class="savings-card">
                    <div class="savings-subtitle">{title_text}</div>
                    <div class="savings-value">{total_savings_mt_day:.2f} MT/day</div>
                    <div class="savings-subtitle">From {overall_start_date} to {overall_end_date}</div>
                    <div class="savings-details">
                        After applying current filters:<br>
                        <b>Total fuel saved:</b> {total_savings:.2f} MT &nbsp; | &nbsp;
                        <b>Total Running Hours:</b> {total_running_hours:.1f} hours
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # Convert FOC values (assuming same conversion needed)
        filtered_df["FOCWindPowerDeflector"] = (filtered_df["FOCWindPowerDeflector"]) * (1440/filtered_df["ME1RunningHoursMinute"])
        filtered_df["FOCWindPowerNoDeflector"] = (filtered_df["FOCWindPowerNoDeflector"]) * (1440/filtered_df["ME1RunningHoursMinute"])
        
        # --- CONTINUE WITH EXISTING CODE ---
        stats_df = calculate_range_stats(filtered_df, speed_ranges)
        
        # Format for display
        display_df = stats_df.copy()
        display_df['WithDeflector_Mean'] = display_df['WithDeflector_Mean'].apply(
            lambda x: f"{x:.4f}" if pd.notna(x) else "None"
        )
        display_df['WithoutDeflector_Mean'] = display_df['WithoutDeflector_Mean'].apply(
            lambda x: f"{x:.4f}" if pd.notna(x) else "None"
        )
        
        # Rename columns for better display
        display_df = display_df.rename(columns={
            'SpeedRange': 'Speed Range',
            'WithDeflector_Mean': 'Avg FOCWindPower With Deflector',
            'WithoutDeflector_Mean': 'Avg FOCWindPower Without Deflector'
        })
        
        # Select columns to display
        display_columns = ['Speed Range', 'Avg FOCWindPower With Deflector', 'Avg FOCWindPower Without Deflector']
        
        # Collapsible SpeedOG Ranges Table
        with st.expander("📊 **FOCWindPower Over SpeedOG Ranges** - Click to view table", expanded=False):
            st.dataframe(display_df[display_columns], height=300, use_container_width=True)

        # Calculate overall averages for the title
        deflector_avg = filtered_df["FOCWindPowerDeflector"].mean()
        no_deflector_avg = filtered_df["FOCWindPowerNoDeflector"].mean()
        
        # Plot Setup - Use full width
        fig, ax = plt.subplots(figsize=(16, 10))  # Increased size for full page width
        
        # Set plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        ax.set_facecolor('#fafafa')

        def plot_scatter_and_fit(x_data, y_data, label, color1, color2, degree):
            if len(x_data) == 0:
                return

            # Scatter plot of raw data
            ax.scatter(x_data, y_data, label=label, color=color1, alpha=0.7, s=40, edgecolors='white', linewidth=0.5)

            # Polynomial Fit
            x = x_data.values.reshape(-1, 1)
            y = y_data.values

            poly = PolynomialFeatures(degree=degree)
            x_poly = poly.fit_transform(x)
            model = LinearRegression().fit(x_poly, y)

            # Predict over sorted x for smooth line
            x_range = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
            y_pred = model.predict(poly.transform(x_range))

            ax.plot(x_range, y_pred, color=color2, linewidth=3, label=f"{label} (trend)", alpha=0.9)

        # Plot both deflector scenarios for the selected vessel
        plot_scatter_and_fit(filtered_df["SpeedOG"], filtered_df["FOCWindPowerDeflector"], 
                           "With Deflector", "#2E8B57", "#FF6B35", degree)  # SeaGreen scatter, OrangeRed trend
        plot_scatter_and_fit(filtered_df["SpeedOG"], filtered_df["FOCWindPowerNoDeflector"], 
                           "Without Deflector", "#4169E1", "#DC143C", degree)  # RoyalBlue scatter, Crimson trend

        # Add vertical lines to show speed ranges
        for range_start, range_end in speed_ranges:
            ax.axvline(x=range_start, color='lightgray', linestyle=':', alpha=0.5, linewidth=1)
            ax.axvline(x=range_end, color='lightgray', linestyle=':', alpha=0.5, linewidth=1)

        # Enhanced styling
        ax.set_xlabel("SpeedOG (knots)", fontsize=16, fontweight='bold', color='#2E4057')
        ax.set_ylabel("FOCWindPower (MT/day)", fontsize=16, fontweight='bold', color='#2E4057')
        
        # Create title with vessel name and averages
        title_parts = [f"FOCWindPower vs SpeedOG - {selected_vessel_name}"]
        title_parts.append(f"Avg: With Deflector={deflector_avg:.3f} MT/day, Without Deflector={no_deflector_avg:.3f} MT/day")
        
        ax.set_title("\n".join(title_parts), 
                    fontsize=18, fontweight='bold', color='#2E4057', pad=20)
        
        # Enhanced legend
        legend = ax.legend(fontsize=13, frameon=True, fancybox=True, shadow=True, 
                         framealpha=0.9, facecolor='white', edgecolor='gray')
        legend.get_frame().set_linewidth(1.5)
        
        # Grid styling
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Spines styling
        for spine in ax.spines.values():
            spine.set_edgecolor('#cccccc')
            spine.set_linewidth(1)
        
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)  # Use full container width
    
    else:
        st.warning("⚠️ No data available for the selected vessel and filter combination.")
        st.info("Please adjust your filter settings to see data.")