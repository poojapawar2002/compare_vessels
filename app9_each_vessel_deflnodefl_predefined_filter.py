import streamlit as st
import pymongo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import math

# --- Page Configuration ---
st.set_page_config(page_title="FOCWindPower Analysis by Vessel", layout="wide")

# --- Custom CSS for Better Styling ---
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E4057;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .wind-header {
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        color: white;
        padding: 0.8rem;
        border-radius: 8px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 500;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .savings-card {
        background: linear-gradient(135deg, #00b894, #00a085);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    
    .savings-value {
        font-size: 3rem;
        font-weight: 800;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .savings-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }
    
    .stExpander > div:first-child {
        background-color: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }
    
    .plot-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Load CSV ---
# st.markdown('<h1 class="main-header">⚓ FOCWindPower vs SpeedOG - Vessel Analysis</h1>', unsafe_allow_html=True)

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
    # Vessel selection in sidebar
    st.sidebar.header("Vessel Selection")
    
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

    # Map name back to ID for filtering
    name_to_id = {v: k for k, v in vessel_names.items()}
    selected_vessel_id = name_to_id[selected_vessel_name]

    # Additional filters in sidebar
    st.sidebar.subheader("Additional Filters")
    
    # Get data for selected vessel to set filter ranges
    vessel_data = df[df["VesselId"] == selected_vessel_id]
    
    st.sidebar.subheader("SpeedOG Range (knots)")
    speedOG_min = st.sidebar.number_input("Min SpeedOG", value=float(vessel_data["SpeedOG"].min()) if not vessel_data.empty else 0.0, step=0.5)
    speedOG_max = st.sidebar.number_input("Max SpeedOG", value=float(vessel_data["SpeedOG"].max()) if not vessel_data.empty else 20.0, step=0.5)

    # st.sidebar.subheader("Relative Wind Direction Range (degrees)")
    # rel_wind_dir_min = st.sidebar.number_input("Min Relative Wind Direction", value=float(vessel_data["RelativeWindDirection"].min()) if not vessel_data.empty else 0.0, step=0.5)
    # rel_wind_dir_max = st.sidebar.number_input("Max Relative Wind Direction", value=float(vessel_data["RelativeWindDirection"].max()) if not vessel_data.empty else 360.0, step=0.5)
    
    df['StartDateUTC'] = pd.to_datetime(df['StartDateUTC'], format = "%d-%m-%Y %H:%M")
    df['EndDateUTC'] = pd.to_datetime(df['EndDateUTC'], format = "%d-%m-%Y %H:%M")

    # Apply initial data quality filters
    df = df[
        (df["MEFOCDeviation"] >= 0) &
        (df["MEFOCDeviation"] <= 100) &
        (df["IsDeltaFOCMEValid"] == 1) & 
        (df["IsSpeedDropValid"] == 1) & 
        (df["FOCWindPowerDeflector"]>=0) &
        (df["FOCWindPowerNoDeflector"]>=0) 
    ]

    df = df.dropna(subset=["VesselId", "StartDateUTC","EndDateUTC","WindSpeedUsed", "MeanDraft", "SpeedOG", "FOCWindPowerDeflector", "FOCWindPowerNoDeflector", "RelativeWindDirection"])

    if df[df["VesselId"] == selected_vessel_id].shape[0] == 0:
        st.error("No valid data available after filtering. Please adjust your filters.")
        st.stop()
        exit()

    

    # Calculate total savings for the selected vessel (with all filters applied)
    overall_filtered_df = df[
        (df["VesselId"] == selected_vessel_id) &
        (df["SpeedOG"] >= speedOG_min) & (df["SpeedOG"] <= speedOG_max) 
        # (df["RelativeWindDirection"] >= rel_wind_dir_min) & (df["RelativeWindDirection"] <= rel_wind_dir_max)
    ]

    st.markdown(f'<div class="section-header">🚢 Wind Deflection Analysis for {selected_vessel_name}</div>', unsafe_allow_html=True)
    
    if not overall_filtered_df.empty:
        total_without_deflector = overall_filtered_df["FOCWindPowerNoDeflector"].sum()
        total_with_deflector = overall_filtered_df["FOCWindPowerDeflector"].sum()
        total_savings = total_without_deflector - total_with_deflector
        
        # Calculate overall date range and total running hours for display
        overall_start_date = overall_filtered_df["StartDateUTC"].min().strftime("%Y-%m-%d")
        overall_end_date = overall_filtered_df["EndDateUTC"].max().strftime("%Y-%m-%d")
        total_running_hours = overall_filtered_df["ME1RunningHoursMinute"].sum() / 60  # Convert minutes to hours
        total_savings_mt_day = total_savings / (total_running_hours / 24) if total_running_hours > 0 else 0
        
        # Display Total Savings prominently at the top
        # Shared CSS style for both cases
        st.markdown("""
            <style>
                .savings-card {
                    background-color: #00b894;
                    color: white;
                    padding: 25px;
                    border-radius: 12px;
                    text-align: center;
                    font-family: 'Segoe UI', sans-serif;
                    margin-bottom: 20px;
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
                }
                .savings-details {
                    font-size: 16px;
                    font-weight: 400;
                    margin-top: 5px;
                }
            </style>
        """, unsafe_allow_html=True)

        # Conditional rendering based on vessel ID
        if selected_vessel_id in [1004, 1007, 1018]:
            title_text = f"Total Fuel Savings for {selected_vessel_name}"
        else:
            title_text = f"Total Fuel Savings for {selected_vessel_name} <i>if Deflector Was Installed</i>"

        # Markdown content
        st.markdown(f"""
            <div class="savings-card">
                <div class="savings-subtitle">{title_text}</div>
                <div class="savings-value">{total_savings_mt_day:.2f} MT/day</div>
                <div class="savings-subtitle">From {overall_start_date} to {overall_end_date}</div>
                <div class="savings-details">
                    After removing outliers:<br>
                    <b>Total fuel saved:</b> {total_savings:.2f} MT &nbsp; | &nbsp;
                    <b>Total Running Hours:</b> {total_running_hours:.1f} hours
                </div>
            </div>
        """, unsafe_allow_html=True)



    # Convert FOC values
    overall_filtered_df["FOCWindPowerDeflector"] = (overall_filtered_df["FOCWindPowerDeflector"]) * (1440/overall_filtered_df["ME1RunningHoursMinute"])
    overall_filtered_df["FOCWindPowerNoDeflector"] = (overall_filtered_df["FOCWindPowerNoDeflector"]) * (1440/overall_filtered_df["ME1RunningHoursMinute"])

    # Define predefined ranges
    draft_ranges = [(12.0, 12.5), (12.5, 13.0), (13.0, 13.5), (13.5, 14.0), (14.0, 14.5), (14.5, 15.0)]
    wind_ranges = [(0, 10), (10, 20), (20, 30)]
    
    degree = 2
    range_width = 1  # for speed ranges
    
    # Function to create speed ranges
    def create_speed_ranges(min_speed, max_speed, width):
        ranges = []
        current = min_speed
        while current < max_speed:
            range_end = min(current + width, max_speed)
            ranges.append((current, range_end))
            current += width
        return ranges

    # Function to calculate statistics for each speed range
    def calculate_range_stats(df, speed_ranges):
        results = []
        
        for range_start, range_end in speed_ranges:
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
            
            with_mean = range_data["FOCWindPowerDeflector"].mean()
            without_mean = range_data["FOCWindPowerNoDeflector"].mean()
            
            results.append({
                'SpeedRange': f"({range_start:.1f}, {range_end:.1f}]",
                'WithDeflector_Mean': with_mean,
                'WithDeflector_Count': len(range_data),
                'WithoutDeflector_Mean': without_mean,
                'WithoutDeflector_Count': len(range_data)
            })
        
        return pd.DataFrame(results)

    # Function to plot scatter and fit
    def plot_scatter_and_fit(ax, x_data, y_data, label, color1, color2, degree):
        if len(x_data) == 0:
            return None, None

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
        
        return y_data.mean(), len(x_data)

    # Main display section
    # st.markdown(f'<div class="section-header">🚢 Analysis for {selected_vessel_name}</div>', unsafe_allow_html=True)

    # Create individual plots for each combination
    plot_count = 1
    
    # Organize plots in 3 columns using Streamlit columns
    for i, (draft_min, draft_max) in enumerate(draft_ranges):
        st.markdown(f'<div class="section-header">🚢 Draft Range: {draft_min} - {draft_max} meters</div>', unsafe_allow_html=True)
        
        # Create 3 columns for wind speed ranges
        col1, col2, col3 = st.columns(3, gap="large")
        columns = [col1, col2, col3]
        
        for j, (wind_min, wind_max) in enumerate(wind_ranges):
            with columns[j]:
                st.markdown(f'<div class="wind-header">💨 Wind: {wind_min}-{wind_max} knots</div>', unsafe_allow_html=True)
                
                # Filter data for current vessel, draft and wind ranges
                filtered_df = overall_filtered_df[
                    (overall_filtered_df["VesselId"] == selected_vessel_id) &
                    (overall_filtered_df["MeanDraft"] > draft_min) & (overall_filtered_df["MeanDraft"] <= draft_max) & 
                    (overall_filtered_df["WindSpeedUsed"] >= wind_min) & (overall_filtered_df["WindSpeedUsed"] < wind_max) &
                    (overall_filtered_df["SpeedOG"] >= speedOG_min) & (overall_filtered_df["SpeedOG"] <= speedOG_max) 
                    # (df["RelativeWindDirection"] >= rel_wind_dir_min) & (df["RelativeWindDirection"] <= rel_wind_dir_max)
                ]
                
                # Calculate and display speed-wise statistics table
                if not filtered_df.empty:
                    speed_ranges = create_speed_ranges(math.floor(filtered_df["SpeedOG"].min()), math.ceil(filtered_df["SpeedOG"].max()), range_width)
                    stats_df = calculate_range_stats(filtered_df, speed_ranges)
                    
                    # Format for display
                    display_df = stats_df.copy()
                    display_df['WithDeflector_Mean'] = display_df['WithDeflector_Mean'].apply(
                        lambda x: f"{x:.4f}" if pd.notna(x) else "None"
                    )
                    display_df['WithoutDeflector_Mean'] = display_df['WithoutDeflector_Mean'].apply(
                        lambda x: f"{x:.4f}" if pd.notna(x) else "None"
                    )
                    
                    # Rename columns
                    display_df = display_df.rename(columns={
                        'SpeedRange': 'Speed Range (knots)',
                        'WithDeflector_Mean': 'With Deflector',
                        'WithoutDeflector_Mean': 'Without Deflector'
                    })
                    
                    # Create expandable section for the table
                    with st.expander("📊 Speed-wise FOC Analysis", expanded=False):
                        st.dataframe(
                            display_df[['Speed Range (knots)', 'With Deflector', 'Without Deflector']], 
                            use_container_width=True, 
                            hide_index=True
                        )
                    
                    # Calculate overall averages
                    deflector_avg = filtered_df["FOCWindPowerDeflector"].mean()
                    no_deflector_avg = filtered_df["FOCWindPowerNoDeflector"].mean()
                else:
                    st.warning("⚠️ No data available for this range combination.")
                    deflector_avg = None
                    no_deflector_avg = None
                
                # Create individual plot with larger size
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Set plot style
                plt.style.use('seaborn-v0_8-whitegrid')
                ax.set_facecolor('#fafafa')
                
                if not filtered_df.empty:
                    # Plot both deflector scenarios for the selected vessel
                    plot_scatter_and_fit(ax, filtered_df["SpeedOG"], filtered_df["FOCWindPowerDeflector"], 
                                       "With Deflector", "#2E8B57", "#FF6B35", degree)  # SeaGreen scatter, OrangeRed trend
                    plot_scatter_and_fit(ax, filtered_df["SpeedOG"], filtered_df["FOCWindPowerNoDeflector"], 
                                       "Without Deflector", "#4169E1", "#DC143C", degree)  # RoyalBlue scatter, Crimson trend

                    # Add vertical lines to show speed ranges
                    for range_start, range_end in speed_ranges:
                        ax.axvline(x=range_start, color='lightgray', linestyle=':', alpha=0.5, linewidth=1)
                        ax.axvline(x=range_end, color='lightgray', linestyle=':', alpha=0.5, linewidth=1)

                # Enhanced styling
                ax.set_xlabel("SpeedOG (knots)", fontsize=14, fontweight='bold', color='#2E4057')
                ax.set_ylabel("FOCWindPower (MT/day)", fontsize=14, fontweight='bold', color='#2E4057')
                
                # Create title with averages
                title_parts = [f"FOCWindPower vs SpeedOG - {selected_vessel_name}"]
                
                if deflector_avg is not None and no_deflector_avg is not None:
                    title_parts.append(f"Avg: With Deflector={deflector_avg:.3f} MT/day, Without Deflector={no_deflector_avg:.3f} MT/day")
                elif deflector_avg is not None:
                    title_parts.append(f"Avg With Deflector={deflector_avg:.3f}")
                elif no_deflector_avg is not None:
                    title_parts.append(f"Avg Without Deflector={no_deflector_avg:.3f}")
                
                ax.set_title("\n".join(title_parts), 
                            fontsize=16, fontweight='bold', color='#2E4057', pad=20)
                
                # Enhanced legend
                if not filtered_df.empty:
                    legend = ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=True, 
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
                st.pyplot(fig, use_container_width=True)
                plt.close()  # Close the figure to free memory
                
                plot_count += 1

    # # Display additional summary statistics at the bottom
    # st.markdown(f'<div class="section-header">📊 Summary Statistics for {selected_vessel_name}</div>', unsafe_allow_html=True)
    
    # if not overall_filtered_df.empty:
    #     col1, col2, col3, col4 = st.columns(4)
        
    #     with col1:
    #         st.metric("Total Records", len(overall_filtered_df))
        
    #     with col2:
    #         overall_deflector_avg = overall_filtered_df["FOCWindPowerDeflector"].mean()
    #         st.metric("Avg FOC With Deflector", f"{overall_deflector_avg:.4f} MT/day")
        
    #     with col3:
    #         overall_no_deflector_avg = overall_filtered_df["FOCWindPowerNoDeflector"].mean()
    #         st.metric("Avg FOC Without Deflector", f"{overall_no_deflector_avg:.4f} MT/day")
        
    #     with col4:
    #         if overall_no_deflector_avg != 0:
    #             percentage_diff = ((overall_deflector_avg - overall_no_deflector_avg) / overall_no_deflector_avg) * 100
    #             st.metric("Percentage Difference", f"{percentage_diff:.2f}%")
        
    #     # Additional info
    #     st.info(f"**Total Fuel Savings:** {total_savings:.2f} MT ({'Positive savings' if total_savings > 0 else 'Negative savings - deflector uses more fuel'})")