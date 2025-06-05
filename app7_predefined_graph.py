import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import math

# --- Page Configuration ---
st.set_page_config(page_title="FOCWindPower Analysis", layout="wide")

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
    
    .filter-section {
        background: linear-gradient(135deg, #ff7675, #fd79a8);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1rem 0 2rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
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
st.markdown('<h1 class="main-header">⚓ FOCWindPower vs SpeedOG - Multi-Range Analysis</h1>', unsafe_allow_html=True)

df = pd.read_csv("final_combined_output.csv")

required_cols = ["VesselId", "WindSpeedUsed", "MeanDraft", "SpeedOG", "MEFOCDeviation", "IsDeltaFOCMEValid", "IsSpeedDropValid","FOCWindPower", "RelativeWindDirection"]
if not all(col in df.columns for col in required_cols):
    st.error(f"CSV must include: {', '.join(required_cols)}")
else:
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
    
    # Apply initial data quality filters
    df = df[
        (df["MEFOCDeviation"] >= 0) &
        (df["MEFOCDeviation"] <= 100) &
        (df["IsDeltaFOCMEValid"] == 1) & 
        (df["IsSpeedDropValid"] == 1) & 
        (df["RelativeWindDirection"] >= 0) &
        (df["RelativeWindDirection"] <= 45) 
    ]

    df["FOCWindPower"] = (df["FOCWindPower"])*(1440/df["ME1RunningHoursMinute"])

    # --- RELATIVE WIND DIRECTION FILTER SECTION ---
    st.info("✅ We have considered here **Head Wind conditions** only (Relative Wind Direction between -45° and 45°).")


    if df.empty:
        st.error("❌ No data remains after applying the relative wind direction filter. Please adjust the range.")
        st.stop()

    # Define predefined ranges
    draft_ranges = [(12.0, 12.5), (12.5, 13.0), (13.0, 13.5), (13.5, 14.0), (14.0, 14.5), (14.5, 15.0)]
    wind_ranges = [(0, 5), (5, 10), (10, 15)]
    
    degree = 2
    range_width = 1  # for speed ranges
    
    # Set default values for SpeedOG (after filtering)
    speedOG_min = float(df["SpeedOG"].min())
    speedOG_max = float(df["SpeedOG"].max())

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
    def calculate_range_stats(df, speed_ranges, deflector_vessel_ids):
        results = []
        
        for range_start, range_end in speed_ranges:
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
            
            with_deflectors = range_data[range_data["VesselId"].isin(deflector_vessel_ids)]
            without_deflectors = range_data[~range_data["VesselId"].isin(deflector_vessel_ids)]
            
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

    # Function to plot scatter and fit
    def plot_scatter_and_fit(ax, group, label, color1, color2, degree):
        if group.empty:
            return None, None

        # Scatter plot of raw data
        ax.scatter(group["SpeedOG"], group["FOCWindPower"], label=label, color=color1, alpha=0.7, s=40, edgecolors='white', linewidth=0.5)

        # Polynomial Fit
        x = group["SpeedOG"].values.reshape(-1, 1)
        y = group["FOCWindPower"].values

        poly = PolynomialFeatures(degree=degree)
        x_poly = poly.fit_transform(x)
        model = LinearRegression().fit(x_poly, y)

        # Predict over sorted x for smooth line
        x_range = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
        y_pred = model.predict(poly.transform(x_range))

        ax.plot(x_range, y_pred, color=color2, linewidth=3, label=f"{label} (trend)", alpha=0.9)
        
        return group["FOCWindPower"].mean(), len(group)

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
                st.markdown(f'<div class="wind-header">💨 Wind: {wind_min}-{wind_max} m/s</div>', unsafe_allow_html=True)
                
                # Filter data for current draft and wind ranges (relative wind direction already applied)
                filtered_df = df[
                    (df["MeanDraft"] > draft_min) & (df["MeanDraft"] <= draft_max) & 
                    (df["WindSpeedUsed"] >= wind_min) & (df["WindSpeedUsed"] < wind_max) &
                    (df["SpeedOG"] >= speedOG_min) & (df["SpeedOG"] <= speedOG_max)
                ]
                
                # Calculate and display speed-wise statistics table
                if not filtered_df.empty:
                    speed_ranges = create_speed_ranges(math.floor(filtered_df["SpeedOG"].min()), math.ceil(filtered_df["SpeedOG"].max()), range_width)
                    stats_df = calculate_range_stats(filtered_df, speed_ranges, vessel_with_deflectors)
                    
                    # Format for display
                    display_df = stats_df.copy()
                    display_df['WithDeflectors_Mean'] = display_df['WithDeflectors_Mean'].apply(
                        lambda x: f"{x:.4f}" if pd.notna(x) else "None"
                    )
                    display_df['WithoutDeflectors_Mean'] = display_df['WithoutDeflectors_Mean'].apply(
                        lambda x: f"{x:.4f}" if pd.notna(x) else "None"
                    )
                    
                    # Rename columns
                    display_df = display_df.rename(columns={
                        'SpeedRange': 'Speed Range (knots)',
                        'WithDeflectors_Mean': 'With Deflector',
                        'WithoutDeflectors_Mean': 'Without Deflector'
                    })
                    
                    # Create expandable section for the table
                    with st.expander("📊 Speed-wise FOC Analysis", expanded=False):
                        st.dataframe(
                            display_df[['Speed Range (knots)', 'With Deflector', 'Without Deflector']], 
                            use_container_width=True, 
                            hide_index=True
                        )
                    
                    # Calculate overall weighted averages (removed for plot horizontal lines, but kept for title)
                    with_deflector_avg = stats_df[stats_df['WithDeflectors_Mean'].notna()]['WithDeflectors_Mean'].mean()
                    without_deflector_avg = stats_df[stats_df['WithoutDeflectors_Mean'].notna()]['WithoutDeflectors_Mean'].mean()
                else:
                    st.warning("⚠️ No data available for this range combination.")
                    with_deflector_avg = None
                    without_deflector_avg = None
                
                # Create individual plot with larger size
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Set plot style
                plt.style.use('seaborn-v0_8-whitegrid')
                ax.set_facecolor('#fafafa')
                
                # Split the groups for plotting
                deflector_group = filtered_df[filtered_df["VesselId"].isin(vessel_with_deflectors)]
                non_deflector_group = filtered_df[~filtered_df["VesselId"].isin(vessel_with_deflectors)]

                # Plot data with better color combinations
                # Option 1: Green/Teal theme
                plot_scatter_and_fit(ax, deflector_group, "With Deflector", "#2E8B57", "#FF6B35", degree)  # SeaGreen scatter, OrangeRed trend
                plot_scatter_and_fit(ax, non_deflector_group, "Without Deflector", "#4169E1", "#DC143C", degree)  # RoyalBlue scatter, Crimson trend
                
                # Option 2: Purple/Blue theme (uncomment to use instead)
                # plot_scatter_and_fit(ax, deflector_group, "With Deflector", "#8A2BE2", "#FF4500", degree)  # BlueViolet scatter, OrangeRed trend
                # plot_scatter_and_fit(ax, non_deflector_group, "Without Deflector", "#1E90FF", "#B22222", degree)  # DodgerBlue scatter, FireBrick trend

                # Add vertical lines to show speed ranges
                if not filtered_df.empty:
                    for range_start, range_end in speed_ranges:
                        ax.axvline(x=range_start, color='lightgray', linestyle=':', alpha=0.5, linewidth=1)
                        ax.axvline(x=range_end, color='lightgray', linestyle=':', alpha=0.5, linewidth=1)

                # Enhanced styling
                ax.set_xlabel("SpeedOG (knots)", fontsize=14, fontweight='bold', color='#2E4057')
                ax.set_ylabel("FOCWindPower (MT/day)", fontsize=14, fontweight='bold', color='#2E4057')
                
                # Create title with averages instead of counts
                title_parts = ["FOCWindPower vs SpeedOG"]
                
                if with_deflector_avg is not None and without_deflector_avg is not None:
                    title_parts.append(f"Avg With Deflector={with_deflector_avg:.3f} MT/day, Without Deflector={without_deflector_avg:.3f} MT/day")
                elif with_deflector_avg is not None:
                    title_parts.append(f"Avg With Deflector={with_deflector_avg:.3f}")
                elif without_deflector_avg is not None:
                    title_parts.append(f"Avg Without Deflector={without_deflector_avg:.3f}")
                
                ax.set_title("\n".join(title_parts), 
                            fontsize=16, fontweight='bold', color='#2E4057', pad=20)
                
                # Enhanced legend
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