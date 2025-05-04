import shlex
import sys
from pathlib import Path
import io
from contextlib import redirect_stdout
import random
import os
# Add these imports at the top of your app.py
import subprocess
import re # For parsing results

import pandas as pd
import pydeck as pdk
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px

# Add the parent directory to the system path to import utils
# Ensure this path logic correctly points to your project root from app.py's location
# If app.py is in 'src/', this should work. Adjust if structure is different.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.sql import get_table, get_circuits_by
# Make sure utilities and models are importable
from utils.utilities import get_random_sample, get_historical_cities
from models.clustering import kmeans_plot_elbow, scale_coords, clusterize_circuits # Import clustering functions
from data.app_data import F1_LOCATION_DESCRIPTIONS # Import descriptions

import functools # Needed for deap_toolbox registration if not already imported
from models import genetic_ops # Import your genetic operators module
# Import functions from run_ga.py (or their original modules)
# It's generally better to import directly from the modules where they are defined
# Assuming these functions are accessible via run_ga for now
from run_ga import (
    prepare_scenario,
    set_default_params,
    update_param, # If you implement interactive updates
    deap_toolbox,
    run_genetic_algorithm
)
# Import DEAP components (ensure DEAP is installed)
try:
    from deap import base, creator, tools, algorithms
except ImportError:
    st.error("❌ DEAP library not found. Please install it (`pip install deap`) and restart.")
    st.stop() # Stop execution if DEAP is missing
# --- End of added imports ---

SEED = 42  # Set a random seed for reproducibility
random.seed(SEED)  # Set the random seed for reproducibility
np.random.seed(SEED)  # Set the random seed for reproducibility


# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(
    page_title="F1 Green Flag | Sustainable Calendar",
    page_icon="🍃",  # Using a leaf emoji as an icon
    layout="wide",   # Use wide layout for more space
    initial_sidebar_state="expanded" # Keep sidebar open initially
)

# --- Custom CSS ---
# (Keep your existing CSS here - it defines the F1 style)
st.markdown("""
            <style>
            /* Import Google Font */
            @import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@400;600;700&display=swap');

            /* === General Styling === */
            body {
                color: #E0E0E0;
                background-color: #1E1E1E;
                font-family: 'Titillium Web', sans-serif;
                line-height: 1.6;
            }

            /* Container Padding */
            .main .block-container {
                padding: 2rem 1rem;
            }

            /* === Typography === */
            h1, h2, h3, h4, h5, h6 {
                color: #FFFFFF;
                font-weight: 700;
                margin-top: 1.5rem;
                margin-bottom: 0.5rem;
            }

            h1 {
                color: #FF1801;
                text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5);
                text-transform: uppercase;
                font-size: 2.5rem;
            }

            /* === Sidebar Styling === */
            [data-testid="stSidebar"] > div:first-child {
                background-color: #2a2a2a;
                border-right: 2px solid #FF1801;
            }

            [data-testid="stSidebar"] h1,
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3 {
                color: #FFFFFF;
            }

            [data-testid="stSidebar"] .stMarkdown,
            [data-testid="stSidebar"] .stRadio,
            [data-testid="stSidebar"] .stSelectbox {
                color: #E0E0E0;
            }

            [data-testid="stSidebar"] .stAlert {
                background-color: rgba(255, 24, 1, 0.1);
                border: 1px solid rgba(255, 24, 1, 0.5);
                border-radius: 5px;
            }

            /* === Tabs Styling === */
            .stTabs [data-baseweb="tab"] {
                background-color: #333333;
                color: #E0E0E0;
                border-radius: 5px 5px 0 0;
                margin: 0 5px;
                padding: 10px 16px;
                font-weight: 600;
                border-bottom: 3px solid transparent;
                transition: all 0.3s ease;
            }

            .stTabs [data-baseweb="tab"]:hover {
                background-color: #444444;
                color: #FFFFFF;
            }

            .stTabs [data-baseweb="tab"][aria-selected="true"] {
                background-color: #2a2a2a;
                color: #FFFFFF;
                font-weight: 700;
                border-bottom: 3px solid #FF1801;
            }

            .stTabs [data-testid="stVerticalBlock"] {
                background-color: none;
                border: none;
                border-top: none;
                border-radius: 0 0 5px 5px;
                padding: 1.5rem;
            }

            /* === DataFrame Styling === */
            .stDataFrame {
                border: 1px solid #444444;
                border-radius: 5px;
                background-color: #2a2a2a;
            }

            .stDataFrame thead th {
                background-color: #333333;
                color: #FFFFFF;
                font-weight: 600;
                padding: 10px;
            }

            .stDataFrame tbody td {
                color: #E0E0E0;
                padding: 8px;
            }

            /* === Button Styling === */
            .stButton button {
                background-color: transparent;
                color: #FF1801;
                border: 2px solid #FF1801;
                border-radius: 5px;
                font-weight: 700;
                padding: 10px 20px;
                text-transform: uppercase;
                transition: all 0.3s ease;
                cursor: pointer;
            }

            .stButton button:hover,
            .stButton button:focus {
                background-color: #FF1801;
                color: white;
                outline: none;
                box-shadow: none;
            }

            .stButton button:disabled {
                color: #555555;
                border-color: #555555;
                background-color: transparent;
                opacity: 0.5;
                cursor: not-allowed;
            }

            /* === Expander Styling === */
            .stExpander {
                background-color: #2a2a2a;
                border: 1px solid #444444;
                border-radius: 5px;
                margin-bottom: 1rem;
            }

            .stExpander header {
                font-weight: 600;
                color: #FFFFFF;
            }

            /* === Footer & Divider === */
            footer {
                color: #888888;
                font-size: 12px;
                text-align: center;
                padding-top: 2rem;
            }

            hr {
                border: none;
                border-top: 1px solid #444444;
                margin: 2rem 0;
            }
            
            [data-testid="column"] {
                background-color: transparent !important; /* Removes any white background */
                border: none !important;                 /* Removes borders */
                padding: 0 !important;                   /* Removes internal spacing */
                margin: 0 !important;
            }
            [data-testid="column"] > div {
                background-color: transparent !important;
                border: none !important;
            }
            
            </style>

            """, unsafe_allow_html=True)

# --- Helper Function for Capturing Output ---
def execute_and_capture( func, *args, **kwargs):
    """
    Executes a function with the given arguments and captures its stdout.

    Args:
        func (callable): The function to execute.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        tuple: A tuple containing the function's return value and the captured stdout as a string.
               Returns (None, error_message_string) if an exception occurs.
    """
    string_io = io.StringIO()
    result = None # Initialize result to None
    captured_string = ""
    try:
        # Redirect stdout to the buffer
        with redirect_stdout(string_io):
            # Execute the function and capture its return value
            result = func(*args, **kwargs)
        captured_string = string_io.getvalue()
    except Exception as e:
        # Capture any exceptions that occur within the block
        # Append error to any existing captured output
        captured_string += f"\n!!! ERROR during execution: {e} !!!"
        print(f"Error in execute_and_capture: {e}") # Also print to terminal for debugging
        return None, captured_string # Return None for result on error

    return result, captured_string

# --- Header ---
with st.container():
    st.title("🏎️ F1 Green Flag: Sustainable Calendar Optimization")
    st.caption("🍃 Analyzing F1 logistics using Clustering, Regression (WIP), and Genetic Algorithms to propose a greener race calendar.")
    st.divider()

# --- Navigation Tabs ---
# Added emojis to tab labels
page1, page2, page3, page4, page5 = st.tabs(
    ["🏠 Overview", "📊 Data Explorer", "📈 Regression Analysis", "🧩 Clustering Analysis", "⚙️ GA Optimization"]
)

# --- Page 1: Overview ---
with page1:
    st.header("ℹ️ Project Overview")
    st.markdown("""
    Welcome to the **F1 Green Flag** dashboard!

    This project analyzes Formula 1 logistics data to develop actionable recommendations for reducing the sport's environmental impact. By leveraging data science techniques, including **clustering**, **regression analysis (WIP)**, and **genetic algorithms**, we aim to optimize the F1 race calendar to minimize travel-related emissions for cars and equipment.

    **Core Components:**
    - 📊 **Data Collection & Preparation:** Consolidating circuit, geographical, and logistical data into the `planet_fone.db` database.
    - 🧩 **Clustering Analysis:** Identifying potential geographical groupings of races using K-Means (`models/clustering.py`).
    - 📈 **Regression Modeling (WIP):** Developing models to estimate emissions per travel leg.
    - ⚙️ **Genetic Algorithm Optimization:** Utilizing DEAP (`models/genetic_ops.py`, `run_ga.py`) to find near-optimal race sequences minimizing travel distance or estimated emissions, considering clustering insights.

    Use the tabs above to navigate through the different sections of our analysis and results.
    """)
    # Consider a more dynamic or relevant image if possible
    # st.image("path/to/your/f1_overview_image.png", caption="Visualizing the F1 Calendar Challenge")


# --- Page 2: Data Explorer ---
with page2:
    st.header("📊 Explore the F1 Data")
    st.markdown("""
    Dive into the datasets used for this analysis, primarily sourced from `planet_fone.db` and files within the `/data` directory.
    You can view raw data, summary statistics, and filter information.
    """)

    # --- Data Loading and Display ---
    st.subheader("🗺️ Circuit Geography Data Sample")
    geo_df = get_table("fone_geography")
    calendar_df = get_table("fone_calendar")
    logistics_df = get_table("travel_logistic")
    regression_df = get_table("training_regression_calendar")

    if not geo_df.empty:
        st.dataframe(geo_df.head(), use_container_width=False) # Show a sample of the geography data
        st.write(f"Loaded **{len(geo_df)}** Locations.")
        with st.expander("📂 Show Full Geography Data"):
            st.dataframe(geo_df, use_container_width=False)
    else:
        st.warning("⚠️ Could not load geography data.")

    st.subheader("📍 Map of Circuit Locations")
    if not geo_df.empty:
        if 'latitude' in geo_df.columns and 'longitude' in geo_df.columns:
            seasons = sorted(calendar_df['year'].unique(), reverse=True) if not calendar_df.empty else []
            # Improved label for selectbox
            selected_season = st.selectbox("🗓️ Select Season to Highlight Circuits:", options=["✨ New Potential"] + list(seasons), index=0, key="season_selector")

            if selected_season == "✨ New Potential" and not calendar_df.empty:
                visited_circuits = calendar_df['geo_id'].unique()
                map_data = geo_df[~geo_df['id'].isin(visited_circuits)]
                map_title = "Potential New Circuits"
            elif selected_season != "✨ New Potential" and not calendar_df.empty:
                season_circuits = calendar_df[calendar_df['year'] == selected_season]['geo_id'].unique()
                map_data = geo_df[geo_df['id'].isin(season_circuits)]
                map_title = f"Circuits in {selected_season} Season"
            else:
                map_data = pd.DataFrame()
                map_title = "No Circuits to Display"

            st.write(f"**{map_title}**") # Display title above map
            if not map_data.empty:
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=map_data,
                    get_position=["longitude", "latitude"],
                    get_radius=100000,
                    get_fill_color=[255, 24, 1, 180],  # Use F1 Red with transparency
                    pickable=True,
                )
                tooltip = {
                    "html": "<b>📍 {circuit_x}</b><br><b>City:</b> {city_x}<br><b>Country:</b> {country_x}<br><b>Continent:</b> {continent}",
                    "style": {"backgroundColor": "rgba(30, 30, 30, 0.85)", "color": "white", "border": "1px solid #FF1801"} # Style tooltip
                }
                st.pydeck_chart(
                    pdk.Deck(
                        map_style='mapbox://styles/mapbox/dark-v10', # Use a dark map style
                        layers=[layer],
                        initial_view_state=pdk.ViewState(
                            latitude=map_data["latitude"].mean(),
                            longitude=map_data["longitude"].mean(),
                            zoom=1.5, # Slightly adjusted zoom
                            pitch=30, # Add some pitch
                        ),
                        tooltip=tooltip,
                    )
                    , use_container_width=False
                )
            else:
                st.info("ℹ️ No map data available for the selected option.")

        else:
            st.warning("⚠️ Geography data does not contain required latitude and longitude columns.")
    else:
        st.warning("⚠️ Could not load geography data to display the map.")

    st.subheader(f"🏙️ City Details: {selected_season if selected_season != '✨ New Potential' else 'Potential New'}")
    if not map_data.empty:
        city_list = sorted(map_data['city_x'].unique()) # Sort city list
        col_select, col_details = st.columns([1, 6]) # Adjust column ratio

        with col_select:
             # Use selectbox for longer lists, radio for shorter
            if len(city_list) > 10:
                 selected_city = st.selectbox("Select City:", options=city_list, key="city_selector", index=0 )
            elif city_list: # Check if list is not empty
                 selected_city = st.radio("Select City:", options=city_list, key="city_radio")
            else:
                 selected_city = None
                 st.write("No cities in selection.")

        with col_details:
            if selected_city:
                city_details_row = map_data[map_data['city_x'] == selected_city].iloc[0] # Get the first row for the city

                # Format months to avoid
                try:
                    months_str = city_details_row.get('months_to_avoid', '[]') # Default to empty list string
                    if isinstance(months_str, str) and months_str.strip("[]"):
                        months_list = [int(m.strip()) for m in months_str.strip("[]").split(',') if m.strip()]
                        mmm_to_avoid = ', '.join(pd.to_datetime(str(month), format='%m').strftime('%b') for month in months_list)
                    else:
                        mmm_to_avoid = "None"
                except:
                    mmm_to_avoid = "Error parsing months" # Handle potential errors

                # Get description and image
                location_id = city_details_row['id']
                description = next((desc for loc_id, _, _, desc, _ in F1_LOCATION_DESCRIPTIONS if loc_id == location_id), "No description available.")
                image_path = next((img for loc_id, _, _, _, img in F1_LOCATION_DESCRIPTIONS if loc_id == location_id), None)

                # Display details in two sub-columns
                sub_col1, sub_col2 = st.columns(2,)
                with sub_col1:
                    st.image(image_path if image_path else "https://placehold.co/300x200/2a2a2a/444444?text=No+Image",
                             caption=f"{city_details_row['circuit_x']} Circuit Area" if image_path else "Image not available", 
                             use_container_width=False,width=800) # Adjust width for better fit
                with sub_col2:
                    st.markdown(f"""
                        **Circuit:** {city_details_row['circuit_x']}
                        ({city_details_row['country_x']})
                        
                        **Continent:** {city_details_row['continent']}
                        
                        **Lat/Lon:** {city_details_row['latitude']:.3f}, {city_details_row['longitude']:.3f}
                        
                        **🗓️ Months to Avoid:** {mmm_to_avoid}
                        
                        **Notes:** {city_details_row.get('notes', 'N/A')}
                        """) # Use get for notes
                    st.markdown(f"**📝 Description:** {description}") # Description below image/details
            else:
                 st.write("Select a city to see details.")
    else:
        st.info("ℹ️ Select a season with circuits to view city details.")

    st.divider() # Add divider

    # --- Other Data Tables ---
    st.subheader("🗓️ Race Calendar Data")
    if not calendar_df.empty:
        with st.expander("📂 Show Race Calendar Data"):
            st.dataframe(calendar_df, use_container_width=False)
    else:
        st.warning("⚠️ Could not load calendar data.")

    st.subheader("🚚 Travel Logistics Data")
    if not logistics_df.empty:
        with st.expander("📂 Show Travel Logistics Data"):
            st.dataframe(logistics_df, use_container_width=False)
    else:
        st.warning("⚠️ Could not load logistics data.")

    st.subheader("⚙️ Training Regression Calendar Data")
    if not regression_df.empty:
        with st.expander("📂 Show Regression Training Data"):
            st.dataframe(regression_df, use_container_width=False)
    else:
        st.warning("⚠️ Could not load regression calendar data.")


# --- Page 3: Regression Analysis ---
with page3:
    st.header("📈 Regression Analysis (WIP)")
    st.markdown("""
    Analyze estimated emissions or other metrics based on travel distances and factors.
    *(Note: This section relies on ongoing regression model development.)*
    """)

    calendar_df = get_table("fone_calendar")
    logistics_df = get_table("travel_logistic")

    if not calendar_df.empty and not logistics_df.empty:
        merged_df = calendar_df.merge(logistics_df, left_on="outbound_route", right_on="id", how="inner")
        seasons = sorted(merged_df['year'].unique(), reverse=True)
        selected_season_reg = st.selectbox("🗓️ Select Season:", options=seasons, key="season_selector_reg") # Unique key
        season_data = merged_df[merged_df['year'] == selected_season_reg]

        if "from_circuit" in season_data.columns and "to_circuit" in season_data.columns:
            top_air_distances = season_data.nlargest(5, "distance_km")[["codes", "from_circuit", "to_circuit", "distance_km"]].sort_values(by="distance_km", ascending=False)
            top_truck_distances = season_data.nlargest(5, "truck_distance_km")[["codes", "from_circuit", "to_circuit", "truck_distance_km"]].sort_values(by="truck_distance_km", ascending=False)
        else:
            top_air_distances = pd.DataFrame()
            top_truck_distances = pd.DataFrame()

        st.subheader(f"✈️ Top Air vs. 🚚 Truck Distances ({selected_season_reg})")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write("**Top 5 Air Distances (km)**")
            if not top_air_distances.empty:
                st.dataframe(top_air_distances.set_index("codes"), use_container_width=False) # Show table for detail
                # Optional: Bar chart
                # st.bar_chart(top_air_distances.set_index("codes")["distance_km"])
            else:
                st.info("ℹ️ No air distance data available.")
        with col2:
            st.write("**Top 5 Truck Distances (km)**")
            if not top_truck_distances.empty:
                 st.dataframe(top_truck_distances.set_index("codes"), use_container_width=False) # Show table for detail
                 # Optional: Bar chart
                 # st.bar_chart(top_truck_distances.set_index("codes")["truck_distance_km"])
            else:
                st.info("ℹ️ No truck distance data available.")
    else:
        st.warning("⚠️ Could not load calendar or logistics data for analysis.")

    st.divider()
    st.subheader("🔬 Regression Model Insights (Placeholder)")
    st.info("🚧 Regression model integration is currently work in progress.")
    # st.image("https://placehold.co/800x300/2a2a2a/444444?text=Regression+Results+Placeholder", caption="Placeholder: Regression model results and insights")


# --- Page 4: Clustering Analysis ---
with page4:
    st.header("🧩 Clustering Analysis")
    st.markdown("""
    Perform K-Means clustering on selected circuits to identify geographical groups.
    This involves building a circuit list, normalizing coordinates, finding the optimal 'K' (number of clusters),
    and finally performing the clusterization.
    """)

    # --- Initialize Session State for Clustering Steps ---
    cluster_step_keys = {
        "normalize_step_done": False, "normalized_coords": None, "scale_coords_verbose": "",
        "evaluate_step_done": False, "optimal_k": None, "chosen_k_verbose": "", "elbow_plot_fig": None,
        "clusterization_step_done": False, "clustered_data": None, "clusterization_verbose": "", "cluster_plot_fig": None,
        "builder_circuit_list_ids": [], "builder_geo_df_to_cluster": pd.DataFrame()
    }
    for key, default in cluster_step_keys.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # --- Calendar Circuit Builder ---
    st.subheader("🛠️ Calendar Circuit Builder")
    st.write("Select or build the list of circuits to be clustered.")

    calendar_df_clust = get_table("fone_calendar") # Use different var names if needed
    geo_df_clust = get_table("fone_geography")

    if not calendar_df_clust.empty and not geo_df_clust.empty:
        col_build1, col_build2 = st.columns([1, 2]) # Adjust layout

        with col_build1:
            st.markdown("**1. Choose Starting Point**")
            start_type = st.radio(
                "Start with:",
                ["Past Calendar", "Random Selection", "Empty List"],
                key="clust_start_type", # Unique key
                horizontal=False,
            )

            starting_list_ids = [] # List to hold IDs
            if start_type == "Past Calendar":
                seasons = sorted(calendar_df_clust['year'].unique(), reverse=True)
                selected_season_clust = st.selectbox("Select Season:", options=seasons, key="clust_season_select")
                # Fetch IDs directly
                starting_list_df_temp, _ = execute_and_capture(get_historical_cities, year=int(selected_season_clust), info=True, verbose=False) # Don't need verbose here
                if starting_list_df_temp is not None:
                     starting_list_ids = starting_list_df_temp['geo_id'].tolist()
            elif start_type == "Random Selection":
                size = st.number_input("Number of circuits:", min_value=15, max_value=len(geo_df_clust), value=15, step=1, key="clust_random_size")
                seed_clust = st.number_input("Random seed:", value=SEED, step=1, key="clust_random_seed")
                starting_list_df_temp, _ = execute_and_capture(get_random_sample, n=size, info=True, seed=seed_clust, verbose=False)
                if starting_list_df_temp is not None:
                     starting_list_ids = starting_list_df_temp['geo_id'].tolist()
            else: # Empty List
                starting_list_ids = []

        with col_build2:
            st.markdown("**2. Edit Selection**")
            circuit_options_clust = geo_df_clust.set_index('id')['circuit_x'].to_dict()
            selected_ids_clust = st.multiselect(
                "Selected Circuits (IDs):",
                options=list(circuit_options_clust.keys()),
                format_func=lambda x: f"{x}: {circuit_options_clust.get(x, 'Unknown')}",
                default=starting_list_ids, # Use the list of IDs determined above
                key="clust_multiselect"
            )

        # --- Finalize Builder ---
        finalize_builder_clicked = st.button("Confirm Circuit List for Clustering", key="finalize_builder_button", disabled=(len(selected_ids_clust) < 15))

        if finalize_builder_clicked:
             if len(selected_ids_clust) >= 15:
                 st.session_state.builder_circuit_list_ids = selected_ids_clust
                 # Prepare the specific DataFrame needed for clustering steps
                 geo_df_to_cluster_temp = geo_df_clust[geo_df_clust['id'].isin(selected_ids_clust)][['id', 'city_x', 'latitude', 'longitude']].copy()
                 geo_df_to_cluster_temp = geo_df_to_cluster_temp.rename(columns={'city_x': 'city', 'id': 'geo_id'})
                 st.session_state.builder_geo_df_to_cluster = geo_df_to_cluster_temp
                 st.success(f"✅ Circuit list confirmed with {len(selected_ids_clust)} circuits.")
                 # Reset clustering steps when list is confirmed
                 st.session_state.normalize_step_done = False
                 st.session_state.evaluate_step_done = False
                 st.session_state.clusterization_step_done = False
             else:
                 st.error("❌ Please select at least 15 circuits.")

        # Display the currently confirmed list
        if not st.session_state.builder_geo_df_to_cluster.empty:
             st.write("**Current List for Clustering:**")
             st.dataframe(st.session_state.builder_geo_df_to_cluster[['geo_id', 'city']].reset_index(drop=True), use_container_width=False)
        else:
             st.info("ℹ️ Confirm a circuit list (min 15) to proceed.")

    else:
        st.warning("⚠️ Could not load calendar or geography data for the builder.")

    st.divider()

    # --- Clustering Steps ---
    st.subheader("🔄 Clustering Workflow")

    # Get the DataFrame prepared by the builder
    geo_df_to_cluster = st.session_state.builder_geo_df_to_cluster

    # --- 1. Normalization ---
    st.markdown("**Step 1: Normalize Coordinates**")
    normalize_clicked = st.button("Normalize", key="normalize_button", disabled=geo_df_to_cluster.empty)

    if normalize_clicked and not geo_df_to_cluster.empty:
        st.write("⏳ Normalizing coordinates...")
        coords, scale_coords_verbose_output = execute_and_capture(scale_coords, geo_df_to_cluster[['latitude', 'longitude']], verbose=True) # Pass only relevant columns

        if coords is not None:
            st.session_state.normalized_coords = coords
            st.session_state.scale_coords_verbose = scale_coords_verbose_output
            st.session_state.normalize_step_done = True
            # Reset subsequent steps
            st.session_state.evaluate_step_done = False
            st.session_state.clusterization_step_done = False
            st.success("✅ Coordinates normalized successfully.")
        else:
             st.error("❌ Normalization failed.")
             st.session_state.normalize_step_done = False
             st.session_state.scale_coords_verbose = scale_coords_verbose_output # Show error output


    if st.session_state.normalize_step_done:
        st.success("✅ Normalization Complete.")
        with st.expander("💻 Normalize Console Output", expanded=st.session_state.normalize_step_done):
            st.code(st.session_state.scale_coords_verbose, language='text')
    elif not geo_df_to_cluster.empty:
        st.info("ℹ️ Click 'Normalize' to process coordinates for the confirmed list.")

    # --- 2. Evaluation (Elbow Method) ---
    st.markdown("**Step 2: Evaluate Number of Clusters (Elbow Method)**")
    evaluate_clicked = st.button("Find Optimal K", key="evaluate_clusters_button", disabled=not st.session_state.normalize_step_done)

    if evaluate_clicked:
        if st.session_state.normalize_step_done and st.session_state.normalized_coords is not None:
            st.write("⏳ Evaluating Optimal K...")
            # Assuming kmeans_plot_elbow returns (optimal_k, fig, verbose_output)
            eval_results, chosen_k_verbose_output = execute_and_capture(
                kmeans_plot_elbow,
                st.session_state.normalized_coords,
                max_clusters=min(len(geo_df_to_cluster), 15), # Limit max clusters reasonable amount
                random_state=SEED,
                verbose=True,
                img_verbose=True
            )

            if eval_results:
                optimal_k_result, elbow_fig = eval_results # Unpack
                st.session_state.optimal_k = optimal_k_result
                st.session_state.elbow_plot_fig = elbow_fig
                st.session_state.chosen_k_verbose = chosen_k_verbose_output
                st.session_state.evaluate_step_done = True
                # Reset clusterization step
                st.session_state.clusterization_step_done = False
                st.success(f"✅ Optimal number of clusters determined: **{st.session_state.optimal_k}**.")
            else:
                 st.error("❌ Failed to evaluate optimal K.")
                 st.session_state.evaluate_step_done = False
                 st.session_state.chosen_k_verbose = chosen_k_verbose_output # Show error output
        else:
            st.error("❌ Normalization must be completed first.")


    if st.session_state.evaluate_step_done:
        st.success(f"✅ Evaluation Complete. Optimal K = **{st.session_state.optimal_k}**")
        if st.session_state.elbow_plot_fig:
             col_elbow1, col_elbow2, col_elbow3 = st.columns([1, 2, 1]) # Center the plot
             with col_elbow2:
                 st.pyplot(st.session_state.elbow_plot_fig, use_container_width=False)
        with st.expander("💻 Evaluate Console Output", expanded=st.session_state.evaluate_step_done):
            st.code(st.session_state.chosen_k_verbose, language='text')
    elif st.session_state.normalize_step_done:
        st.info("ℹ️ Click 'Find Optimal K' to run the elbow method.")

    # --- 3. Clusterization ---
    st.markdown("**Step 3: Perform Clusterization**")
    clusterize_clicked = st.button("Cluster Circuits", key="perform_clusterization_button", disabled=not st.session_state.evaluate_step_done)

    if clusterize_clicked:
        if st.session_state.evaluate_step_done and st.session_state.optimal_k is not None and st.session_state.normalized_coords is not None:
            st.write("⏳ Performing clusterization...")
            # Assuming clusterize_circuits returns (df_with_clusters, fig, verbose_output)
            cluster_results, clusterization_verbose_output = execute_and_capture(
                clusterize_circuits,
                df=geo_df_to_cluster.copy(), # Pass a copy to avoid modifying state df
                verbose=True,
                opt_k_img_verbose=False, # Already done
                fig_verbose=True # Get the cluster plot
            )

            if cluster_results:
                clustered_data_result, cluster_plot_fig = cluster_results # Unpack
                st.session_state.clustered_data = clustered_data_result
                st.session_state.cluster_plot_fig = cluster_plot_fig
                st.session_state.clusterization_verbose = clusterization_verbose_output
                st.session_state.clusterization_step_done = True
                st.success(f"✅ Clusterization completed successfully into **{st.session_state.optimal_k}** clusters.")
            else:
                 st.error("❌ Clusterization failed.")
                 st.session_state.clusterization_step_done = True
                 st.session_state.clusterization_verbose = clusterization_verbose_output # Show error output
        else:
            st.error("❌ Evaluation step must be completed first.")


    if st.session_state.clusterization_step_done:
        st.success("✅ Clusterization Complete.")
        col1, col2, col3 = st.columns([1, 2, 1]) # Center the plot
        with col2:
            if st.session_state.cluster_plot_fig:
                st.plotly_chart(st.session_state.cluster_plot_fig, use_container_width=True) # Display cluster plot
        with col1:
            if st.session_state.clustered_data is not None:
                st.markdown("**Clustered Data:**")
                st.dataframe(st.session_state.clustered_data, use_container_width=False, height=400) # Show a sample of the clustered data
        with st.expander("💻 Clusterization Console Output", expanded=st.session_state.clusterization_step_done):
            st.code(st.session_state.clusterization_verbose, language='text')
    elif st.session_state.evaluate_step_done:
        st.info("ℹ️ Click 'Cluster Circuits' to perform the final clustering.")


# --- Page 5: GA Optimization ---
with page5:
    st.header("⚙️ Genetic Algorithm Optimization")
    st.markdown("""
    Configure and run the Genetic Algorithm step-by-step using the selected circuits and parameters.
    View the results, including the proposed optimized calendar sequence and its fitness value.
    """)

    # --- Initialize Session State for GA Steps ---
    ga_step_keys = {
        "scenario_prepared": False, "circuits_df_scenario": None, "prepare_scenario_verbose": "",
        "params_set": False, "ga_params": None, "set_params_verbose": "",
        "toolbox_setup": False, "toolbox": None, "stats": None, "hof": None, "deap_toolbox_verbose": "",
        "ga_run_complete": False, "final_population": None, "logbook": None,
        "best_individual": None, "best_fitness": None, "run_ga_verbose": "", "ga_fitness_plot_fig": None
    }
    for key, default_value in ga_step_keys.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # --- 1. Prepare Scenario ---
    st.markdown("### 1. Prepare Scenario")
    st.write("Select the source for the list of circuits to optimize.")

    col_scen1, col_scen2 = st.columns([1, 2]) # Layout columns

    with col_scen1:
        run_type = st.radio(
            "Select Scenario Source:",
            ["Historical Season", "Random Sample", "Custom List"], # Simplified names
            key="ga_run_type",
            horizontal=False, # Vertical might be better here
        )

        scenario_input_valid = False
        prepare_args = {}

        if run_type == "Historical Season":
            season_year = st.number_input("Season Year (2000-2025):", min_value=2000, max_value=2025, value=2024, step=1, key="ga_season_year")
            prepare_args = {"from_season": int(season_year), "verbose": True}
            scenario_input_valid = True
        elif run_type == "Random Sample":
            sample_size = st.number_input("Sample Size (min 15):", min_value=15, value=20, step=1, key="ga_sample_size")
            prepare_args = {"from_sample": int(sample_size), "verbose": True}
            scenario_input_valid = True
        elif run_type == "Custom List":
            all_geo_df_ga = get_table("fone_geography")
            if not all_geo_df_ga.empty:
                circuit_options_ga = all_geo_df_ga.set_index('id')['circuit_x'].to_dict()
                # Use text area for easier pasting, then parse
                id_list_str_ga = st.text_area("Enter Circuit IDs (comma-separated, min 15):", "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15", key="ga_custom_ids_text", height=100)
                try:
                    selected_ids_ga = [int(x.strip()) for x in id_list_str_ga.split(',') if x.strip()]
                    if len(selected_ids_ga) >= 15:
                        prepare_args = {"from_input": selected_ids_ga, "verbose": True}
                        scenario_input_valid = True
                    else:
                        st.warning("⚠️ Please enter at least 15 circuit IDs.")
                except ValueError:
                    st.error("❌ Invalid input. Please enter comma-separated numbers.")
            else:
                st.error("❌ Could not load geography data for custom list.")

    with col_scen2:
        st.write(" ") # Spacer
        st.write(" ") # Spacer
        prepare_clicked = st.button("✅ Prepare Scenario", key="prepare_scenario_button", disabled=not scenario_input_valid, use_container_width=True)

        if prepare_clicked and scenario_input_valid:
            st.write("⏳ Preparing scenario...")
            circuits_df_pack, verbose_output = execute_and_capture(prepare_scenario, **prepare_args)
            circuits_df, fig = circuits_df_pack
            code_list = circuits_df['circuit_name'].to_list() if circuits_df is not None else [] # Get the list of circuit IDs
            code_names = get_circuits_by('code_6', code_list, 'circuit_x') # Call to ensure the function is executed
            
            if circuits_df is not None and not circuits_df.empty:
                st.session_state.circuits_df_scenario = circuits_df
                st.session_state.code_names = code_names # Store the circuit names
                st.session_state.prepare_scenario_fig = fig # Store the figure
                st.session_state.prepare_scenario_verbose = verbose_output
                st.session_state.scenario_prepared = True
                # Reset subsequent steps
                st.session_state.params_set = False
                st.session_state.toolbox_setup = False
                st.session_state.ga_run_complete = False
                st.success("✅ Scenario prepared successfully.")
            else:
                st.error("❌ Failed to prepare scenario.")
                st.session_state.scenario_prepared = False
                st.session_state.prepare_scenario_verbose = verbose_output
                st.code(st.session_state.prepare_scenario_verbose, language='text')
    if st.session_state.scenario_prepared:
        st.success("✅ Scenario Prepared.")
        st.write("**Circuit List for Optimization:**")
        col1, col2, col3 = st.columns([2,3,13]) # Adjust layout
        with col1:
            st.markdown("**Circuit Codes:**")
            st.write(st.session_state.code_names.get("code_6", "No data available"))
        with col2:
            st.markdown("**Circuit Names:**")
            st.write(st.session_state.code_names.get("circuit_x", "No data available"))
        with col3:
            st.markdown("**Circuit Clusters:**")
            if st.session_state.prepare_scenario_fig:
                st.plotly_chart(st.session_state.prepare_scenario_fig, use_container_width=True)

        with st.expander("💻 Prepare Scenario Console Output", expanded=False): # Keep collapsed by default
            st.code(st.session_state.prepare_scenario_verbose, language='text')
    elif scenario_input_valid:
         st.info("ℹ️ Click 'Prepare Scenario' to load circuit data.")

    st.divider()

    # --- 2. Set Parameters ---
    st.markdown("### 2. Set Genetic Algorithm Parameters")

    col_param1, col_param2 = st.columns(2)

    with col_param1:
        load_defaults_clicked = st.button("⚙️ Load Default Parameters", key="load_defaults_button", use_container_width=True)
        if load_defaults_clicked:
             st.session_state.ga_params = set_default_params({})
             st.session_state.params_set = True
             st.session_state.toolbox_setup = False
             st.session_state.ga_run_complete = False
             st.success("✅ Default parameters loaded.")

    if st.session_state.params_set:
        with col_param2:
             st.success("✅ Parameters Loaded/Set.")

        st.write("**Current Parameters (Editable below):**")
        with st.expander("Show/Edit Parameters", expanded=False): # Keep collapsed by default
            current_params = st.session_state.ga_params.copy()
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                current_params["POPULATION_SIZE"] = st.number_input("Population Size", min_value=10, max_value=1000, value=current_params["POPULATION_SIZE"], step=10, key="pop_size")
                current_params["NUM_GENERATIONS"] = st.number_input("Number of Generations", min_value=5, max_value=1000, value=current_params["NUM_GENERATIONS"], step=5, key="num_gen")
                current_params["CROSSOVER_PROB"] = st.slider("Crossover Probability (cxpb)", min_value=0.0, max_value=1.0, value=current_params["CROSSOVER_PROB"], step=0.05, key="cxpb")
                current_params["REGRESSION"] = st.checkbox("📈 Use Regression for Fitness", value=current_params["REGRESSION"], key="use_regr")

            with col_p2:
                 current_params["MUTATION_PROB"] = st.slider("Mutation Probability (mutpb)", min_value=0.0, max_value=1.0, value=current_params["MUTATION_PROB"], step=0.05, key="mutpb")
                 current_params["TOURNAMENT_SIZE"] = st.number_input("Tournament Size (Selection)", min_value=2, max_value=20, value=current_params["TOURNAMENT_SIZE"], step=1, key="tourn_size")
                 current_params["CLUSTERS"] = st.checkbox("🧩 Use Clusters in Fitness", value=current_params["CLUSTERS"], key="use_clust")

            update_params_clicked = st.button("💾 Update Parameters", key="update_params_button")
            if update_params_clicked:
                st.session_state.ga_params = current_params
                st.session_state.toolbox_setup = False
                st.session_state.ga_run_complete = False
                st.success("✅ Parameters updated.")

    else:
        st.info("ℹ️ Click 'Load Default Parameters' to initialize.")

    st.divider()

    # --- 3. Setup DEAP Toolbox ---
    st.markdown("### 3. Setup DEAP Toolbox")
    setup_toolbox_clicked = st.button(
        "🛠️ Setup DEAP Toolbox",
        key="setup_toolbox_button",
        disabled=not (st.session_state.scenario_prepared and st.session_state.params_set)
    )

    if setup_toolbox_clicked:
        if st.session_state.scenario_prepared and st.session_state.params_set:
            st.write("⏳ Setting up DEAP toolbox...")
            fitness_func = genetic_ops.calculate_fitness
            toolbox_results, verbose_output = execute_and_capture(
                deap_toolbox,
                circuits_df_scenario=st.session_state.circuits_df_scenario,
                fitness_function=fitness_func,
                params=st.session_state.ga_params,
                seed=st.session_state.ga_params.get("RANDOM_SEED", SEED),
                verbose=True
            )

            if toolbox_results:
                toolbox, stats, hof = toolbox_results
                st.session_state.toolbox = toolbox
                st.session_state.stats = stats
                st.session_state.hof = hof
                st.session_state.deap_toolbox_verbose = verbose_output
                st.session_state.toolbox_setup = True
                st.session_state.ga_run_complete = False
                st.success("✅ DEAP Toolbox setup complete.")
            else:
                 st.error("❌ Failed to set up DEAP toolbox.")
                 st.session_state.toolbox_setup = False
                 st.session_state.deap_toolbox_verbose = verbose_output

        else:
            st.error("❌ Scenario must be prepared and parameters set first.")

    if st.session_state.toolbox_setup:
        st.success("✅ DEAP Toolbox Ready.")
        with st.expander("💻 DEAP Toolbox Setup Console Output", expanded=False): # Keep collapsed
            st.code(st.session_state.deap_toolbox_verbose, language='text')
    elif st.session_state.scenario_prepared and st.session_state.params_set:
         st.info("ℹ️ Click 'Setup DEAP Toolbox' to initialize.")

    st.divider()

    # --- 4. Run Genetic Algorithm ---
    st.markdown("### 4. Run Genetic Algorithm")
    run_ga_clicked = st.button(
        "🚀 Run Genetic Algorithm",
        key="run_ga_streamlit_button",
        disabled=not st.session_state.toolbox_setup
    )

    if run_ga_clicked:
        if st.session_state.toolbox_setup:
            st.write("⏳ Running Genetic Algorithm... Please wait.")
            with st.spinner('🧬 GA is evolving... This might take a while!'):
                ga_results, verbose_output = execute_and_capture(
                    run_genetic_algorithm,
                    toolbox=st.session_state.toolbox,
                    stats=st.session_state.stats,
                    hof=st.session_state.hof,
                    params=st.session_state.ga_params,
                    verbose=True
                )

            if ga_results:
                pop, log, best_ind, best_fit = ga_results
                st.session_state.final_population = pop
                st.session_state.logbook = log
                st.session_state.best_individual = best_ind
                st.session_state.best_fitness = best_fit
                st.session_state.run_ga_verbose = verbose_output
                st.session_state.ga_run_complete = True
                st.success("🏁 Genetic Algorithm finished successfully!")

                # --- Generate Plot Figure ---
                try:
                    logbook = st.session_state.logbook
                    gen = logbook.select("gen")
                    min_fitness = logbook.select("min")
                    avg_fitness = logbook.select("avg")
                    fig, ax1 = plt.subplots(figsize=(10, 5))
                    # Use F1 colors
                    line1 = ax1.plot(gen, min_fitness, color='#FF1801', linestyle='-', marker='o', markersize=4, label="Minimum Fitness (Best)")
                    line2 = ax1.plot(gen, avg_fitness, color='#888888', linestyle='--', label="Average Fitness")
                    ax1.set_xlabel("Generation", fontsize=12)
                    ax1.set_ylabel("Fitness Score", color="white", fontsize=12)
                    ax1.tick_params(axis='y', labelcolor="white")
                    ax1.tick_params(axis='x', labelcolor="white")
                    ax1.grid(True, linestyle='--', alpha=0.3)
                    ax1.set_facecolor('#2a2a2a') # Match background
                    fig.patch.set_facecolor('#1E1E1E') # Match body background
                    ax1.spines['top'].set_visible(False)
                    ax1.spines['right'].set_visible(False)
                    ax1.spines['bottom'].set_color('#555555')
                    ax1.spines['left'].set_color('#555555')
                    lns = line1 + line2
                    labs = [l.get_label() for l in lns]
                    ax1.legend(lns, labs, loc="upper right", facecolor='#333333', edgecolor='#555555', labelcolor='white')
                    plt.title("Fitness over Generations", color='white', fontsize=14, fontweight='bold')
                    st.session_state.ga_fitness_plot_fig = fig # Store the figure
                except Exception as plot_err:
                     st.warning(f"⚠️ Could not generate fitness logbook plot: {plot_err}")
                     st.session_state.ga_fitness_plot_fig = None
                # --- End Plot Generation ---

            else:
                st.error("❌ Genetic Algorithm execution failed.")
                st.session_state.ga_run_complete = False
                st.session_state.run_ga_verbose = verbose_output

        else:
            st.error("❌ DEAP Toolbox must be set up before running the algorithm.")

    # --- Display Final GA Results ---
    st.divider()
    st.subheader("🏆 Genetic Algorithm Final Results")

    if st.session_state.ga_run_complete:
        st.success("🏁 Optimization Complete!")

        col_res1, col_res2 = st.columns([1, 2]) # Layout for results

        with col_res1:
            if st.session_state.best_fitness is not None:
                 fitness_type = "Est. Emissions (Units)" if st.session_state.ga_params.get("REGRESSION", False) else "Distance (km)"
                 st.metric(label=f"🏆 Best Fitness ({fitness_type})", value=f"{st.session_state.best_fitness:,.2f}")

            if st.session_state.best_individual is not None:
                st.write("**📅 Optimized Sequence:**")
                scenario_df = st.session_state.circuits_df_scenario
                # Attempt to map codes back to full names
                name_map = {}
                if scenario_df is not None and 'circuit_name' in scenario_df.columns and 'circuit' in scenario_df.columns:
                     # Assuming 'circuit_name' holds the codes/IDs and 'circuit' holds the full name
                     name_map = scenario_df.set_index('circuit_name')['circuit'].to_dict()

                sequence_df = pd.DataFrame({
                    'Order': range(1, len(st.session_state.best_individual) + 1),
                    'Circuit Code': st.session_state.best_individual
                })
                # Add mapped names if available
                if name_map:
                     sequence_df['Circuit Name'] = sequence_df['Circuit Code'].map(name_map).fillna("N/A")
                     st.dataframe(sequence_df[['Order', 'Circuit Name', 'Circuit Code']], use_container_width=True, height=400) # Limit height
                else:
                     st.dataframe(sequence_df, use_container_width=True, height=400) # Limit height
            else:
                 st.warning("⚠️ Best sequence data not available.")

        with col_res2:
             st.write("**📉 Fitness Evolution:**")
             if st.session_state.ga_fitness_plot_fig is not None:
                  st.pyplot(st.session_state.ga_fitness_plot_fig)
             else:
                  st.info("ℹ️ Fitness plot not available.")

        # Display console output collapsed by default
        with st.expander("💻 Full GA Run Console Output", expanded=False):
            st.code(st.session_state.run_ga_verbose, language='text')


    elif st.session_state.toolbox_setup:
         st.info("ℹ️ Click 'Run Genetic Algorithm' to start the optimization.")


# --- Footer ---
st.divider()
st.markdown("---")
st.caption("F1 Green Flag | Developed by Jakob Spranger, Juan Jose Montesinos, Maximilian von Braun, Massimiliano Napolitano")
st.caption("Find the project on [GitHub](https://github.com/Ulquiorra-23/planetf1)")
