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
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.sql import get_table
from utils.utilities import get_random_sample, get_historical_cities
from models.clustering import kmeans_plot_elbow, scale_coords, clusterize_circuits
from data.app_data import F1_LOCATION_DESCRIPTIONS

import functools # Needed for deap_toolbox registration if not already imported
from models import genetic_ops # Import your genetic operators module
# Import functions from run_ga.py (or their original modules)
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
    st.error("DEAP library not found. Please install it (`pip install deap`) and restart.")
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

# --- Custom CSS (Optional - for more advanced styling) ---
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@400;600;700&display=swap');

    /* General Page Styling */
    body {
        color: #E0E0E0; /* Light grey text for dark background */
        background-color: #1E1E1E; /* Dark background */
        font-family: 'Titillium Web', sans-serif; /* F1-style font */
    }

    /* Main container adjustments (optional, might need tweaking) */
    .main .block-container {
         padding-top: 2rem; /* Adjust top padding */
         padding-bottom: 2rem;
    }

    /* --- Title and Headers --- */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF; /* White headers */
        font-weight: 700; /* Bolder font weight */
    }

    /* Specifically target Streamlit's title element if needed */
    /* Use browser dev tools to find the exact class if h1 isn't enough */
    h1 {
        color: #FF1801; /* F1 Red for main title */
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5);
        text-transform: uppercase; /* Uppercase for impact */
    }

    /* --- Sidebar Styling --- */
    /* Target the sidebar's inner container */
    [data-testid="stSidebar"] > div:first-child {
        background-color: #2a2a2a; /* Slightly lighter dark for sidebar */
        border-right: 2px solid #FF1801; /* Red accent border */
    }
    /* Sidebar header */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
         color: #FFFFFF; /* White headers in sidebar */
    }
    /* Sidebar text */
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stRadio,
    [data-testid="stSidebar"] .stSelectbox {
         color: #E0E0E0;
    }
    /* Sidebar info box */
    [data-testid="stSidebar"] .stAlert {
        background-color: rgba(255, 24, 1, 0.1); /* Red tint for info box */
        border: 1px solid rgba(255, 24, 1, 0.5);
    }


    /* --- Tabs Styling --- */
    /* Unselected tab */
    .stTabs [data-baseweb="tab"] {
        background-color: #333333; /* Dark grey for inactive tabs */
        color: #E0E0E0;
        border-radius: 5px 5px 0 0; /* Rounded top corners */
        margin: 0 3px;
        padding: 10px 15px;
        font-weight: 600;
        border-bottom: 3px solid transparent; /* Space for selected indicator */
        transition: background-color 0.3s ease, border-color 0.3s ease;
    }
    /* Hover effect for unselected tab */
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #444444;
        color: #FFFFFF;
    }
    /* Selected tab */
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #2a2a2a; /* Match sidebar bg */
        color: #FFFFFF; /* White text for selected */
        font-weight: 700;
        border-bottom: 3px solid #FF1801; /* Red underline indicator */
    }
    /* Tab content panel */
    .stTabs [data-testid="stVerticalBlock"] {
         background-color: #2a2a2a; /* Match selected tab bg for content area */
         border: 1px solid #444444;
         border-top: none; /* Remove top border as tab provides it */
         border-radius: 0 0 5px 5px;
         padding: 1rem;
    }

    /* --- DataFrame Styling --- */
    .stDataFrame {
        border: 1px solid #444444; /* Darker border */
        border-radius: 5px;
        background-color: #2a2a2a; /* Dark background for table */
    }
    /* Header */
    .stDataFrame thead th {
        background-color: #333333;
        color: #FFFFFF;
        font-weight: 600;
    }
    /* Body cells */
     .stDataFrame tbody td {
        color: #E0E0E0;
    }

    /* --- Button Styling --- */
    .stButton button {
        background-color: transparent; /* Transparent background */
        color: #FF1801; /* Red text */
        border: 2px solid #FF1801; /* Red border */
        border-radius: 5px;
        font-weight: 700; /* Bold text */
        padding: 8px 15px;
        transition: background-color 0.3s ease, color 0.3s ease;
        text-transform: uppercase;
    }

    .stButton button:hover {
        background-color: #FF1801; /* Red background on hover */
        color: white; /* White text on hover */
    }
     .stButton button:focus { /* Keep focus style consistent */
        background-color: #FF1801;
        color: white;
        box-shadow: none; /* Remove default focus shadow if desired */
        outline: none;
    }
     .stButton button:disabled {
        background-color: transparent;
        color: #555555;
        border-color: #555555;
        opacity: 0.5;
    }

    /* --- Expander Styling --- */
    .stExpander {
        background-color: #2a2a2a; /* Dark background */
        border: 1px solid #444444; /* Darker border */
        border-radius: 5px;
    }
    .stExpander header { /* Target the header part */
        font-weight: 600;
        color: #FFFFFF; /* White header text */
    }

    /* --- Other Widget Styling (Examples) --- */
    .stRadio [role="radiogroup"],
    .stSelectbox > div {
        /* Add subtle styling if needed */
    }

    /* --- Footer Styling --- */
    footer {
        color: #888888; /* Lighter grey for footer */
        font-size: 12px;
        text-align: center;
    }

    /* --- Dividers --- */
    hr {
        border-top: 1px solid #444444; /* Darker divider */
    }

</style>

""", unsafe_allow_html=True)

def execute_and_capture( func, *args, **kwargs):
    """
    Executes a function with the given arguments and captures its stdout.

    Args:
        func (callable): The function to execute.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        tuple: A tuple containing the function's return value and the captured stdout as a string.
    """
    # Create a StringIO buffer to capture stdout
    string_io = io.StringIO()

    # Redirect stdout to the buffer
    with redirect_stdout(string_io):
        try:
            # Execute the function and capture its return value
            result = func(*args, **kwargs)
        except Exception as e:
            # Capture any exceptions that occur within the block
            captured_string = string_io.getvalue()
            error_message = f"An error occurred during execution: {e}"
            return None, f"{captured_string}\n!!! {error_message} !!!"

    # Get the captured output from the buffer
    captured_string = string_io.getvalue()

    # Update the captured output in the sidebar

    return result, captured_string

# --- Header ---
with st.container():
    st.title("🏎️ F1 Green Flag: Sustainable Calendar Optimization")
    st.caption("Analyzing F1 logistics using Clustering, Regression (WIP), and Genetic Algorithms to propose a greener race calendar.")
    st.divider()

    # --- Navigation Tabs ---
page1,page2,page3,page4,page5 = st.tabs(
        ["🏠 Overview", "📊 Data Explorer", "📈 Regression Analysis", "📈 Clustering Analysis", "⚙️ GA Optimization & Results"]
    )

# --- Main Content Area ---

with page1:
    st.header("Project Overview")
    st.markdown("""
    Welcome to the **F1 Green Flag** dashboard!

    This project analyzes Formula 1 logistics data to develop actionable recommendations for reducing the sport's environmental impact. By leveraging data science techniques, including **clustering**, **regression analysis (WIP)**, and **genetic algorithms**, we aim to optimize the F1 race calendar to minimize travel-related emissions for cars and equipment.

    **Core Components:**
    - **Data Collection & Preparation:** Consolidating circuit, geographical, and logistical data into the `planet_fone.db` database.
    - **Clustering Analysis:** Identifying potential geographical groupings of races using K-Means (`models/clustering.py`).
    - **Regression Modeling (WIP):** Developing models to estimate emissions per travel leg.
    - **Genetic Algorithm Optimization:** Utilizing DEAP (`models/genetic_ops.py`, `run_ga.py`) to find near-optimal race sequences minimizing travel distance or estimated emissions, considering clustering insights.

    Use the sidebar to navigate through the different sections of our analysis and results.
    """)
    st.image("https://placehold.co/800x300/228B22/FFFFFF?text=F1+Track+Map+Placeholder", caption="Placeholder for a relevant F1 image or map")


with page2:
    st.header("Explore the F1 Data")
    st.markdown("""
    Dive into the datasets used for this analysis, primarily sourced from `planet_fone.db` and files within the `/data` directory.
    You can view raw data, summary statistics, and filter information.
    """)

    # --- Data Loading and Display ---
    st.subheader("Circuit Geography Data")
    geo_df = get_table("fone_geography")  # Fetch data using get_table
    calendar_df = get_table("fone_calendar")  # Fetch data using get_table
    logistics_df = get_table("travel_logistic")  # Fetch data using get_table
    regression_df = get_table("training_regression_calendar")  # Fetch data using get_table
    
    if not geo_df.empty:
        st.dataframe(geo_df.head())
        st.write(f"Loaded {len(geo_df)} Locations.")
        with st.expander("Show Full Geography Data"):
            st.dataframe(geo_df)
    else:
        st.warning("Could not load geography data.")
        st.subheader("Map of Circuit Locations")
    if not geo_df.empty:
        # Ensure the DataFrame has latitude and longitude columns
        if 'latitude' in geo_df.columns and 'longitude' in geo_df.columns:
            # Add a selector for season
            seasons = sorted(calendar_df['year'].unique(), reverse=True) if not calendar_df.empty else []
            selected_season = st.selectbox("Select a Season to Highlight Circuits:", options=["New"] + list(seasons), index=0,key="season_selector")
            if selected_season == "New" and not calendar_df.empty:
                # Highlight circuits never visited
                visited_circuits = calendar_df['geo_id'].unique()
                map_data = geo_df[~geo_df['id'].isin(visited_circuits)]
            elif selected_season != "New" and not calendar_df.empty:
                # Get circuits for the selected season
                season_circuits = calendar_df[calendar_df['year'] == selected_season]['geo_id'].unique()
                map_data = geo_df[geo_df['id'].isin(season_circuits)]
            else:
                map_data = pd.DataFrame()  # Empty DataFrame if no valid data

            if not map_data.empty:
                # Create a PyDeck layer for better customization
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=map_data,
                    get_position=["longitude", "latitude"],
                    get_radius=100000,  # Increased marker size (in meters)
                    get_fill_color=[0, 128, 0, 200],  # Green color with transparency
                    pickable=True,
                )

                # Add tooltips for labels
                tooltip = {
                    "html": "<b>City:</b> {city_x}<br><b>Circuit:</b> {circuit_x}<br><b>Country:</b> {country_x}<br><b>Continent:</b> {continent}",
                    "style": {"backgroundColor": "steelblue", "color": "white"},
                }

                # Render the map
                st.pydeck_chart(
                    pdk.Deck(
                        layers=[layer],
                        initial_view_state=pdk.ViewState(
                            latitude=map_data["latitude"].mean(),
                            longitude=map_data["longitude"].mean(),
                            zoom=2,
                            pitch=0,
                        ),
                        tooltip=tooltip,
                    )
                )
            else:
                st.warning("No data available to display on the map.")
                
        else:
            st.warning("Geography data does not contain latitude and longitude columns.")
    else:
        st.warning("Could not load geography data to display the map.")
        
    st.subheader(f"Select a {selected_season} City to View Details")
    if not map_data.empty:
        city_list = map_data['city_x'].unique()
        col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 2, 2])

        if len(city_list) > 12:
            half = len(city_list) // 2
            with col1:
                selected_city = st.selectbox("Select a City:", options=city_list, key="city_selector", index=0 )
        else:
            with col1:
                selected_city = st.radio("Select a City:", options=city_list)

        city_details = map_data[map_data['city_x'] == selected_city]
        city_details['mmm_to_avoid'] = city_details['months_to_avoid'].apply(
            lambda months: ', '.join(
                pd.to_datetime(str(month), format='%m').strftime('%b') for month in months.strip("[]").split(", ") if months.strip("[]") # Handle empty or invalid values
        ))
        
        if not city_details.empty:
            for _, row in city_details.iterrows():
                with col5:
                    st.markdown(f"""
                    **City:** {row['city_x']}  
                    **Circuit:** {row['circuit_x']}  
                    **Country:** {row['country_x']}  
                    **Continent:** {row['continent']}  
                    **Latitude:** {row['latitude']}  
                    **Longitude:** {row['longitude']}  
                    **Months to Avoid:** {row['mmm_to_avoid']}  
                    **Notes:** {row['notes'] if 'notes' in row else 'No additional notes available.'}
                    """)

                with col4:
                    # Retrieve the image path from F1_LOCATION_DESCRIPTIONS
                    image_path = next((img for loc_id, _, _, _, img in F1_LOCATION_DESCRIPTIONS if loc_id == row['id']), None)
                    if image_path:
                        st.image(image_path, caption=f"{row['city_x']} Image")
                    else:
                        st.image("https://placehold.co/150x150/cccccc/FFFFFF?text=No+Image", caption=f"No image available for {row['city_x']}")
                
                with col3:
                    if not map_data.empty and 'id' in map_data.columns:
                        location_id = row['id']
                        description = next((desc for loc_id, _, _, desc, _ in F1_LOCATION_DESCRIPTIONS if loc_id == location_id), "No description available.")
                        st.markdown(f"**Description:** {description}")
        else:
            st.warning(f"No details available for {selected_city}.")
    else:
        st.warning("No geography data available to display city details.")
        
    st.subheader("Race Calendar Data")
    if not calendar_df.empty:
        st.dataframe(calendar_df.head())
        st.write(f"Loaded {len(calendar_df)} calendar entries.")
        with st.expander("Show Full Calendar Data"):
            st.dataframe(calendar_df)
    else:
        st.warning("Could not load calendar data.")

    st.subheader("Travel Logistics Data")
    if not logistics_df.empty:
        st.dataframe(logistics_df.head())
        st.write(f"Loaded {len(logistics_df)} logistics entries.")
        with st.expander("Show Full Logistics Data"):
            st.dataframe(logistics_df)
    else:
        st.warning("Could not load logistics data.")

    st.subheader("Training Regression Calendar Data")
    if not regression_df.empty:
        st.dataframe(regression_df.head())
        st.write(f"Loaded {len(regression_df)} regression calendar entries.")
        with st.expander("Show Full Regression Calendar Data"):
            st.dataframe(regression_df)
    else:
        st.warning("Could not load regression calendar data.")

    # Add more data exploration sections (e.g., for calendar.csv, constraints)


with page3:
    st.header("Regression Analysis (WIP)")
    st.markdown("""
    This section showcases insights derived from regression models. These models aim to estimate emissions
    or other key metrics based on travel distances and other factors.
    """)

    # --- Load Data ---
    calendar_df = get_table("fone_calendar")
    logistics_df = get_table("travel_logistic")

    if not calendar_df.empty and not logistics_df.empty:
        # Merge calendar and logistics data on outbound route (foreign key)
        merged_df = calendar_df.merge(
            logistics_df,
            left_on="outbound_route",
            right_on="id",
            how="inner"
        )

        # Add a selector for the season
        seasons = sorted(merged_df['year'].unique(), reverse=True)
        selected_season = st.selectbox("Select a Season:", options=seasons, key="season_selector_2")

        # Filter data for the selected season
        season_data = merged_df[merged_df['year'] == selected_season]

        # Calculate top 5 distances for the selected season
        if "from_circuit" in season_data.columns and "to_circuit" in season_data.columns:
            top_air_distances = (
                season_data.nlargest(5, "distance_km")[["codes", "from_circuit", "to_circuit", "distance_km"]]
                .sort_values(by="distance_km", ascending=False)
            )

            top_truck_distances = (
                season_data.nlargest(5, "truck_distance_km")[["codes", "from_circuit", "to_circuit", "truck_distance_km"]]
                .sort_values(by="truck_distance_km", ascending=False)
            )
        else:
            top_air_distances = pd.DataFrame()
            top_truck_distances = pd.DataFrame()

        # --- Visualization ---
        st.subheader(f"Top Distances for {selected_season}")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Top 5 Air Distances")
            if not top_air_distances.empty:
                st.bar_chart(
                    top_air_distances.set_index(["codes"])["distance_km"]
                )
            else:
                st.warning("No air distance data available for the selected season.")

        with col2:
            st.subheader("Top 5 Truck Distances")
            if not top_truck_distances.empty:
                st.bar_chart(
                    top_truck_distances.set_index(["codes"])["truck_distance_km"]
                )
            else:
                st.warning("No truck distance data available for the selected season.")
    else:
        st.warning("Could not load calendar or logistics data.")

    # --- Placeholder for Regression Analysis ---
    st.subheader("Regression Model Insights")
    st.image("https://placehold.co/800x300/cccccc/FFFFFF?text=Regression+Results+Placeholder", caption="Placeholder: Regression model results and insights")

with page4:
    st.header("Clustering Analysis")
    st.markdown("""
    This section presents insights from clustering analysis, such as grouping races geographically
    to minimize travel distances and emissions.
    """)

    # --- Calendar Circuit Builder ---
    st.subheader("Calendar Circuit Builder")

    # Load data
    calendar_df = get_table("fone_calendar")
    geo_df = get_table("fone_geography")

    if not calendar_df.empty and not geo_df.empty:
        # Merge calendar and geography data
        merged_df = calendar_df.merge(geo_df, left_on="geo_id", right_on="id", how="inner")
        col1, col2, col3 = st.columns([1,2,2])

        with col1:
            st.markdown("**Build your Calendar**")
            
            # Step 1: Select starting list type
            start_type = st.radio(
                "",
                ["From Past Calendar", "Random Selection", "Empty List"],
                horizontal=False,
            )

            if start_type == "From Past Calendar":
                # Select season
                seasons = sorted(calendar_df['year'].unique(), reverse=True)
                selected_season = st.selectbox("Select a Season:", options=seasons, key="season_selector_3")
                starting_list, captured_output_text = execute_and_capture(get_historical_cities, year=int(selected_season), info=True, verbose=True)
            elif start_type == "Random Selection":
                # Random selection
                size = st.number_input("Enter number of cities to select:", min_value=15, max_value=len(geo_df), value=15, step=1)
                seed = st.number_input("Enter random seed:", value=42, step=1)
                starting_list, captured_output_text = execute_and_capture(get_random_sample, size, info=True,seed=seed, verbose=True)
                starting_list = starting_list.rename(columns={
                    'id': 'geo_id',
                    'code_6': 'code',
                    'circuit_x': 'circuit',
                    'city_x': 'city',
                    'country_x': 'country',
                    'latitude': 'latitude',
                    'longitude': 'longitude',
                    'first_gp_probability': 'first_gp_probability',
                    'last_gp_probability': 'last_gp_probability'
                })
            else:
                # Empty list
                starting_list = pd.DataFrame( columns=['geo_id', 'code', 'circuit',
                                                    'city', 'country', 'latitude', 
                                                    'longitude','first_gp_probability',
                                                    'last_gp_probability'])

            # Step 2: Edit the list
            remaining_cities = geo_df[~geo_df['id'].isin(starting_list['geo_id'])][['city_x', 'id']]
            all_cities = geo_df[['city_x', 'id']]
        

        with col2:
            st.markdown("**Edit Cities**")
            if start_type in ["From Past Calendar", "Random Selection"]:
                cities_to_add = st.multiselect(
                    "Select cities to add:",
                    options=all_cities['city_x'],
                    default=starting_list['city']
                )
            else:
                cities_to_add = st.multiselect("Select cities to add:", options=all_cities['city_x'])

            if st.button("Update Selected Cities", key="update_cities"):
                cities_rows = geo_df[geo_df['id'].isin(cities_to_add)]
                st.success(f"Added {len(cities_to_add)} cities to the calendar.")
                with col3:  
                    # Create and show updated list of IDs after updating selected cities
                    cities_rows = geo_df[geo_df['city_x'].isin(cities_to_add)]
                    updated_ids = cities_rows['id'].tolist()  # Extract updated list of IDs
                    st.write("Updated List of IDs:", updated_ids)
            else:
                updated_ids = starting_list['geo_id'].tolist()

    with st.expander("💻 Console Output"):
        st.code(captured_output_text, language='text')  # Display captured output in the expander
    
    # --- Normalization Section ---
    st.markdown("### 1. Normalize coordinates")
    normalize_clicked = st.button("Normalize", key="normalize_button")

    
    geo_df_to_cluster = geo_df[geo_df['id'].isin(updated_ids)][['city_x', 'latitude', 'longitude']]
    geo_df_to_cluster = geo_df_to_cluster.rename(columns={
        'city_x': 'city',
        'latitude': 'latitude',
        'longitude': 'longitude'
    })
    
    # --- Initialize Session State ---
    if "normalize_step_done" not in st.session_state:
        st.session_state.normalize_step_done = False
        st.session_state.normalized_coords = None
        st.session_state.scale_coords_verbose = ""

    if "evaluate_step_done" not in st.session_state:
        st.session_state.evaluate_step_done = False
        st.session_state.optimal_k = None
        st.session_state.chosen_k_verbose = ""
        st.session_state.elbow_plot_fig = None # Store the figure object

    if "clusterization_step_done" not in st.session_state:
        st.session_state.clusterization_step_done = False
        st.session_state.clustered_data = None
        st.session_state.clusterization_verbose = ""

    if normalize_clicked:
        st.write("Normalizing coordinates...") # Provide immediate feedback
        # Execute the normalization function
        coords, scale_coords_verbose_output = execute_and_capture(
            scale_coords,
            geo_df_to_cluster, # Ensure this DataFrame is available
            verbose=True
        )

        # Store results and status in session state
        st.session_state.normalized_coords = coords
        st.session_state.scale_coords_verbose = scale_coords_verbose_output
        st.session_state.normalize_step_done = True

        # Reset the subsequent step states if normalization is re-run
        st.session_state.evaluate_step_done = False
        st.session_state.optimal_k = None
        st.session_state.chosen_k_verbose = ""
        st.session_state.elbow_plot_fig = None
        st.session_state.clusterization_step_done = False
        st.session_state.clustered_data = None
        st.session_state.clusterization_verbose = ""

        st.success("Coordinates normalized successfully.")
        # st.rerun() # Optional rerun

    # Display normalization results if the step is marked as done
    # Use the session state flag to control expansion persistence
    if st.session_state.normalize_step_done:
        st.success("Normalization Complete.") # Indicate status clearly
        with st.expander("💻 Normalize Console Output", expanded=st.session_state.normalize_step_done): # <-- CHANGE HERE
            st.code(st.session_state.scale_coords_verbose, language='text')
    else:
        st.info("Click 'Normalize' to process coordinates.")


    # --- Evaluation Section ---
    st.markdown("### 2. Evaluate number of clusters")

    # Disable the button if the prerequisite step (normalization) isn't done
    evaluate_clicked = st.button(
        "Evaluate Number of Clusters",
        key="evaluate_clusters_button",
        disabled=not st.session_state.normalize_step_done # Disable if coords aren't ready
    )

    if evaluate_clicked:
        # Ensure we proceed only if normalization is actually done
        if st.session_state.normalize_step_done and st.session_state.normalized_coords is not None:
            st.write("Evaluating Optimal K...") # Provide immediate feedback
            # Execute the evaluation function using stored coordinates
            # Assuming kmeans_plot_elbow returns (optimal_k, fig, verbose_output)
            optimal_k_result, chosen_k_verbose_output = execute_and_capture(
                kmeans_plot_elbow,
                st.session_state.normalized_coords, # Use stored coords
                max_clusters=len(geo_df_to_cluster), # Ensure this is available
                random_state=SEED, # Ensure SEED is available
                verbose=True,
                img_verbose=True # Assuming this controls figure return
            )

            # Store results and status in session state
            st.session_state.optimal_k = optimal_k_result
            st.session_state.elbow_plot_fig = optimal_k_result[1] # Store the figure
            st.session_state.chosen_k_verbose = chosen_k_verbose_output
            st.session_state.evaluate_step_done = True

            # Reset the clusterization step state if evaluation is re-run
            st.session_state.clusterization_step_done = False
            st.session_state.clustered_data = None
            st.session_state.clusterization_verbose = ""

            st.success(f"Optimal number of clusters determined: {st.session_state.optimal_k[0]}.")
            # st.rerun() # Optional rerun
        elif not st.session_state.normalize_step_done:
            st.error("Please run the 'Normalize' step first.")
        else:
            st.error("Normalized coordinates are missing. Please re-run 'Normalize'.")


    # Display evaluation results if the step is marked as done
    # Use the session state flag to control expansion persistence
    if st.session_state.evaluate_step_done:
        st.success(f"Evaluation Complete. Optimal K = {st.session_state.optimal_k[0]}") # Indicate status
        # Display the stored elbow plot
        if st.session_state.elbow_plot_fig:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.pyplot(st.session_state.elbow_plot_fig)
        with st.expander("💻 Evaluate Console Output", expanded=st.session_state.evaluate_step_done): # <-- CHANGE HERE
            st.code(st.session_state.chosen_k_verbose, language='text')
    elif st.session_state.normalize_step_done:
        st.info("Click 'Evaluate Number of Clusters' to determine optimal K.")

    # --- Clusterization Section ---
    st.markdown("### 3. Perform Clusterization")

    # Disable the button if the prerequisite step (evaluation) isn't done
    clusterize_clicked = st.button(
        "Perform Clusterization",
        key="perform_clusterization_button",
        disabled=not st.session_state.evaluate_step_done # Disable if optimal K isn't determined
    )

    if clusterize_clicked:
        # Ensure we proceed only if evaluation is actually done
        if st.session_state.evaluate_step_done and st.session_state.optimal_k is not None and st.session_state.normalized_coords is not None:
            st.write("Performing clusterization...") # Provide immediate feedback
            # Execute the clusterization function using stored coordinates and optimal K
            clustered_data_result, clusterization_verbose_output = execute_and_capture(
                clusterize_circuits,
                df = geo_df_to_cluster,
                verbose=True,
                opt_k_img_verbose=True,
                fig_verbose=True,
            )

            # Store results and status in session state
            st.session_state.clustered_data = clustered_data_result
            st.session_state.clusterization_verbose = clusterization_verbose_output
            st.session_state.clusterization_step_done = True

            st.success(f"Clusterization completed successfully. We have {len(clustered_data_result)} clusters.")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.plotly_chart(clustered_data_result[1])
            st.rerun() # Optional rerun
        elif not st.session_state.evaluate_step_done:
            st.error("Please run the 'Evaluate' step first.")
        else:
            st.error("Required data (Optimal K or Coordinates) is missing. Please re-run previous steps.")


    # Display clusterization results if the step is marked as done
    # Use the session state flag to control expansion persistence
    if st.session_state.clusterization_step_done:
        st.success("Clusterization Complete.") # Indicate status
        # Display clustered data (example)
        if st.session_state.clustered_data is not None:
            st.write("Clustered Data :")
            st.dataframe(st.session_state.clustered_data[0]) # Display head or summary
        with st.expander("💻 Clusterization Console Output", expanded=st.session_state.clusterization_step_done): # <-- CHANGE HERE
            st.code(st.session_state.clusterization_verbose, language='text')
    elif st.session_state.evaluate_step_done:
        st.info("Click 'Perform Clusterization' to clusterize the data.")

# Assuming page5 corresponds to this tab in your st.tabs setup
with page5:
    st.header("Genetic Algorithm Optimization & Results")
    st.markdown("""
    Configure and run the Genetic Algorithm step-by-step.
    View the results, including the proposed optimized calendar sequence and its fitness value.
    """)

    # --- Initialize Session State for GA Steps ---
    default_ga_keys = {
        "scenario_prepared": False, "circuits_df_scenario": None, "prepare_scenario_verbose": "",
        "params_set": False, "ga_params": None, "set_params_verbose": "", # verbose might not be needed here
        "toolbox_setup": False, "toolbox": None, "stats": None, "hof": None, "deap_toolbox_verbose": "",
        "ga_run_complete": False, "final_population": None, "logbook": None,
        "best_individual": None, "best_fitness": None, "run_ga_verbose": ""
    }
    for key, default_value in default_ga_keys.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # --- 1. Prepare Scenario ---
    st.markdown("### 1. Prepare Scenario")
    st.write("Select the source for the list of circuits to optimize.")

    # --- Scenario Configuration UI ---
    run_type = st.radio(
        "Select Scenario Source:",
        ["Optimize Historical Season", "Optimize Random Sample", "Optimize Custom List"],
        key="ga_run_type", # Use a unique key
        horizontal=True,
    )

    scenario_input_valid = False
    prepare_args = {} # Dictionary to hold arguments for prepare_scenario

    if run_type == "Optimize Historical Season":
        season_year = st.number_input("Enter Season Year (2000-2025):", min_value=2000, max_value=2025, value=2024, step=1, key="ga_season_year")
        prepare_args = {"from_season": int(season_year), "verbose": True}
        scenario_input_valid = True
    elif run_type == "Optimize Random Sample":
        sample_size = st.number_input("Enter Sample Size (min 15):", min_value=15, value=20, step=1, key="ga_sample_size")
        prepare_args = {"from_sample": int(sample_size), "verbose": True}
        scenario_input_valid = True
    elif run_type == "Optimize Custom List":
        # Fetch all available circuits for multiselect
        all_geo_df = get_table("fone_geography") # Assuming get_table is efficient or cached
        if not all_geo_df.empty:
            circuit_options = all_geo_df.set_index('id')['circuit_x'].to_dict() # Map ID to Name
            selected_ids = st.multiselect(
                "Select at least 15 Circuit IDs:",
                options=list(circuit_options.keys()),
                format_func=lambda x: f"{x}: {circuit_options.get(x, 'Unknown')}", # Show ID and Name
                key="ga_custom_ids"
            )
            if len(selected_ids) >= 15:
                prepare_args = {"from_input": selected_ids, "verbose": True}
                scenario_input_valid = True
            else:
                st.warning("Please select at least 15 circuits.")
        else:
            st.error("Could not load geography data to populate custom list.")


    prepare_clicked = st.button("Prepare Scenario", key="prepare_scenario_button", disabled=not scenario_input_valid)

    if prepare_clicked and scenario_input_valid:
        st.write("Preparing scenario...")
        # prepare_scenario returns only the DataFrame according to run_ga.py
        circuits_df, verbose_output = execute_and_capture(
            prepare_scenario,
            **prepare_args # Unpack arguments
        )

        if circuits_df is not None and not circuits_df.empty:
            st.session_state.circuits_df_scenario = circuits_df
            st.session_state.prepare_scenario_verbose = verbose_output
            st.session_state.scenario_prepared = True
            # Reset subsequent steps
            st.session_state.params_set = False
            st.session_state.toolbox_setup = False
            st.session_state.ga_run_complete = False
            st.success("Scenario prepared successfully.")
        else:
            st.error("Failed to prepare scenario. Check console output.")
            st.session_state.scenario_prepared = False
            st.session_state.prepare_scenario_verbose = verbose_output # Show error output

        # st.rerun() # Optional

    if st.session_state.scenario_prepared:
        st.success("Scenario Prepared.")
        st.write("Circuit List for Optimization:")
        st.dataframe(st.session_state.circuits_df_scenario[['circuit_name']].reset_index(drop=True)) # Show just names
        with st.expander("💻 Prepare Scenario Console Output", expanded=st.session_state.scenario_prepared):
            st.code(st.session_state.prepare_scenario_verbose, language='text')
    elif scenario_input_valid:
         st.info("Click 'Prepare Scenario' to load circuit data.")


    # --- 2. Set Parameters ---
    st.markdown("### 2. Set Genetic Algorithm Parameters")

    # Button to load defaults
    load_defaults_clicked = st.button("Load Default Parameters", key="load_defaults_button")
    if load_defaults_clicked:
         st.session_state.ga_params = set_default_params({}) # Start with empty dict
         st.session_state.params_set = True
         # Reset subsequent steps
         st.session_state.toolbox_setup = False
         st.session_state.ga_run_complete = False
         st.success("Default parameters loaded.")
         # st.rerun()

    if st.session_state.params_set:
        st.success("Parameters Loaded/Set.")
        st.write("Current Parameters (Editable below):")
        # Display current params (non-editable view)
        st.json(st.session_state.ga_params)

        st.write("Modify Parameters:")
        # --- Parameter Editing UI ---
        # Use columns for better layout
        col_p1, col_p2 = st.columns(2)
        current_params = st.session_state.ga_params.copy() # Work on a copy

        with col_p1:
            current_params["POPULATION_SIZE"] = st.number_input(
                "Population Size", min_value=10, max_value=1000,
                value=current_params["POPULATION_SIZE"], step=10, key="pop_size"
            )
            current_params["NUM_GENERATIONS"] = st.number_input(
                "Number of Generations", min_value=5, max_value=1000,
                value=current_params["NUM_GENERATIONS"], step=5, key="num_gen"
            )
            current_params["CROSSOVER_PROB"] = st.slider(
                "Crossover Probability (cxpb)", min_value=0.0, max_value=1.0,
                value=current_params["CROSSOVER_PROB"], step=0.05, key="cxpb"
            )

        with col_p2:
             current_params["MUTATION_PROB"] = st.slider(
                "Mutation Probability (mutpb)", min_value=0.0, max_value=1.0,
                value=current_params["MUTATION_PROB"], step=0.05, key="mutpb"
            )
             current_params["TOURNAMENT_SIZE"] = st.number_input(
                 "Tournament Size (Selection)", min_value=2, max_value=20,
                 value=current_params["TOURNAMENT_SIZE"], step=1, key="tourn_size"
             )
             # Add toggles for boolean flags if needed
             current_params["REGRESSION"] = st.checkbox("Use Regression for Fitness", value=current_params["REGRESSION"], key="use_regr")
             current_params["CLUSTERS"] = st.checkbox("Use Clusters in Fitness", value=current_params["CLUSTERS"], key="use_clust")


        update_params_clicked = st.button("Update Parameters", key="update_params_button")
        if update_params_clicked:
            st.session_state.ga_params = current_params # Update state with modified values
            # Reset subsequent steps
            st.session_state.toolbox_setup = False
            st.session_state.ga_run_complete = False
            st.success("Parameters updated.")
            # st.rerun() # Rerun to reflect changes immediately if needed

    else:
        st.info("Click 'Load Default Parameters' to initialize.")


    # --- 3. Setup DEAP Toolbox ---
    st.markdown("### 3. Setup DEAP Toolbox")
    setup_toolbox_clicked = st.button(
        "Setup DEAP Toolbox",
        key="setup_toolbox_button",
        disabled=not (st.session_state.scenario_prepared and st.session_state.params_set)
    )

    if setup_toolbox_clicked:
        if st.session_state.scenario_prepared and st.session_state.params_set:
            st.write("Setting up DEAP toolbox...")
            # Ensure the fitness function is available
            fitness_func = genetic_ops.calculate_fitness

            # Call deap_toolbox and capture output
            toolbox_results, verbose_output = execute_and_capture(
                deap_toolbox,
                circuits_df_scenario=st.session_state.circuits_df_scenario,
                fitness_function=fitness_func,
                params=st.session_state.ga_params,
                seed=st.session_state.ga_params.get("RANDOM_SEED", SEED), # Use seed from params or default
                verbose=True # Capture verbose output
            )

            if toolbox_results:
                toolbox, stats, hof = toolbox_results # Unpack results
                st.session_state.toolbox = toolbox
                st.session_state.stats = stats
                st.session_state.hof = hof
                st.session_state.deap_toolbox_verbose = verbose_output
                st.session_state.toolbox_setup = True
                # Reset GA run step
                st.session_state.ga_run_complete = False
                st.success("DEAP Toolbox setup complete.")
            else:
                 st.error("Failed to set up DEAP toolbox.")
                 st.session_state.toolbox_setup = False
                 st.session_state.deap_toolbox_verbose = verbose_output # Show error output

            # st.rerun() # Optional
        else:
            st.error("Scenario must be prepared and parameters set before setting up the toolbox.")

    if st.session_state.toolbox_setup:
        st.success("DEAP Toolbox Ready.")
        with st.expander("💻 DEAP Toolbox Setup Console Output", expanded=st.session_state.toolbox_setup):
            st.code(st.session_state.deap_toolbox_verbose, language='text')
    elif st.session_state.scenario_prepared and st.session_state.params_set:
         st.info("Click 'Setup DEAP Toolbox' to initialize.")


    # --- 4. Run Genetic Algorithm ---
    st.markdown("### 4. Run Genetic Algorithm")
    run_ga_clicked = st.button(
        "Run Genetic Algorithm",
        key="run_ga_streamlit_button",
        disabled=not st.session_state.toolbox_setup
    )

    if run_ga_clicked:
        if st.session_state.toolbox_setup:
            st.write("Running Genetic Algorithm... Please wait.")
            with st.spinner('GA is evolving... This might take a while!'):
                # Call run_genetic_algorithm and capture output
                ga_results, verbose_output = execute_and_capture(
                    run_genetic_algorithm,
                    toolbox=st.session_state.toolbox,
                    stats=st.session_state.stats,
                    hof=st.session_state.hof,
                    params=st.session_state.ga_params,
                    verbose=True # Capture verbose output
                )

            if ga_results:
                pop, log, best_ind, best_fit = ga_results # Unpack
                st.session_state.final_population = pop
                st.session_state.logbook = log
                st.session_state.best_individual = best_ind
                st.session_state.best_fitness = best_fit
                st.session_state.run_ga_verbose = verbose_output
                st.session_state.ga_run_complete = True
                st.success("Genetic Algorithm finished successfully!")
            else:
                st.error("Genetic Algorithm execution failed.")
                st.session_state.ga_run_complete = False
                st.session_state.run_ga_verbose = verbose_output # Show error output

            # st.rerun() # Optional
        else:
            st.error("DEAP Toolbox must be set up before running the algorithm.")

    # --- Display Final GA Results ---
    st.divider()
    st.subheader("Genetic Algorithm Final Results")

    if st.session_state.ga_run_complete:
        st.success("Optimization Complete!")
        # Display captured output
        with st.expander("💻 Full GA Run Console Output (stdout/stderr)", expanded=False):
            st.code(st.session_state.run_ga_verbose, language='text')

        # Display parsed results
        if st.session_state.best_fitness is not None:
             # Determine fitness type based on params
             fitness_type = "Estimated Emissions (Units)" if st.session_state.ga_params.get("REGRESSION", False) else "Distance (km)"
             st.metric(label=f"Best Fitness ({fitness_type})", value=f"{st.session_state.best_fitness:,.2f}") # Format fitness

        if st.session_state.best_individual is not None:
            st.write("**Optimized Sequence (Circuit Codes/IDs):**")
            # Map circuit codes/IDs back to names using the scenario DataFrame
            scenario_df = st.session_state.circuits_df_scenario
            if scenario_df is not None and 'circuit_name' in scenario_df.columns:
                 # Assuming best_individual contains the codes/IDs that are in scenario_df['circuit_name']
                 # Create a mapping if needed, or just display the codes/IDs
                 sequence_df = pd.DataFrame({
                     'Order': range(1, len(st.session_state.best_individual) + 1),
                     'Circuit Code/ID': st.session_state.best_individual
                 })
                 # Optional: Add circuit names if mapping is easy
                 # name_map = scenario_df.set_index('circuit_name')['Actual_Name_Column_If_Available'].to_dict()
                 # sequence_df['Circuit Name'] = sequence_df['Circuit Code/ID'].map(name_map)
                 st.dataframe(sequence_df, use_container_width=True)
            else:
                 st.warning("Could not map sequence to circuit names (scenario data missing). Displaying raw sequence.")
                 st.write(st.session_state.best_individual)


            # Plot Logbook statistics
            if st.session_state.logbook is not None:
                 try:
                     logbook = st.session_state.logbook
                     gen = logbook.select("gen")
                     min_fitness = logbook.select("min")
                     avg_fitness = logbook.select("avg")
                     # max_fitness = logbook.select("max") # Max might not be interesting for minimization

                     fig, ax1 = plt.subplots(figsize=(10, 5))

                     line1 = ax1.plot(gen, min_fitness, "b-", label="Minimum Fitness")
                     line2 = ax1.plot(gen, avg_fitness, "r-", label="Average Fitness")
                     ax1.set_xlabel("Generation")
                     ax1.set_ylabel("Fitness", color="black")
                     ax1.tick_params(axis='y', labelcolor="black")

                     # Add Std Dev if needed
                     # std_fitness = logbook.select("std")
                     # ax2 = ax1.twinx()
                     # line3 = ax2.plot(gen, std_fitness, "g-", label="Standard Deviation")
                     # ax2.set_ylabel("Standard Deviation", color="g")
                     # ax2.tick_params(axis='y', labelcolor="g")

                     lns = line1 + line2 # + line3
                     labs = [l.get_label() for l in lns]
                     ax1.legend(lns, labs, loc="best")

                     plt.title("Fitness over Generations")
                     st.pyplot(fig)
                 except Exception as plot_err:
                     st.warning(f"Could not plot fitness logbook: {plot_err}")


        elif st.session_state.run_ga_verbose: # Show message if run finished but no results parsed
             st.warning("Could not display parsed results. Check script output above.")

    elif st.session_state.toolbox_setup:
         st.info("Click 'Run Genetic Algorithm' to start the optimization.")


# ... (keep existing footer) ...


# --- Footer ---
st.divider()


st.caption("Footer text below the expander")
st.markdown("---")
st.caption("F1 Green Flag | Developed by Jakob Spranger, Juan Jose Montesinos, Maximilian von Braun, Massimiliano Napolitano")
# Link to GitHub repo if available
# st.caption("Find the project on [GitHub](your-repo-link)")
