# Standard library imports
import sqlite3

# Third-party library imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from kneed import KneeLocator
import plotly.express as px

def get_historical_cities(year: int, verbose: bool = False, info: bool = False) -> pd.DataFrame:
    """
    Retrieves the DataFrame of cities for the given year from dfs_by_year_dict.

    Args:
        year (int): The year for which to retrieve the city data.
        verbose (bool): If True, print debug information. Default is False.
        info (bool): If True, include the 'geo_id' and the 'circuit' in the DataFrame. Default is False.

    Returns:
        pandas.DataFrame: DataFrame containing city, latitude, and longitude for the given year.
    """
    if verbose:
        print("Connecting to the database...")
    # Connect to the database
    connection = sqlite3.connect('planet_fone.db')

    # Create a cursor object to execute SQL queries
    cursor = connection.cursor()

    # Execute the query to fetch city_x, latitude, and longitude from fone_geography
    cursor.execute("SELECT  fc.geo_id, fg.code_6 ,fg.circuit_x, fg.city_x, fg.country_x, fg.latitude, fg.longitude \
                    ,fg.first_gp_probability, fg.last_gp_probability\
                    FROM fone_calendar fc \
                    LEFT JOIN fone_geography fg ON fc.geo_id = fg.id \
                    WHERE fc.year = ?", (year,))

    # Fetch all results
    city_data = cursor.fetchall()

    # Close the connection
    connection.close()
    if verbose:
        print("Database connection closed.")
    
    # Convert the list of city data to a DataFrame
    if info:
        city_data_df = pd.DataFrame(city_data, columns=['geo_id', 'code', 'circuit', 'city', 'country', 'latitude', 'longitude','first_gp_probability','last_gp_probability'])
    else:
        # Extract only the city_x, latitude, and longitude columns
        city_data_df = pd.DataFrame(city_data, columns=['geo_id', 'code', 'circuit', 'city', 'country', 'latitude', 'longitude','first_gp_probability','last_gp_probability'])[['city', 'latitude', 'longitude']]

    # Remove duplicates and print what is being removed
    duplicates = city_data_df[city_data_df.duplicated()]
    if verbose and not duplicates.empty:
        print("Removing duplicates:")
        print(duplicates)

    city_data_df = city_data_df.drop_duplicates()

    # Print the DataFrame
    if verbose:
        print("Extracted city data from year", year, "with info:")
        print(city_data_df.info())
        
    return city_data_df

def scale_coords(data: pd.DataFrame, verbose: bool = False):
    """
    Scales the latitude and longitude features of the given DataFrame.

    Args:
        data (pd.DataFrame): DataFrame containing 'latitude' and 'longitude' columns.
        verbose (bool): If True, print debug information. Default is False.

    Returns:
        np.ndarray: Scaled coordinates as a numpy array.
    """
    # --- 1. Select Features for Scaling ---
    feature_cols = ['latitude', 'longitude']
    coords = data[feature_cols].values  # Get as numpy array
    if verbose:
        print(f"\nSelected features: {feature_cols}")
        print("Original Coordinates (first 5 rows):")
        print(coords[:5])

    # --- 2. Scale Features (using StandardScaler) ---
    scaler = StandardScaler()

    # Fit the scaler to the data and transform the data
    scaled_coords = scaler.fit_transform(coords)

    if verbose:
        print("\nScaled Coordinates (first 5 rows):")
        # Print with formatting for better readability
        print(np.round(scaled_coords[:5], 4))
        print("\nFeature scaling complete.")
    
    return scaled_coords

def kmeans_plot_elbow(coord, min_clusters=3, max_clusters=10, random_state=23, verbose=False, img_verbose=False):
    """
    Plots the Elbow Method to determine the optimal number of clusters for K-Means.

    Args:
        coord (np.ndarray): Scaled coordinates as a numpy array (e.g., latitude and longitude).
        min_clusters (int): Minimum number of clusters to evaluate. Default is 5.
        max_clusters (int): Maximum number of clusters to evaluate. Default is 10.
        random_state (int): Random state for reproducibility. Default is 23.
        verbose (bool): If True, print debug information. Default is False.

    Returns:
        int: optimal_k, the elbow point determined by kneed.
    """
    if not isinstance(coord, np.ndarray):
        raise ValueError("Input 'coord' must be a numpy array.")
    if coord.shape[1] != 2:
        raise ValueError("Input 'coord' must have exactly 2 columns (latitude and longitude).")
    if max_clusters < min_clusters:
        raise ValueError("max_clusters must be greater than or equal to min_clusters.")

    inertia = []
    k_range = range(min_clusters, max_clusters + 1)

    for k in k_range:
        kmeans_elbow = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        kmeans_elbow.fit(coord)
        inertia.append(kmeans_elbow.inertia_)

    if verbose:
        print("\nInertia values for each k:")
        print(inertia)

    if img_verbose:
        plt.figure(figsize=(12, 8))

        # Plot inertia
        plt.plot(k_range, inertia, marker='o', linestyle='-', color='b', label='Inertia')

        # Calculate and plot the absolute change in inertia
        abs_change_inertia = [inertia[i - 1] - inertia[i] for i in range(1, len(inertia))]
        plt.plot(k_range[1:], abs_change_inertia, marker='o', linestyle='--', color='g', label='Absolute Change in Inertia')

    # --- Using kneed library to find elbow ---
    if verbose:
        print("\n--- Using kneed library to find elbow ---")
    try:
        # Instantiate KneeLocator
        kneedle = KneeLocator(
            list(k_range),         # X values (number of clusters)
            inertia,               # Y values (corresponding inertia/WCSS)
            curve='convex',       # Shape of the inertia curve (it drops and flattens)
            direction='decreasing', # Trend of the inertia values (they go down)
            S=1,            # Sensitivity parameter (0.1 is a good default)
        )

        # Access the detected elbow point (k value)
        optimal_k_kneed = kneedle.elbow

        if optimal_k_kneed:
            if verbose:
                print(f"Optimal k found by kneed: {optimal_k_kneed}")
            chosen_k = optimal_k_kneed
            if img_verbose:
                # Add kneed's result to the plot
                plt.vlines(optimal_k_kneed, plt.ylim()[0], plt.ylim()[1], 
                           linestyles='--', color='r', label=f'Elbow (k={optimal_k_kneed})')
        else:
            if verbose:
                print("kneed could not find a distinct elbow point.")
            chosen_k = min_clusters  # Example fallback
            if verbose:
                print(f"Falling back to k={chosen_k}")

    except ImportError:
        print("Error: 'kneed' library not found. Please install it using: pip install kneed")
        chosen_k = None
    except Exception as e:
        print(f"An error occurred using kneed: {e}")
        chosen_k = None

    if img_verbose:
        # Highlight the area ±1 around the chosen_k
        if chosen_k:
            plt.axvspan(max(chosen_k - 1, min_clusters), min(chosen_k + 1, max_clusters), color='yellow', alpha=0.3, label='±1 Around Chosen k')

        # Add labels, title, and legend
        plt.xlabel('Number of Clusters (k)', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.title('Elbow Method with Absolute Change in Inertia', fontsize=14)
        plt.grid(True)
        plt.legend()
        plt.show()

    return chosen_k

def clusterize_circuits(year=None, df=None, verbose=False, opt_k_img_verbose=False, fig_verbose=False):
    """
    Clusterize circuits based on their geographical coordinates.

    Args:
        year (int, optional): The year to fetch historical cities data. Required if df is not provided.
        df (pd.DataFrame, optional): DataFrame containing 'city', 'latitude', and 'longitude'. Required if year is not provided.
        verbose (bool, optional): Whether to print debug information. Default is False.
        img_verbose (bool, optional): Whether to display the elbow plot. Default is False.
        fig_verbose (bool, optional): Whether to display the final cluster visualization. Default is False.

    Returns:
        pd.DataFrame: DataFrame with an additional 'cluster_id' column.
    """
    if verbose:
        print("Starting clusterization process...")

    if df is None and year is None:
        raise ValueError("Either 'year' or 'df' must be provided.")
    
    # Fetch data if year is provided
    if year is not None:
        if verbose:
            print(f"Fetching historical cities data for year {year}...")
        df = get_historical_cities(year, verbose=verbose)
    
    # Ensure the DataFrame has the required columns
    required_columns = {'city', 'latitude', 'longitude'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")
    
    # Scale the coordinates
    if verbose:
        print("Scaling coordinates...")
    coords = scale_coords(df, verbose=verbose)
    
    # Determine the optimal number of clusters using the elbow method
    max_clusters = len(df)
    if verbose:
        print(f"Determining optimal number of clusters (max_clusters={max_clusters})...")
    optimal_k = kmeans_plot_elbow(coords, max_clusters=max_clusters, random_state=23, verbose=verbose, img_verbose=opt_k_img_verbose)
    
    # Perform K-Means clustering
    if verbose:
        print(f"Running K-Means with k={optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=23)
    kmeans.fit(coords)
    cluster_labels = kmeans.labels_
    
    # Add cluster IDs to the DataFrame
    if verbose:
        print("Assigning cluster IDs to circuits...")
    clustered_df = df.copy()
    clustered_df['cluster_id'] = cluster_labels
    clustered_df['cluster_id'] = clustered_df['cluster_id'].astype(str)
    if verbose:
        print("Cluster IDs assigned successfully.")
    
    # Generate visualization if fig_verbose is True
    if fig_verbose:
        if verbose:
            print("\n--- Generating Enhanced Plotly Map ---")
        # Group cities by cluster_id for legend
        cluster_cities = clustered_df.groupby('cluster_id')['city'].apply(lambda x: ', '.join(x)).to_dict()
        fig = px.scatter_geo(
            data_frame=clustered_df,
            lat='latitude',
            lon='longitude',
            color='cluster_id',
            hover_name='city',
            projection='natural earth',
            title=f'F1 Circuit Clusters (k={optimal_k})',
            color_discrete_sequence=px.colors.qualitative.Set1,
            size_max=20
        )
        fig.update_traces(
            marker=dict(size=10)
        )
        fig.update_geos(
            visible=True, resolution=50,
            showcountries=True, countrycolor="Black",
            showsubunits=True, subunitcolor="Blue"
        )
        # Update legend to include cities in each cluster with text wrapping
        fig.update_layout(
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
            title_font_size=14,  # Scale down the title size
            legend_title_text='Cluster (Cities)',
            legend=dict(
            itemsizing='constant',
            title_font_size=12,
            font_size=10,
            traceorder='normal'
            )
        )
        for trace in fig.data:
            cluster_id = trace.name
            if cluster_id in cluster_cities:
                # Wrap text to include 3 cities per line in the legend
                cities = cluster_cities[cluster_id].split(', ')
                wrapped_cities = '<br>'.join([', '.join(cities[i:i+3]) for i in range(0, len(cities), 3)])
                trace.name = f"Cluster {cluster_id}:<br>{wrapped_cities}"
        fig.show()
        if verbose:
            print("\nEnhanced Plotly figure object 'fig' created.")
    if verbose:
        print("Clusterization process completed.")
    
    return clustered_df

def get_random_sample(n, info: bool, verbose=False, seed=None):
    """
    Fetches a random n-sized sample of city_x, latitude, and longitude from the fone_geography table.

    Args:
        n (int): The number of random rows to fetch.
        info (bool): If True, includes additional columns in the DataFrame.
        verbose (bool): If True, prints debug information.
        seed (int, optional): Seed for randomization to ensure reproducibility.

    Returns:
        pandas.DataFrame: A DataFrame containing the random sample.
    """
    if verbose:
        print(f"Fetching a random sample of {n} rows from the database...")

    # Connect to the database
    connection = sqlite3.connect('planet_fone.db')

    # SQL query to fetch all rows
    query = """
    SELECT fg.id, fg.code_6, fg.circuit_x, fg.city_x, fg.country_x, fg.latitude, fg.longitude,fg.first_gp_probability, fg.last_gp_probability
    FROM fone_geography fg;
    """

    # Load the results into a DataFrame
    full_df = pd.read_sql_query(query, connection)

    # Close the connection
    connection.close()

    # Apply random sampling in pandas
    sample_df = full_df.sample(n=n, random_state=seed)

    if not info:
        # Extract only the city_x, latitude, and longitude columns
        sample_df = sample_df[['city_x', 'latitude', 'longitude']]
        # Rename city_x to city
    sample_df.rename(columns={'city_x': 'city'}, inplace=True)

    if verbose:
        print(f"Random sample of {n} rows fetched successfully.")
    return sample_df
