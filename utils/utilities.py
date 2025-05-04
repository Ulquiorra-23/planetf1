# Standard library imports
import json
import datetime
import sys
import os
from pathlib import Path

# Third-party library imports
import pandas as pd
import numpy as np
import sqlite3

# Local application/library imports
from utils.sql import get_table
from models import clustering


def get_historical_seq(db_path: str, verbose = False):
    """
    Fetches historical sequence data from the database, processes it, and returns a JSON string.
    """
    # Pass db_path to get_table
    historical_races = get_table('fone_calendar', db_path=db_path, verbose=verbose)
    geography = get_table('fone_geography', db_path=db_path, verbose=verbose)
    if verbose:
        print("Fetched historical races and geography data.")

    # Sort the DataFrame by date
    historical_races_sorted = historical_races.sort_values(by='date')
    if verbose:
        print("Sorted historical races by date.")

    # Identify rows where the geo_id and year of consecutive rows are the same
    consecutive_geo_id_indices = historical_races_sorted[
        (historical_races_sorted['geo_id'] == historical_races_sorted['geo_id'].shift()) &
        (historical_races_sorted['year'] == historical_races_sorted['year'].shift())
    ].index

    # Include both consecutive records
    consecutive_geo_id = historical_races_sorted.loc[consecutive_geo_id_indices.union(consecutive_geo_id_indices - 1).sort_values()]
    consecutive_races_same_geo_id = consecutive_geo_id.iloc[0::2]
    historical_races_clean = historical_races.drop(consecutive_races_same_geo_id.index)

    # Ensure the 'date' column is in datetime format
    historical_races_clean['date'] = pd.to_datetime(historical_races_clean['date'])
    merged_data = historical_races_clean.merge(geography, left_on='geo_id', right_on='id', how='left')
    

    # Group the data by year and sort by date within each year
    grouped = merged_data.groupby('year', sort=False)

    # Create the JSON structure
    result = []
    for year, group in grouped:
        group_sorted = group.sort_values(by='date')
        circuit_seq = group_sorted['circuit'].tolist()
        geo_id_seq = group_sorted['geo_id'].tolist()
        length = len(circuit_seq)
        day_month_seq = group_sorted['date'].dt.strftime('%d-%m').tolist()
        code_seq = group_sorted['code_6'].tolist()

        result.append({
            'historical' : True,
            'season': year,
            'length': length,
            'day_month_seq': day_month_seq,
            'circuit_seq': circuit_seq,
            'geo_id_seq': geo_id_seq,
            'code_seq': code_seq
        })

    # Convert the result to JSON
    historical_seq = json.dumps(result, indent=4, ensure_ascii=False)
    if verbose:
        print("Generated historical sequence JSON.")

    return historical_seq

def get_historical_cities(year: int, db_path: str, verbose: bool = False, info: bool = False) -> pd.DataFrame:
    """
    Retrieves the DataFrame of cities for the given year from dfs_by_year_dict.

    Args:
        year (int): The year for which to retrieve the city data.
        verbose (bool): If True, print debug information. Default is False.
        info (bool): If True, include the 'geo_id' and the 'circuit' in the DataFrame. Default is False.

    Returns:
        pandas.DataFrame: DataFrame containing city, latitude, and longitude for the given year.

    Raises:
        ConnectionError: If connection to the database fails.
        ValueError: If no data is found for the specified year.
    """
    connection = None
    try:
        # Add check for file existence
        if not Path(db_path).is_file():
             raise FileNotFoundError(f"Database file not found at specified path: {db_path}")
        if verbose:
            print(f"Connecting to the database at: {db_path}")
        # Use the passed db_path argument here
        connection = sqlite3.connect(db_path)
    # ... (rest of try/except/finally block is mostly unchanged, just uses connection variable) ...
    except sqlite3.Error as e:
        raise ConnectionError(f"Failed to connect to the database at '{db_path}': {e}")
    except FileNotFoundError as e:
         raise e
    try:
        # Create a cursor object to execute SQL queries
        cursor = connection.cursor()

        # Execute the query to fetch city_x, latitude, and longitude from fone_geography
        # Added aliases for clarity
        sql_query = """
            SELECT
                fc.geo_id,
                fg.code_6,
                fg.circuit_x,
                fg.city_x,
                fg.country_x,
                fg.latitude,
                fg.longitude,
                fg.first_gp_probability,
                fg.last_gp_probability
            FROM fone_calendar fc
            LEFT JOIN fone_geography fg ON fc.geo_id = fg.id
            WHERE fc.year = ?
        """
        cursor.execute(sql_query, (year,))

        # Fetch all results
        city_data = cursor.fetchall()

        # --- This is where the ValueError originates ---
        if not city_data:
            # Raise the error if the query returned nothing for that year
            raise ValueError(f"No data found for the year {year} in the database '{DB_PATH}'. Query: {sql_query}")
        # ----------------------------------------------

    except sqlite3.Error as e:
        # Handle potential SQL errors during query execution
        raise RuntimeError(f"Database query failed: {e}")
    finally:
        # Ensure the connection is closed even if errors occur
        if connection:
            connection.close()
            if verbose:
                print("Database connection closed.")

    # Convert the list of city data to a DataFrame
    column_names = ['geo_id', 'code', 'circuit', 'city', 'country', 'latitude', 'longitude','first_gp_probability','last_gp_probability']
    city_data_df = pd.DataFrame(city_data, columns=column_names)

    # --- Logic for 'info' parameter ---
    # Select columns based on the 'info' flag AFTER creating the full DataFrame
    if not info:
         # Keep only specific columns if info is False
         city_data_df = city_data_df[['city', 'latitude', 'longitude']]
    # ------------------------------------

    # Remove duplicates and print what is being removed
    duplicates = city_data_df[city_data_df.duplicated()]
    if verbose and not duplicates.empty:
        print("Removing duplicates:")
        print(duplicates)

    city_data_df = city_data_df.drop_duplicates().reset_index(drop=True) # Reset index after dropping

    # Print the DataFrame info if verbose
    if verbose:
        print(f"Extracted city data from year {year} (info={info}):")
        # Use head() for brevity in verbose output, info() can be long
        print(city_data_df.head())
        print(f"Total rows: {len(city_data_df)}")

    return city_data_df

def generate_f1_calendar(year: int, n: int, verbose: bool = False) -> list[str]:
    assert 2026 <= year <= 2030, "Year must be between 2026 and 2030"
    assert 15 <= n <= 25, "Races must be between 15 and 25"

    if verbose:
        print(f"Generating calendar for year {year} with {n} races.")

    sundays = []
    dt = datetime.date(year, 1, 1)
    while dt.year == year:
        if dt.weekday() == 6:
            sundays.append(dt)
        dt += datetime.timedelta(days=1)

    if verbose:
        print(f"Identified {len(sundays)} Sundays in the year {year}.")

    season_start = max([d for d in sundays if d.month == 2])
    season_end = min([d for d in sundays if d.month == 12])
    sundays = [d for d in sundays if season_start <= d <= season_end]

    if verbose:
        print(f"Season starts on {season_start} and ends on {season_end}.")
        print(f"{len(sundays)} Sundays remain after filtering for season start and end.")

    sundays = [d for d in sundays if not (d.month == 8 and d.day <= 21)]

    if verbose:
        print(f"{len(sundays)} Sundays remain after filtering for August break.")

    race_days = []
    triple_header_count = 0
    i = 0
    p = 1

    if n == 25:
        triple_header = 3
    elif n > 21:
        triple_header = 2
    elif n > 19:
        triple_header = 1
    else:
        triple_header = 0

    if verbose:
        print(f"Triple-header allowance: {triple_header}.")

    while len(race_days) < n and i < len(sundays):
        current = sundays[i]

        if len(race_days) >= 3:
            d1, d2, d3 = race_days[-3], race_days[-2], race_days[-1]
            if (
                (d2 - d1).days == 7 and
                (d3 - d2).days == 7 and
                (current - d3).days == 7
            ):
                if verbose:
                    print(f"Skipping {current} to avoid 4-in-a-row.")
                i += 1
                continue

        if (
            triple_header_count < triple_header and
            len(race_days) + 3 <= n and
            i + 2 < len(sundays) and
            p == 0
        ):
            s1, s2, s3 = sundays[i], sundays[i + 1], sundays[i + 2]
            if (s2 - s1).days == 7 and (s3 - s2).days == 7:
                race_days.extend([s1, s2, s3])
                triple_header_count += 1
                i += 3
                p = triple_header_count + 1 if triple_header == 3 else triple_header_count + 6
                if verbose:
                    print(f"Added triple-header: {s1}, {s2}, {s3}.")
                continue

        if not race_days or (current - race_days[-1]).days >= 14:
            race_days.append(current)
            if i + 1 < len(sundays):
                race_days.append(sundays[i + 1])
                if triple_header_count < 3:
                    p -= 1
                else:
                    p = 1
                i += 2
                if verbose:
                    print(f"Added race days: {race_days[-2]}, {race_days[-1]}.")

        i += 1

    race_days = race_days[:n]
    if verbose:
        print(f"Final race days: {[d.strftime('%d-%m') for d in race_days]}.")

    return [d.strftime("%d-%m") for d in race_days]

def get_random_sample(n: int, db_path: str, info: bool, verbose=False, seed=None):
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
    connection = None
    try:
        # Add check for file existence
        if not Path(db_path).is_file():
             raise FileNotFoundError(f"Database file not found at specified path: {db_path}")
        if verbose:
            print(f"Fetching a random sample of {n} rows from the database at {db_path}...")
        # Use the passed db_path argument here
        connection = sqlite3.connect(db_path)
    # ... (rest of try/except/finally block is mostly unchanged, just uses connection variable) ...
    except sqlite3.Error as e:
        raise ConnectionError(f"Failed to connect to the database at '{db_path}': {e}")
    except FileNotFoundError as e:
         raise e
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

def get_circuit_for_pop(id_val: int, db_path: str, verbose: bool = False):
    """
    Fetches the circuit data for a given ID from the database and returns it as a DataFrame.
    """
    circuits = get_table('fone_geography', db_path=db_path, verbose=verbose)
    if verbose:
        print("Fetched circuit data from the database.")        
    circuit = circuits[circuits['id'] == id_val][['id', 'code_6', 'circuit_x', 'city_x', 'country_x', 'latitude', 'longitude', 'first_gp_probability', 'last_gp_probability']]
    if verbose:
        print(f"Fetched circuit details for ID {id_val}: {circuit}.")
    return circuit

def get_circuits_for_population(db_path: str, n:int =None, seed:int =None, season:int =None, custom:list = None,verbose=False):
    """
    Generate a DataFrame based on the provided seed and n or season.

    Args:
        n (int, optional): Number of circuits to sample.
        seed (int, optional): Seed value for random operations.
        season (int, optional): Season year for filtering or processing.
        verbose (bool, optional): If True, print debug information. Default is False.

    Returns:
        pd.DataFrame: ['geo_id', 'code', 'circuit', 'city', 'country', 'latitude', 'longitude',
       'first_gp_probability', 'last_gp_probability', 'cluster_id'] 
    """
    if sum([season is not None, custom is not None, (n is not None and seed is not None)]) != 1:
        raise ValueError("Exactly one of 'season', 'custom', or both 'n' and 'seed' must be provided.")

    if custom is not None:
        if verbose:
            print(f"Generating circuits for custom population with n = {len(custom)}.")
        prereq_custom = pd.DataFrame()
        for id in custom:
            if verbose:
                print(f"Fetching circuit details for ID {id}...")
            circuit = get_circuit_for_pop(id, db_path=db_path, verbose=verbose)
            prereq_custom = pd.concat([prereq_custom, circuit], ignore_index=True)
        if verbose:
            print(f"Fetched {prereq_custom['circuit_x'].tolist()} circuits for custom population.")
        
        clustersized_circuits = prereq_custom[['city_x', 'latitude', 'longitude']].copy()
        clustersized_circuits.rename(columns={'city_x': 'city'}, inplace=True)
        clustersized_circuits, fig = clustering.clusterize_circuits(df=clustersized_circuits, verbose=verbose, fig_verbose=True)
        prereq_custom = pd.merge(prereq_custom, clustersized_circuits[['city', 'cluster_id']], left_on='city_x',right_on='city', how='left')
        prereq_custom = prereq_custom[['id', 'code_6', 'circuit_x', 'city_x', 'country_x', 'latitude', 'longitude',
                                    'first_gp_probability', 'last_gp_probability', 'cluster_id']]
        prereq_custom.columns = ['geo_id', 'code', 'circuit', 'city', 'country', 'latitude', 'longitude',
                                    'first_gp_probability', 'last_gp_probability', 'cluster_id']
        if verbose:
            print("Generated DataFrame for custom circuits:")
            print(prereq_custom['circuit'].tolist())
        return prereq_custom, fig 

    if seed is not None:
        if verbose:
            print(f"Generating circuits for population with seed={seed} and n={n}.")
        circuit_names_random = get_random_sample(n, seed=seed, db_path=db_path, info=True, verbose=verbose)
        circuits_random = get_random_sample(n, seed=seed, db_path=db_path, info=False, verbose=verbose)
        clustersized_circuits, fig = clustering.clusterize_circuits(df=circuits_random, verbose=verbose, fig_verbose=True)
        prereq_random = pd.merge(circuit_names_random, clustersized_circuits[['city', 'cluster_id']], on='city', how='left')
        prereq_random.columns = ['geo_id', 'code', 'circuit', 'city', 'country', 'latitude', 'longitude',
                                 'first_gp_probability', 'last_gp_probability', 'cluster_id']
        if verbose:
            print("Generated DataFrame for random circuits:")
            print(prereq_random['circuit'].tolist())
        return prereq_random, fig

    if season is not None:
        if verbose:
            print(f"Generating circuits for population for season={season}.")
        circuit_names = get_historical_cities(season, db_path=db_path, info=True, verbose=verbose)
        clustersized_circuits, fig = clustering.clusterize_circuits(df=circuit_names[['city', 'latitude', 'longitude']], verbose=verbose, fig_verbose=True)
        prereq = pd.merge(circuit_names, clustersized_circuits[['city', 'cluster_id']], on='city', how='left')
        if verbose:
            print("Generated DataFrame for historical circuits:")
            print(prereq['circuit'].tolist())
        return prereq, fig