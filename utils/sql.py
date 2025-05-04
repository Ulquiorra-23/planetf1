import sqlite3
import pandas as pd

def get_table(table_name: str, verbose: bool = False) -> pd.DataFrame:
    """
    Fetches the contents of a table from the planet_fone.db database and returns it as a pandas DataFrame.

    Args:
        table_name (str): The name of the table to retrieve.
        verbose (bool): If True, prints additional information. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing the table's data.
    """
    db_path = r"data\planet_fone.db"  # Path to the database file
    try:
        if verbose:
            print(f"Connecting to database at {db_path}...")
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        if verbose:
            print(f"Fetching data from table '{table_name}'...")
        # Query the table and load it into a DataFrame
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, conn)
        if verbose:
            print(f"Successfully fetched {len(df)} rows from table '{table_name}'.")
    except Exception as e:
        raise RuntimeError(f"Error fetching data from table '{table_name}': {e}")
    finally:
        # Ensure the connection is closed
        conn.close()
        if verbose:
            print("Database connection closed.")
    
    return df

def get_circuits_by(key: str, key_values: list, result_key: str, verbose: bool = False) -> list:
    """
    Fetches specific values from the 'fone_geography' table in the planet_fone.db database based on a given key and its values.

    Args:
        key (str): The column name to filter by.
        key_values (list): A list of values to match in the specified key column.
        result_key (str): The column name whose values should be returned.
        verbose (bool): If True, prints additional information. Defaults to False.

    Returns:
        list: A list of values from the result_key column that match the filter criteria.
    """
    db_path = r"data\planet_fone.db"  # Path to the database file
    try:
        if verbose:
            print(f"Connecting to database at {db_path}...")
        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        if verbose:
            print(f"Fetching data from table 'fone_geography'...")
        # Query the table and load it into a DataFrame
        key_values_str = "', '".join(key_values)
        key_values_str = f"('{key_values_str}')"
        query = f"SELECT {key}, {result_key} FROM fone_geography WHERE {key} IN {key_values_str}"
        df = pd.read_sql_query(query, conn)
        final_list = df.to_dict(orient='list')
        if verbose:
            print(f"Successfully fetched {len(df)} rows from table 'fone_geography'.")
    except Exception as e:
        raise RuntimeError(f"Error fetching data from table 'fone_geography': {e}")
    finally:
        # Ensure the connection is closed
        conn.close()
        if verbose:
            print("Database connection closed.")
    
    return final_list