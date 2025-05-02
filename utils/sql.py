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
    db_path = "planet_fone.db"  # Path to the database file
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