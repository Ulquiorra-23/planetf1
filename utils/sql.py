import sqlite3
import pandas as pd
from pathlib import Path

def get_table(table_name: str, db_path: str, verbose: bool = False) -> pd.DataFrame:
    """
    Fetches the contents of a table from the specified SQLite database path.
    """
    connection = None
    try:
        # Add check for file existence
        if not Path(db_path).is_file():
             raise FileNotFoundError(f"Database file not found at specified path: {db_path}")

        if verbose:
            print(f"Connecting to database for table '{table_name}' at: {db_path}")
        conn = sqlite3.connect(db_path)
        if verbose:
            print(f"Fetching data from table '{table_name}'...")
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, conn)
        if verbose:
            print(f"Successfully fetched {len(df)} rows from table '{table_name}'.")
    # Add specific exception types
    except sqlite3.Error as e:
        raise ConnectionError(f"Failed to connect to or query the database at '{db_path}': {e}")
    except FileNotFoundError as e:
         raise e # Re-raise file not found
    except Exception as e:
        raise RuntimeError(f"Error fetching data from table '{table_name}' at '{db_path}': {e}")
    finally:
        # Check if connection exists before closing
        if conn:
            conn.close()
            if verbose:
                print("Database connection closed.")
    return df

def get_circuits_by(key: str, key_values: list, result_key: str, db_path: str, verbose: bool = False) -> list:
    """
    Fetches specific values from the 'fone_geography' table using the specified database path.
    """
    connection = None
    try:
        # Add check for file existence
        if not Path(db_path).is_file():
             raise FileNotFoundError(f"Database file not found at specified path: {db_path}")

        if verbose:
            print(f"Connecting to database at {db_path}...")
        # Use the passed db_path argument here
        conn = sqlite3.connect(db_path)
        # ... (rest of try block, using parameter substitution is safer) ...
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
        raise RuntimeError(f"Error fetching data from table 'fone_geography': {e} | query: {key_values}")
    finally:
        # Ensure the connection is closed
        conn.close()
        if verbose:
            print("Database connection closed.")
    
    return final_list