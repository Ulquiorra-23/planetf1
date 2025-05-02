# Standard library imports
import json
import datetime
import random

# Third-party library imports
import pandas as pd
import numpy as np

# Local application/library imports
from utils.sql import get_table

def get_historical_seq(verbose = False):
    """
    Fetches historical sequence data from the database, processes it, and returns a JSON string.
    """
    historical_races = get_table('fone_calendar')
    geography = get_table('fone_geography')
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


def get_circuit_for_pop(id: int, verbose: bool = False):
    """
    Fetches the circuit data for a given ID from the database and returns it as a DataFrame.
    """
    circuits = get_table('fone_geography')
    if verbose:
        print("Fetched circuit data from the database.")        
    circuit = circuits[circuits['id'] == id][['id', 'code_6', 'circuit_x', 'city_x', 'country_x', 'latitude', 'longitude', 'first_gp_probability', 'last_gp_probability']]
    if verbose:
        print(f"Fetched circuit details for ID {id}: {circuit}.")
    return circuit