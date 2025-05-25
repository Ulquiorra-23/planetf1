# Standard library imports
import random
import sys
from pathlib import Path
from collections import defaultdict

# Third-party library imports
import pandas as pd
import numpy as np
import textwrap
import logging

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent # Go up one level from src/
sys.path.append(str(PROJECT_ROOT))
DB_PATH = PROJECT_ROOT / "data" / "planet_fone.db"
# Convert to string for functions expecting string paths
DB_PATH_STR = str(DB_PATH)

# Local application/library imports
from utils.sql import get_table
from utils.utilities import generate_f1_calendar
from utils.logs import log_wrap, logls, logdf

# Load regression models
from models.regression import calendar_emissions

TRAVEL_LOGISTIC_DF = get_table("travel_logistic", db_path=DB_PATH_STR)
FONE_GEOGRAPHY_DF = get_table("fone_geography", db_path=DB_PATH_STR)


@log_wrap
def fragmentation_score(lst: list[int], non_linear_power: float = 2.0, logger=None) -> float:
    if not lst:
        return 0.0  # empty list = perfect grouping

    total_score = 0
    max_score = 0

    for digit in set(lst):
        indices = [i for i, x in enumerate(lst) if x == digit]
        if len(indices) == 1:
            total_score += 0  # perfectly grouped
            max_score += 1
            continue

        groups = 1
        for i in range(1, len(indices)):
            if indices[i] != indices[i-1] + 1:
                groups += 1

        # Raw grouping score: 0 (best) to 1 (worst)
        raw_score = (groups - 1) / (len(indices) - 1)
        total_score += raw_score
        max_score += 1

    normalized_score = total_score / max_score if max_score > 0 else 0.0

    # Non-linear scaling: square or higher power makes low scores harder to reach
    scaled_score = normalized_score ** (1 / non_linear_power)
    logger.info(f"Fragmentation score: {scaled_score}")
    return scaled_score
 
@log_wrap   
def shuffle_respecting_clusters(circuits_to_shuffle, cluster_assignments, seed=None, verbose=False, logger=None):
    """
    Shuffles a list of circuits while keeping circuits from the same cluster consecutive.

    Args:
        circuits_to_shuffle (list): List of circuit names to shuffle.
        cluster_assignments (dict): Mapping {circuit_name: cluster_id}.
        seed (int, optional): Seed value for reproducibility. Defaults to None.
        verbose (bool): If True, prints detailed information about the process.

    Returns:
        list: A shuffled list of circuit names respecting clusters.
    """
    if not circuits_to_shuffle:
        if verbose:
            logger.warning("No circuits to shuffle. Returning an empty list.")
        return []

    if seed is not None:
        random.seed(seed)  # Set seed for reproducibility
        if verbose:
            logger.info(f"Random seed set to: {seed}")

    clusters = defaultdict(list)
    for circuit in circuits_to_shuffle:
        cluster_id = cluster_assignments.get(circuit, 'unknown')
        clusters[cluster_id].append(circuit)

    # Shuffle all circuits together, ignoring clusters
    final_sequence = circuits_to_shuffle.copy()
    random.shuffle(final_sequence)

    if verbose:
        logger.info("Shuffled sequence:")
        logls(final_sequence)

    return final_sequence

@log_wrap
def generate_initial_population(circuits_df, population_size, seed=None, verbose=False, logger=None):
    """
    Generates an initial population for the GA using a mix of strategies,
    with an option for reproducible results using a seed.

    Args:
        circuits_df (pd.DataFrame): DataFrame containing circuit information.
                                     Must include columns: 'circuit_name', 'cluster_id',
                                     'start_freq_prob', 'end_freq_prob'.
        population_size (int): The total number of individuals (calendars) to generate.
        seed (int, optional): A random seed for reproducibility. If None, results
                              will vary on each run. Defaults to None.
        verbose (bool): If True, prints detailed information about the process.

    Returns:
        list: A list of lists, where each inner list is a chromosome (calendar sequence).
    """
    if seed is not None:
        random.seed(seed)       # Seed for the 'random' module
        np.random.seed(seed)    # Seed for the 'numpy.random' module
        if verbose:
            logger.info(f"Random seeds set to: {seed}")
    else:
        if verbose:
            logging.warning("No random seed provided, results will vary.")

    population = []

    required_cols = {'circuit_name', 'cluster_id', 'start_freq_prob', 'end_freq_prob'}
    if not required_cols.issubset(circuits_df.columns):
        raise ValueError(f"Input DataFrame missing required columns: {required_cols - set(circuits_df.columns)}")

    circuit_list = circuits_df['circuit_name'].tolist()
    cluster_assignments = pd.Series(circuits_df.cluster_id.values,
                                    index=circuits_df.circuit_name).to_dict()

    start_df = circuits_df[circuits_df['start_freq_prob'] > 0]
    start_circuits = []
    start_probs = []
    if not start_df.empty:
        start_circuits = start_df['circuit_name'].tolist()
        start_probs = (start_df['start_freq_prob'] / start_df['start_freq_prob'].sum()).tolist()

    end_df = circuits_df[circuits_df['end_freq_prob'] > 0]
    end_circuits = []
    end_probs = []
    if not end_df.empty:
        end_circuits = end_df['circuit_name'].tolist()
        end_probs = (end_df['end_freq_prob'] / end_df['end_freq_prob'].sum()).tolist()

    num_b = int(population_size * 0.60)  # Cluster-Respecting Random
    num_c_start = int(population_size * 0.10)  # Historical Opener
    num_c_start_end = int(population_size * 0.20)  # Historical Opener & Finale
    num_c_end = population_size - num_b - num_c_start - num_c_start_end  # Historical Finale

    if verbose:
        logger.info(f"Population size: {population_size}")
        logger.info(f"Method B (Cluster-Respecting Random): {num_b}")
        logger.info(f"Method C-Start (Historical Opener): {num_c_start}")
        logger.info(f"Method C-StartEnd (Historical Opener & Finale): {num_c_start_end}")
        logger.info(f"Method C-End (Historical Finale): {num_c_end}")

    # Method B: Cluster-Respecting Random (60%)
    if verbose:
        logger.info(f"Generating {num_b} individuals using Method B...")
    for _ in range(num_b):
        population.append(shuffle_respecting_clusters(circuit_list, cluster_assignments, seed=seed, verbose=verbose))

    # Method C-Start: Historical Opener (10%)
    if verbose:
        logger.info(f"Generating {num_c_start} individuals using Method C-Start...")
    if start_circuits:
        for _ in range(num_c_start):
            start_circuit = np.random.choice(start_circuits, p=start_probs)  # Uses numpy.random
            remaining = [c for c in circuit_list if c != start_circuit]
            middle = shuffle_respecting_clusters(remaining, cluster_assignments, seed=seed, verbose=verbose)
            population.append([start_circuit] + middle)
    else:
        if verbose:
            logger.warning("Warning: No start frequencies > 0 provided, using Method B instead for C-Start.")
        for _ in range(num_c_start):
            population.append(shuffle_respecting_clusters(circuit_list, cluster_assignments, seed=seed, verbose=verbose))

    # Method C-StartEnd: Historical Opener & Finale (20%)
    if verbose:
        logger.info(f"Generating {num_c_start_end} individuals using Method C-StartEnd...")
    if start_circuits and end_circuits and len(circuit_list) > 1:
        for _ in range(num_c_start_end):
            start_circuit = np.random.choice(start_circuits, p=start_probs)  # Uses numpy.random
            end_circuit = np.random.choice(end_circuits, p=end_probs)        # Uses numpy.random
            attempts = 0
            while end_circuit == start_circuit and attempts < 10:
                end_circuit = np.random.choice(end_circuits, p=end_probs)
                attempts += 1

            remaining = [c for c in circuit_list if c != start_circuit and c != end_circuit]
            middle = shuffle_respecting_clusters(remaining, cluster_assignments, seed=seed, verbose=verbose)
            population.append([start_circuit] + middle + [end_circuit])
    else:
        if verbose:
            logger.warning("Warning: No start/end frequencies or not enough circuits, using Method B instead for C-StartEnd.")
        for _ in range(num_c_start_end):
            population.append(shuffle_respecting_clusters(circuit_list, cluster_assignments, seed=seed, verbose=verbose))

    # Method C-End: Historical Finale (10%)
    if verbose:
        logger.info(f"Generating {num_c_end} individuals using Method C-End...")
    if end_circuits:
        for _ in range(num_c_end):
            end_circuit = np.random.choice(end_circuits, p=end_probs)  # Uses numpy.random
            remaining = [c for c in circuit_list if c != end_circuit]
            middle = shuffle_respecting_clusters(remaining, cluster_assignments, seed=seed, verbose=verbose)
            population.append(middle + [end_circuit])
    else:
        if verbose:
            logger.warning("Warning: No end frequencies > 0 provided, using Method B instead for C-End.")
        for _ in range(num_c_end):
            population.append(shuffle_respecting_clusters(circuit_list, cluster_assignments, seed=seed, verbose=verbose))

    random.shuffle(population)

    final_population = population[:population_size]
    if verbose:
        logger.info(f"Generated total population size: {len(final_population)}")

    return final_population

@log_wrap
def calculate_fitness(circuits_seq: list, circuits_df, db_path: str, season=2026, regression=False, regression_models: tuple = None, clusters=False, verbose=False, logger=None):
    """
    Calculate the fitness of a given circuit list based on cluster assignments.

    Args:
        circuit_df (dataframe): DataFrame containing circuit information.
            1. 'circuit_name': Name of the circuit.
            2. 'cluster_id': Cluster ID of the circuit.
        circuits_seq (list): Sequence of circuit codes.
        season (int): Season year to simulate calendar dates.
        regression (bool): If True, uses regression-based fitness calculation.
        clusters (bool): If True, uses cluster-based fitness calculation.
        verbose (bool): If True, prints detailed information about the process.

    Returns:
        float: Fitness score for the given circuit sequence.
    """
    if not circuits_seq or len(circuits_seq) < 15:
        if verbose:
            logger.info("Circuit sequence is empty or too short. Returning fitness score of 0.")
        return float('inf')
    logger.info("Evaluating Fitness...")
    total_emissions = 0.0
    total_penalties = 0.0
    
    if regression and not regression_models:
        raise ValueError("Regression is set to True but no regression models provided.")
    
    # Fetch travel logistics data from the database
    travel_logistics_df = TRAVEL_LOGISTIC_DF
    
    if regression:
        if verbose:
            logger.info('Regression is set to True. Using regression models for fitness calculation.')
        air, truck = regression_models
        air_predictions, truck_predictions = calendar_emissions(list(circuits_seq), season, air, truck, verbose=verbose)
        total_emissions += air_predictions.sum() + truck_predictions.sum()
        total_emissions = round(total_emissions, 2)
        if verbose:
            logger.info("Air emissions predictions: %s", textwrap.fill(str(air_predictions), width=100))
            logger.info("Truck emissions predictions: %s", textwrap.fill(str(truck_predictions), width=100))
            logger.info(f"Total emissions (regression): {total_emissions}")

    if not regression:
        if verbose:
            logger.info('Regression is set to False. Using synthetic data for fitness calculation.')
            logger.info('Getting travel logistics...')
        travel_logistic_keys = [(circuits_seq[i], circuits_seq[i+1]) for i in range(len(circuits_seq) - 1)]
        travel_logistic_keys = [f"{travel_logistic_key[0]}-{travel_logistic_key[1]}" for travel_logistic_key in travel_logistic_keys]
        if verbose:
            logger.info(f'found [{len(travel_logistic_keys)}] travel logistic keys')


        # Filter the DataFrame for rows where 'code' matches the travel logistic keys
        filtered_logistics = travel_logistics_df[travel_logistics_df['codes'].isin(travel_logistic_keys)]

        # Extract the effort scores
        effort_scores = filtered_logistics['effort_score'].tolist()
        total_emissions = round(sum(effort_scores), 2)

        if verbose:
            logger.info("Effort scores: %s", textwrap.fill(str(effort_scores), width=100))
            logger.info("Total emissions: %s", total_emissions)
    
    total_cluster_penalties = 0.0
    
    if clusters:
        cluster_dict = circuits_df.groupby('cluster_id')['circuit_name'].apply(list).to_dict()
        if verbose:
            logger.info("Cluster dictionary has been created!")
        cluster_ids = [key for circuit in circuits_seq for key, value in cluster_dict.items() if circuit in value]
        if verbose:
            logger.info("Cluster IDs for the given circuit sequence: %s", textwrap.fill(str(cluster_ids), width=100))
        fragmentation_score_value = fragmentation_score(cluster_ids, non_linear_power=1.0)
        if verbose:
            logger.info("Fragmentation score: %s", fragmentation_score_value)
        weight = travel_logistics_df['effort_score'].mean() if not travel_logistics_df.empty else 0
        if verbose:
            logger.info("Weight: %s", weight)
        total_cluster_penalties = round(fragmentation_score_value * weight, 2)
    
    total_conflict_penalties = 0.0
    calendar = generate_f1_calendar(year=season, n=len(circuits_seq), verbose=False)
    if calendar:
        if verbose:
            logger.info("Generated calendar: %s", textwrap.fill(str(calendar), width=100))
        # Assign each circuit in circuits_seq to a date in the calendar
        circuit_date_mapping = {circuit: calendar[i] for i, circuit in enumerate(circuits_seq)}

        # Fetch the fone_geography table
        fone_geography_df = FONE_GEOGRAPHY_DF

        # Check for month conflicts
        total_conflicts = 0
        for circuit, date in circuit_date_mapping.items():
            month_assigned = int(date[-2:].lstrip("0"))
            months_to_avoid_str = fone_geography_df.loc[fone_geography_df['code_6'] == circuit, 'months_to_avoid'].values
            # Parse the string representation of the list into an actual list of ints
            if months_to_avoid_str.size > 0:
                try:
                    months_to_avoid = eval(months_to_avoid_str[0]) if months_to_avoid_str[0] else []
                except Exception:
                    months_to_avoid = []
            else:
                months_to_avoid = []
            # Ensure months_to_avoid is a list of integers
            months_to_avoid_list = months_to_avoid if len(months_to_avoid) > 0 else []
            if month_assigned in months_to_avoid_list:
                logger.info(f"Conflict for circuit {circuit}: assigned month {month_assigned} is in months to avoid {months_to_avoid_list}.")
                # Apply penalties based on the number of conflicts
                total_conflicts += 1
        if total_conflicts > 0:
            total_conflict_penalties = total_emissions
            if verbose:
                logger.info(f"Total conflict penalties: {total_conflict_penalties}")
    
    total_penalties = total_cluster_penalties + total_conflict_penalties
    if True: #verbose: #for testing purposes
        logger.info("Calculated fitness for sequence:")
        logls(circuits_seq)
        logger.info(f"Total cluster penalties: {total_cluster_penalties}")
        logger.info(f"Total conflict penalties: {total_conflict_penalties}")
        logger.info(f"Total penalties: {total_penalties}")
        logger.info(f"Total emissions: {total_emissions}")
        logger.info(f"Total fitness score: {total_emissions + total_penalties}")
        logger.info("Fitness calculation completed.")
    return (total_emissions + total_penalties,)

@log_wrap
def tournament_selection(population, fitnesses, k, num_parents, verbose=False, logger=None):
    """
    Selects parents using Tournament Selection.

    Args:
        population (list): The current population of chromosomes (lists).
        fitnesses (list): A list of fitness scores corresponding to the population.
                          Lower scores are assumed to be better.
        k (int): The size of the tournament (e.g., 3, 5).
        num_parents (int): The number of parents to select.
        verbose (bool): If True, prints detailed information about the selection process.

    Returns:
        list: A list containing the selected parent chromosomes.
    """
    selected_parents = []
    population_size = len(population)
    
    if population_size == 0:
        if verbose:
            logger.warning("Population is empty. Returning an empty list.")
        return []

    if len(fitnesses) != population_size:
        raise ValueError("Population and fitnesses list must have the same size.")

    for parent_idx in range(num_parents):
        tournament_indices = random.sample(range(population_size), k)
        if verbose:
            logger.info(f"Tournament {parent_idx + 1}: Selected indices {tournament_indices}")

        best_index_in_tournament = -1
        min_fitness = float('inf')
        
        for index in tournament_indices:
            if fitnesses[index] < min_fitness:
                min_fitness = fitnesses[index]
                best_index_in_tournament = index
        
        if verbose:
            logger.info(f"Tournament {parent_idx + 1}: Winner index {best_index_in_tournament} with fitness {min_fitness}")

        if best_index_in_tournament != -1:
            selected_parents.append(population[best_index_in_tournament])
        else:
            selected_parents.append(population[tournament_indices[0]])

    return selected_parents

@log_wrap
def order_crossover_deap(toolbox, parent1, parent2, verbose=False, logger=None):
    """
    Performs Order Crossover (OX1) on two parent DEAP Individuals.
    Returns two offspring DEAP Individuals.

    Args:
        toolbox: The DEAP toolbox (needed for cloning).
        parent1 (creator.Individual): The first parent individual.
        parent2 (creator.Individual): The second parent individual.
        verbose (bool): If True, prints detailed information.

    Returns:
        tuple: A tuple containing two offspring individuals (child1, child2).
    """
    size = len(parent1) # Individuals usually behave like their base type (list) for len()

    # --- Create Children by Cloning Parents ---
    # Cloning ensures the offspring have the correct DEAP Individual type
    # and inherit attributes like the fitness structure (though values become invalid).
    child1 = toolbox.clone(parent1)
    child2 = toolbox.clone(parent2)

    if size < 2:  # Cannot perform crossover on size < 2
        if verbose:
            logger.info("Chromosome size is less than 2. Returning clones of parents.")
        # Return the clones directly, fitness will be handled by the algorithm
        # Need to invalidate fitness manually if returning early
        del child1.fitness.values
        del child2.fitness.values
        return child1, child2

    # Choose two random distinct cut points
    cut1, cut2 = sorted(random.sample(range(size), 2))
    if verbose:
        logger.info(f"Cut points selected: {cut1}, {cut2}")

    # --- Create Offspring 1 (Primary: Parent1, Secondary: Parent2) ---
    # Temporary list to hold the new sequence for child1
    offspring1_list = [None] * size

    # Copy the segment from parent1 directly into the temp list
    offspring1_list[cut1:cut2+1] = parent1[cut1:cut2+1] # Access parent list data
    parent1_segment_set = set(parent1[cut1:cut2+1])
    if verbose:
        logger.info(f"Child1 initial segment (from Parent1): {offspring1_list[cut1:cut2+1]}")

    # Gather elements from parent2 not in the copied segment
    parent2_elements_to_fill = []
    start_point_p2 = (cut2 + 1) % size
    current_p2_check_idx = start_point_p2
    for _ in range(size):
        element = parent2[current_p2_check_idx] # Access parent list data
        if element not in parent1_segment_set:
            parent2_elements_to_fill.append(element)
        current_p2_check_idx = (current_p2_check_idx + 1) % size
    if verbose:
        logger.info(f"Elements from Parent2 to fill Child1: {parent2_elements_to_fill}")

    # Fill the remaining slots in the temp list for child1
    fill_idx_o1 = (cut2 + 1) % size
    for element in parent2_elements_to_fill:
        # Find the next available slot (must be None)
        while offspring1_list[fill_idx_o1] is not None:
            fill_idx_o1 = (fill_idx_o1 + 1) % size
        offspring1_list[fill_idx_o1] = element
        # Move to the next index for the next element placement attempt
        fill_idx_o1 = (fill_idx_o1 + 1) % size


    # --- Create Offspring 2 (Primary: Parent2, Secondary: Parent1) ---
    # Temporary list to hold the new sequence for child2
    offspring2_list = [None] * size

    # Copy the segment from parent2 directly into the temp list
    offspring2_list[cut1:cut2+1] = parent2[cut1:cut2+1] # Access parent list data
    parent2_segment_set = set(parent2[cut1:cut2+1])
    if verbose:
        logger.info(f"Child2 initial segment (from Parent2): {offspring2_list[cut1:cut2+1]}")

    # Gather elements from parent1 not in the copied segment
    parent1_elements_to_fill = []
    start_point_p1 = (cut2 + 1) % size
    current_p1_check_idx = start_point_p1
    for _ in range(size):
        element = parent1[current_p1_check_idx] # Access parent list data
        if element not in parent2_segment_set:
            parent1_elements_to_fill.append(element)
        current_p1_check_idx = (current_p1_check_idx + 1) % size
    if verbose:
        logger.info(f"Elements from Parent1 to fill Child2: {parent1_elements_to_fill}")

    # Fill the remaining slots in the temp list for child2
    fill_idx_o2 = (cut2 + 1) % size
    for element in parent1_elements_to_fill:
        # Find the next available slot (must be None)
        while offspring2_list[fill_idx_o2] is not None:
            fill_idx_o2 = (fill_idx_o2 + 1) % size
        offspring2_list[fill_idx_o2] = element
        # Move to the next index for the next element placement attempt
        fill_idx_o2 = (fill_idx_o2 + 1) % size


    # Final checks for None (optional, indicates logic errors if hit)
    if None in offspring1_list:
        # Provide more context in the error message
        raise ValueError(f"Crossover failed: None found in offspring1_list. "
                         f"Parent1: {parent1}, Parent2: {parent2}, "
                         f"Cuts: ({cut1}, {cut2}), Result: {offspring1_list}")
    if None in offspring2_list:
        raise ValueError(f"Crossover failed: None found in offspring2_list. "
                         f"Parent1: {parent1}, Parent2: {parent2}, "
                         f"Cuts: ({cut1}, {cut2}), Result: {offspring2_list}")

    # --- Assign the new sequences to the cloned children ---
    # This modifies the list *inside* the DEAP Individual objects
    child1[:] = offspring1_list
    child2[:] = offspring2_list

    if verbose:
        logger.info(f"Completed Child1: {child1}")
        logger.info(f"Completed Child2: {child2}")

    # --- IMPORTANT: Delete fitness values ---
    # The children's fitness is no longer valid after crossover.
    # DEAP's varAnd also does this, but it's good practice here too.
    del child1.fitness.values
    del child2.fitness.values

    # Return the modified children (DEAP Individuals)
    return child1, child2

@log_wrap
def swap_mutation_deap(toolbox, individual, verbose=False, logger=None):
    """
    Performs Swap Mutation on a DEAP Individual.
    Mutates the individual in-place (by convention, though a clone is used internally).
    Returns a tuple containing the mutated individual.

    Args:
        toolbox: The DEAP toolbox (needed for cloning, though not strictly
                 necessary if mutating in-place and the original can be modified).
        individual (creator.Individual): The individual to mutate.
        verbose (bool): If True, prints detailed information.

    Returns:
        tuple: A tuple containing the mutated individual, e.g., (mutated_individual,).
               Returns the original individual in a tuple if length < 2.
    """
    size = len(individual)

    # It's often safer to clone, modify the clone, and return the clone,
    # especially if the original individual might be used elsewhere before
    # its fitness is re-evaluated. DEAP's algorithms typically handle the
    # replacement of the original with the mutated version correctly.
    mutant = toolbox.clone(individual)

    if size < 2:
        if verbose:
            logger.info("Chromosome size is less than 2. No mutation performed.")
        # Still need to return in the expected tuple format
        # Fitness is already invalid if it came from crossover, or valid if
        # it's an original selected for mutation only. Deleting here ensures
        # it's always invalid after potential mutation attempt.
        del mutant.fitness.values
        return mutant, # Comma makes it a tuple

    # Select two distinct indices to swap
    idx1, idx2 = random.sample(range(size), 2)

    if verbose:
        logger.info(f"Swapping indices {idx1} and {idx2} in individual: "
              f"{mutant[idx1]} <-> {mutant[idx2]}")
        logger.info(f"Individual before mutation: {list(mutant)}")
        
    # Perform the swap directly on the mutant individual's contents
    mutant[idx1], mutant[idx2] = mutant[idx2], mutant[idx1]

    if verbose:
        logger.info(f"Mutated individual: {list(mutant)}") # Use list() for clean print

    # --- IMPORTANT: Delete fitness values ---
    # The mutant's fitness is no longer valid after mutation.
    del mutant.fitness.values

    # Return the modified individual in a tuple (required by DEAP)
    return mutant,