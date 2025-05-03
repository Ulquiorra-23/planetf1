# --- Standard Library Imports ---
import random
import functools
import sys

# --- Third-Party Library Imports ---
import numpy as np
import pandas as pd

# --- Custom Library Imports ---
from models import genetic_ops
from utils.utilities import get_circuits_for_population
from utils.sql import get_table

# --- DEAP Imports ---
try:
    from deap import base, creator, tools, algorithms
except ImportError:
    print("DEAP library not found. Please install it using: pip install deap")
    exit()  # Exit or handle appropriately if DEAP is missing

# --- Main GA Execution ---

def update_param(params: dict, key: str, value, verbose: bool = False) -> dict:
    """
    Update a specific parameter in the parameters dictionary with a datatype check.

    Parameters:
    - params: dict, the parameters dictionary to update.
    - key: str, the key of the parameter to update.
    - value: any, the new value for the parameter.
    - verbose: bool, whether to print detailed updates.

    Returns:
    - dict, the updated parameters dictionary.
    """
    if key in params:
        if not isinstance(value, type(params[key])):
            raise TypeError(f"Type mismatch for parameter '{key}': "
                            f"expected {type(params[key]).__name__}, got {type(value).__name__}.")
        if verbose:
            print(f"Updating parameter '{key}' from {params[key]} to {value}.")
    else:
        if verbose:
            print(f"Adding new parameter '{key}' with value {value}.")
    params[key] = value
    return params

def set_default_params(params: dict) -> dict:
    """
    Update the given parameters dictionary with default values if not already set.

    Parameters:
    - params: dict, the input parameters.

    Returns:
    - dict, the updated parameters with defaults applied.
    """
    defaults = {
        "POPULATION_SIZE": 100,
        "CROSSOVER_PROB": 0.8,  # Probability of mating two individuals
        "MUTATION_PROB": 0.15,  # Probability of mutating an individual
        "NUM_GENERATIONS": 50,  # Start with fewer generations for testing
        "TOURNAMENT_SIZE": 5,  # For tournament selection
        "RANDOM_SEED": 42,
        "SEASON_YEAR": 2026, # For fitness calculation context
        "REGRESSION": False,  # Set to True for regression estimates
        "CLUSTERS": True,  # Set to True for clustering
    }
    # Update params with defaults if not already set
    for key, value in defaults.items():
        params.setdefault(key, value)
    return params

# Example usage
params = {}
params = set_default_params(params)


# Set random seed
random.seed(params["RANDOM_SEED"])
np.random.seed(params["RANDOM_SEED"])

# --- Prepare Scenario Data ---
def prepare_scenario(from_season: int = None, from_sample: int = None, from_input: list = None, verbose: bool = False):
    """
    Prepare the scenario data for the genetic algorithm.

    Parameters:
    - from_season: int, the season year to fetch data for.
    - from_sample: int, the sample size to fetch data for.
    - from_input: list, additional input for customization.
    - verbose: bool, whether to print detailed information.

    Returns:
    - circuits_df_scenario: pd.DataFrame, the prepared scenario data.
    - circuit_list_scenario: list, list of circuit names.
    """
    # Ensure only one argument is populated
    args = [from_season, from_sample, from_input]
    if sum(arg is not None for arg in args) != 1:
        raise ValueError("Only one of 'from_season', 'from_sample', or 'from_input' must be provided.")

    # Fetch and prepare the scenario data
    if from_season is not None:
        circuits_df_scenario = get_circuits_for_population(season=from_season, verbose=verbose)[['code', 'cluster_id', 'first_gp_probability', 'last_gp_probability']]
        if verbose:
            print(f"Scenario prepared using season: {from_season}")
    
    if from_sample is not None:
        circuits_df_scenario = get_circuits_for_population(n=from_sample, seed=params["RANDOM_SEED"], verbose=verbose)[['code', 'cluster_id', 'first_gp_probability', 'last_gp_probability']]
        if verbose:
            print(f"Scenario prepared using sample size: {from_sample}")
        
    if from_input is not None:
        circuits_df_scenario = get_circuits_for_population(custom=from_input, verbose=verbose)[['code', 'cluster_id', 'first_gp_probability', 'last_gp_probability']]
        if verbose:
            print(f"Scenario prepared using custom input: {from_input}")
        
    circuits_df_scenario.columns = ['circuit_name', 'cluster_id', 'start_freq_prob', 'end_freq_prob']
    circuit_list_scenario = circuits_df_scenario['circuit_name'].tolist()
    
    # Print details for debugging if verbose
    if verbose:
        print(f"Optimizing for {len(circuit_list_scenario)} circuits.")
        print(f"circuit_list_scenario: {circuit_list_scenario}")
    
    return circuits_df_scenario

# --- DEAP Setup ---
# Create Fitness and Individual types
# weights=(-1.0,) means we want to minimize the fitness score
def deap_toolbox(circuits_df_scenario: pd.DataFrame, fitness_function: callable, params:dict, seed:int=None, verbose=False):
    """
    Create and configure a DEAP toolbox for the genetic algorithm.

    Parameters:
    - circuits_df_scenario: pd.DataFrame, the scenario data for the circuits.
    - fitness_function: callable, the fitness function to evaluate individuals.
    - seed: int, random seed for reproducibility.
    - verbose: bool, whether to print detailed information.

    Returns:
    - toolbox: deap.base.Toolbox, the configured DEAP toolbox.
    - stats: deap.tools.support.Statistics, the statistics object for logging.
    """
    if verbose:
        print("Initializing DEAP toolbox...")

    # Create Fitness and Individual types
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Initialize Toolbox
    toolbox = base.Toolbox()

    # Register functions for creating individuals and population
    initial_pop_list = genetic_ops.generate_initial_population(circuits_df_scenario, params['POPULATION_SIZE'], seed=seed)
    toolbox.register("population_custom", lambda: [creator.Individual(ind) for ind in initial_pop_list])

    # Register the genetic operators
    toolbox.register("evaluate", fitness_function, 
                     circuits_df=circuits_df_scenario, 
                     season=params['SEASON_YEAR'], 
                     regression=params['REGRESSION'],
                     clusters=params['CLUSTERS'], 
                     verbose=verbose)
    toolbox.register("mate", functools.partial(genetic_ops.order_crossover_deap, toolbox))
    toolbox.register("mutate", functools.partial(genetic_ops.swap_mutation_deap, toolbox))
    toolbox.register("select", tools.selTournament, tournsize=params['TOURNAMENT_SIZE'])

    # Statistics and Hall of Fame
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Create the Hall of Fame object - stores the best individual found
    hof = tools.HallOfFame(1) # Store only the single best

    if verbose:
        print("DEAP toolbox initialized successfully.")

    return toolbox, stats, hof

def run_genetic_algorithm(toolbox, stats, hof, params, verbose=False):
    """
    Run the genetic algorithm using the provided toolbox, stats, and hall of fame.

    Parameters:
    - toolbox: deap.base.Toolbox, the configured DEAP toolbox.
    - stats: deap.tools.support.Statistics, the statistics object for logging.
    - hof: deap.tools.support.HallOfFame, the hall of fame object to store the best individual.
    - params: dict, the parameters for the genetic algorithm.
    - verbose: bool, whether to print detailed information during execution.

    Returns:
    - population: list, the final population after running the genetic algorithm.
    - logbook: deap.tools.Logbook, the logbook containing statistics for each generation.
    - best_individual: list, the best individual found by the genetic algorithm.
    - best_fitness: float, the fitness score of the best individual.
    """
    if verbose:
        print("\n--- Starting Genetic Algorithm ---")
        print(f"Parameters: {params}")

    # Create the initial population
    population = toolbox.population_custom()
    if verbose:
        print(f"Initial Population Size: {len(population)}")

    # Run the genetic algorithm
    population, logbook = algorithms.eaSimple(
        population, toolbox,
        cxpb=params['CROSSOVER_PROB'],
        mutpb=params['MUTATION_PROB'],
        ngen=params['NUM_GENERATIONS'],
        stats=stats,
        halloffame=hof,
        verbose=verbose
    )

    # Retrieve the best individual and its fitness
    best_individual = hof[0]
    best_fitness = best_individual.fitness.values[0]

    if verbose:
        print("\n--- Genetic Algorithm Finished ---")
        print(f"Best Individual Found (Calendar Sequence): {best_individual}")
        print(f"Best Fitness Score Found: {best_fitness}")
        print(f"Logbook: {logbook}")

    return population, logbook, best_individual, best_fitness



if __name__ == "__main__":

    # Check for command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python run_ga.py <scenario_choice> [additional_input]")
        print("Scenario Choices:")
        print("1 - From Season (requires a year as additional input)")
        print("2 - From Random Sample (requires a sample size as additional input)")
        print("3 - From Custom List (requires a comma-separated list of integers as additional input)")
        sys.exit(1)

    # Parse scenario choice
    scenario_choice = int(sys.argv[1])
    additional_input = None

    if scenario_choice == 1:
        if len(sys.argv) < 3:
            print("Error: Year input required for scenario choice 1.")
            sys.exit(1)
        additional_input = int(sys.argv[2])
        circuits_df_scenario = prepare_scenario(from_season=additional_input, verbose=True)
    elif scenario_choice == 2:
        if len(sys.argv) < 3:
            print("Error: Sample size input required for scenario choice 2.")
            sys.exit(1)
        additional_input = int(sys.argv[2])
        circuits_df_scenario = prepare_scenario(from_sample=additional_input, verbose=True)
    elif scenario_choice == 3:
        if len(sys.argv) < 3:
            print("Error: Comma-separated list input required for scenario choice 3.")
            sys.exit(1)
        additional_input = list(map(int, sys.argv[2].split(',')))
        print(f"Custom input: {additional_input}")
        circuits_df_scenario = prepare_scenario(from_input=additional_input, verbose=True)
    else:
        print("Error: Invalid scenario choice. Must be 1, 2, or 3.")
        sys.exit(1)

    # Allow user to update parameters
    print("\nCurrent Parameters:")
    for key, value in params.items():
        print(f"{key} (type: {type(value).__name__}): {value}")

    print("\nWould you like to update any parameters? (yes/no)")
    if input().strip().lower() == "yes":
        while True:
            print("\nEnter the parameter name to update (or type 'done' to finish):")
            param_name = input().strip()
            if param_name == "done":
                break
            if param_name not in params:
                print(f"Error: Parameter '{param_name}' does not exist.")
                continue
            print(f"Enter the new value for '{param_name}' (current value: {params[param_name]}):")
            new_value = input().strip()
            try:
                # Convert input to the correct type
                new_value = type(params[param_name])(new_value)
                params = update_param(params, param_name, new_value, verbose=True)
            except ValueError:
                print(f"Error: Invalid value for parameter '{param_name}'. Expected type: {type(params[param_name]).__name__}.")
            except TypeError as e:
                print(e)

    # Run the genetic algorithm
    toolbox, stats, hof = deap_toolbox(circuits_df_scenario, genetic_ops.calculate_fitness, params, seed=params['RANDOM_SEED'], verbose=True)
    pop, log, best_ind, best_fitness = run_genetic_algorithm(toolbox, stats, hof, params, verbose=True)