# --- Standard Library Imports ---
import random
import functools
import sys
import json
from pathlib import Path
from datetime import datetime

# --- Third-Party Library Imports ---
import numpy as np
import pandas as pd

# --- Custom Library Imports ---
from models.regression import load_models
from models import genetic_ops
from utils.utilities import get_circuits_for_population, generate_f1_calendar
from utils.sql import get_table
from utils.logs import log_wrap, logls, logdf

# --- DEAP Imports ---
try:
    from deap import base, creator, tools, algorithms
except ImportError:
    print("DEAP library not found. Please install it using: pip install deap")
    exit()  # Exit or handle appropriately if DEAP is missing

# --- Determine Project Root and DB Path ---
# Assuming app.py is in 'src/' relative to the project root
# Adjust if your structure is different (e.g., app.py in root)
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR 
DB_PATH = PROJECT_ROOT / "data" / "planet_fone.db"
# Convert to string for functions expecting string paths
DB_PATH_STR = str(DB_PATH)
MODELS_DIR = PROJECT_ROOT / "models"
RUN_LOGS_PATH = PROJECT_ROOT / "logs" 

# Load the geography data
GEO_DF = get_table("fone_geography", db_path=DB_PATH_STR)

# Load regression models if available
try:
    air, truck = load_models()
except FileNotFoundError:
    print("Regression models not found. Ensure they are available in the 'models' directory.")
    air, truck = None, None

# --- Main GA Execution ---

@log_wrap
def update_param(params: dict, key: str, value, verbose: bool = False, logger = None) -> dict:
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
            logger.info(f"Updating parameter '{key}' from {params[key]} to {value}.")
    else:
        if verbose:
            logger.info(f"Adding new parameter '{key}' with value {value}.")
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
        "REGRESSION": True,  # Set to True for regression estimates
        "REGRESSION_MODELS": {
            "air": air,
            "truck": truck
        }, # Regression models to use
        "CLUSTERS": True,  # Set to True for clustering
        "VERBOSE": False,  # Set to True for detailed output
        "LOG_RESULTS_NAME": f"GA_RUN_OUTPUT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
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
@log_wrap
def prepare_scenario(db_path: str, from_season: int = None, from_sample: int = None, from_input: list = None, verbose: bool = False, logger = None) -> tuple:
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
        circuits_df_scenario, fig = get_circuits_for_population(db_path=db_path, season=from_season, verbose=verbose)
        if verbose:
            logger.info(f"Scenario prepared using season: {from_season}")
    
    if from_sample is not None:
        circuits_df_scenario, fig = get_circuits_for_population(db_path=db_path, n=from_sample, seed=params["RANDOM_SEED"], verbose=verbose)
        if verbose:
            logger.info(f"Scenario prepared using sample size: {from_sample}")

    if from_input is not None:
        circuits_df_scenario, fig = get_circuits_for_population(db_path=db_path, custom=from_input, verbose=verbose)
        if verbose:
            logger.info(f"Scenario prepared using custom input: {from_input}")

    circuits_df_scenario = circuits_df_scenario[['code', 'cluster_id', 'first_gp_probability', 'last_gp_probability']]
    circuits_df_scenario.columns = ['circuit_name', 'cluster_id', 'start_freq_prob', 'end_freq_prob']
    circuit_list_scenario = circuits_df_scenario['circuit_name'].tolist()
    
    # Print details for debugging if verbose
    if verbose:
        logger.info(f"Optimizing for {len(circuit_list_scenario)} circuits.")
        logls(circuit_list_scenario)

    return circuits_df_scenario, fig

# --- DEAP Setup ---
# Create Fitness and Individual types
# weights=(-1.0,) means we want to minimize the fitness score
@log_wrap
def deap_toolbox(circuits_df_scenario: pd.DataFrame, db_path: str, fitness_function: callable, params:dict, seed:int=None, verbose=False, logger = None) -> tuple:
    """
    Initialize and configure the DEAP toolbox for the genetic algorithm.

    Parameters:
    - circuits_df_scenario (pd.DataFrame): DataFrame containing scenario circuit data.
    - db_path (str): Path to the database file.
    - fitness_function (callable): Function to evaluate the fitness of individuals.
    - params (dict): Dictionary of GA parameters.
    - seed (int, optional): Random seed for reproducibility.
    - verbose (bool, optional): If True, prints detailed setup information.

    Returns:
    - toolbox (deap.base.Toolbox): Configured DEAP toolbox.
    - stats (deap.tools.Statistics): Statistics object for tracking GA progress.
    - hof (deap.tools.HallOfFame): Hall of Fame object to store the best individual(s).
    """
    if verbose:
        logger.info("Initializing DEAP toolbox...")
        logger.info("Parameters fed into toolbox:")
        for k, v in params.items():
            logger.info(f"  {k}: {v}")

    # Create Fitness and Individual types (avoid re-creation if already exists)
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    # Initialize Toolbox
    toolbox = base.Toolbox()

    # Register functions for creating individuals and population
    initial_pop_list = genetic_ops.generate_initial_population(circuits_df_scenario, params['POPULATION_SIZE'], seed=seed)
    toolbox.register("population_custom", lambda: [creator.Individual(ind) for ind in initial_pop_list])

    # Register the genetic operators
    toolbox.register("evaluate", fitness_function, 
                     circuits_df=circuits_df_scenario, 
                     db_path=db_path,
                     season=params['SEASON_YEAR'], 
                     regression=params['REGRESSION'],
                     regression_models=(params['REGRESSION_MODELS']['air'], params['REGRESSION_MODELS']['truck']),
                     clusters=params['CLUSTERS'], 
                     verbose=params['VERBOSE'])
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
        logger.info("DEAP toolbox initialized successfully.")

    return toolbox, stats, hof

@log_wrap
def run_genetic_algorithm(toolbox, stats, hof, params, circuits_df_scenario: pd.DataFrame = None, verbose=False, logger=None):
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
        logger.info("--- Starting Genetic Algorithm ---")
        logger.info(f"Parameters: {params}")

    # Save all arguments and returned value in JSON format
    start_time = datetime.now()

    # Create the initial population
    population = toolbox.population_custom()

    if verbose:
        logger.info(f"Initial Population Size: {len(population)}")

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
        logger.info("--- Genetic Algorithm Finished ---")
        logger.info(f"Best Individual Found (Calendar Sequence): {best_individual}")
        logger.info(f"Best Fitness Score Found: {best_fitness}")
        logger.info(f"Logbook: {logbook}")

    end_time = datetime.now()
    run_time = (end_time - start_time).total_seconds()

    if circuits_df_scenario is not None:
        initial_circuits = circuits_df_scenario.merge(
            GEO_DF[['code_6', 'circuit_x', 'city_x', 'country_x', 'latitude', 'longitude','months_to_avoid']],
            left_on='circuit_name',
            right_on='code_6',
            how='left'
        )
        initial_circuits = initial_circuits.drop(columns=['code_6'])
    
    RUN_LOGS_NAME = params['LOG_RESULTS_NAME']

    log_data = {
        "id": RUN_LOGS_NAME,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "run_time_seconds": run_time,
        "arguments": {
        "circuits_df" : initial_circuits.to_dict(orient='records'),
        "params": params,
        "population_size": len(population),
        },
        "results": {
        "best_individual": list(best_individual),
        "calendar": generate_f1_calendar(year=params['SEASON_YEAR'],n=len(best_individual)),
        "best_fitness": best_fitness,
        "logbook": [dict(gen) for gen in logbook],
        }
    }
    RUN_LOGS_PATH.mkdir(parents=True, exist_ok=True)
    with open(RUN_LOGS_PATH / RUN_LOGS_NAME, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, default=str)
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
        circuits_df_scenario, fig  = prepare_scenario(from_season=additional_input, db_path=DB_PATH_STR, verbose=True)
    elif scenario_choice == 2:
        if len(sys.argv) < 3:
            print("Error: Sample size input required for scenario choice 2.")
            sys.exit(1)
        additional_input = int(sys.argv[2])
        circuits_df_scenario, fig  = prepare_scenario(from_sample=additional_input, db_path=DB_PATH_STR, verbose=True)
    elif scenario_choice == 3:
        if len(sys.argv) < 3:
            print("Error: Comma-separated list input required for scenario choice 3.")
            sys.exit(1)
        additional_input = list(map(int, sys.argv[2].split(',')))
        print(f"Custom input: {additional_input}")
        circuits_df_scenario, fig  = prepare_scenario(from_input=additional_input, db_path=DB_PATH_STR, verbose=True)
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
    toolbox, stats, hof = deap_toolbox(circuits_df_scenario, DB_PATH_STR, genetic_ops.calculate_fitness, params, seed=params['RANDOM_SEED'], verbose=params['VERBOSE'])
    pop, log, best_ind, best_fitness = run_genetic_algorithm(toolbox, stats, hof, params, verbose=params['VERBOSE'])