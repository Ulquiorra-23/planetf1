# main imports
import sys
from pathlib import Path

# Third-party library imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib
import seaborn as sns


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent # Go up one level from src/
DB_PATH = PROJECT_ROOT / "data" / "planet_fone.db"
DB_PATH_STR = str(DB_PATH)
LOG_PATH = PROJECT_ROOT / "logs"
MODEL_PATH = PROJECT_ROOT / "models"

# custom imports
from utils.logs import log_wrap, logls, logdf
from utils.sql import get_table

TL_DF = get_table(table_name="travel_logistic", db_path=DB_PATH_STR)
FG_DF = get_table(table_name="fone_geography", db_path=DB_PATH_STR)
FC_DF = get_table(table_name="fone_calendar", db_path=DB_PATH_STR)

@log_wrap
def save_models(air_model, truck_model, filename_prefix='fone_regression', logger=None):
    """
    Save the trained models to disk in the specified model_path directory.
    """
    joblib.dump(air_model, MODEL_PATH / f"{filename_prefix}_air_model.pkl")
    joblib.dump(truck_model, MODEL_PATH / f"{filename_prefix}_truck_model.pkl")

    if logger:
        logger.info("Models saved successfully.")

@log_wrap
def load_models(filename_prefix='fone_regression', logger=None):
    """
    Load the trained models from disk.
    
    Returns:
        tuple: (air_model, truck_model)
    """
    air_model_path = MODEL_PATH / f"{filename_prefix}_air_model.pkl"
    truck_model_path = MODEL_PATH / f"{filename_prefix}_truck_model.pkl"

    if not air_model_path.exists() or not truck_model_path.exists():
        raise FileNotFoundError("Model files not found. Please train and save the models first.")

    air_model = joblib.load(air_model_path)
    truck_model = joblib.load(truck_model_path)

    if logger:
        logger.info("Models loaded successfully.")

    return air_model, truck_model

@log_wrap
def _generate_features(
    codes: list = None,
    n_random: int = None,
    year: int = 2026,
    seed: int = None,
    verbose: bool = False,
    logger=None
) -> pd.DataFrame:
    """
    Generate features for a sequence of circuits.

    Parameters:
        codes (list): List of circuit codes to use as the calendar. If None, random circuits are selected.
        n_random (int): Number of random circuits to select if codes is None.
        year (int): The year for which to generate features (used to compute 'from_thirty').
        seed (int): Random seed for reproducibility when sampling random circuits.
        verbose (bool): If True, prints information about the process.

    Returns:
        pd.DataFrame: DataFrame containing features for each leg in the calendar.
    """

    travel_logistics_df = TL_DF.copy()
    fone_geography_df = FG_DF.copy()

    if verbose:
        logger.info("Generating features for circuits...")

    if year is not None:
        from_thirty = 2030 - year

    if n_random is not None and n_random > 0 and codes is None:
        if seed is None:
            seed = 42
        calendar = fone_geography_df['code_6'].sample(n_random, random_state=seed).to_list()
        if verbose:
            logger.info(f"Randomly selected {n_random} circuits: {calendar}")
    elif codes is not None:
        calendar = list(codes)
        if verbose:
            logger.info(f"Using provided codes: {calendar}")
    else:
        raise ValueError("Either 'codes' must be provided or 'n_random' must be > 0.")

    legs = [{'from_circuit': calendar[i],
             'to_circuit': calendar[i + 1]} for i in range(len(calendar) - 1)]
    legs = pd.DataFrame(legs)
    legs['codes'] = legs['from_circuit'].astype(str) + '-' + legs['to_circuit'].astype(str)

    features_df = travel_logistics_df[travel_logistics_df['codes'].isin(legs['codes'])].set_index('codes').loc[legs['codes']].reset_index()
    features_df = features_df.rename(columns={"truck_viable": "truck_feasible", "distance_km": "air_distance_km"})
    features_df = features_df.drop(columns=['needs_air', 'transport_mode', 'effort_score', "air_emissions", "truck_emissions"])

    features_df['from_thirty'] = from_thirty
    
    features_df = features_df.merge(
        fone_geography_df[['id','code_6', 'latitude', 'longitude']],
        left_on='from_id',
        right_on='id',
        how='left',
        suffixes=('', '_from')
    ).rename(columns={'code_6': 'from_code', 'latitude': 'from_latitude', 'longitude': 'from_longitude'})
    features_df = features_df.merge(
        fone_geography_df[['id','code_6', 'latitude', 'longitude']],
        left_on='to_id',
        right_on='id',
        how='left',
        suffixes=('', '_to')
    ).rename(columns={'code_6': 'to_code', 'latitude': 'to_latitude', 'longitude': 'to_longitude'})

    features_df = features_df.drop(columns=['from_id', 'to_id', 'id_from', 'id_to','from_code', 'to_code'])
    if codes is not None:
        features_df['n_circuits'] = len(codes)
    else:
        features_df['n_circuits'] = n_random

    return features_df

@log_wrap
def fetch_training_data(verbose=False, logger=None) -> pd.DataFrame:
    """
    Fetches and prepares training data for regression models by merging calendar and geography data,
    generating features, and splitting the dataset into air and truck-feasible subsets.
    The function filters calendar data for entries with positive emissions, merges with geography data,
    generates features for each year, and concatenates the results. It then splits the final dataset
    into two sets: one for air routes (where 'truck_feasible' is False) and one for truck-feasible routes
    (where 'truck_feasible' is True).
    Parameters:
        verbose (bool, optional): If True, logs detailed progress information using the provided logger.
        logger (logging.Logger, optional): Logger instance to use for logging progress if verbose is True.
    Returns:
        tuple:
            - training_set_air (pd.DataFrame): Training data for air routes (truck_feasible == False).
            - training_set_truck (pd.DataFrame): Training data for truck-feasible routes (truck_feasible == True).
    """

    fone_calendar_df = FC_DF.copy()
    fone_geography_df = FG_DF.copy()

    fone_calendar_df = fone_calendar_df[fone_calendar_df['leg_emissions'] > 0]

    training_data = pd.DataFrame()
    for year in fone_calendar_df['year'].unique():
        year_data = fone_calendar_df[fone_calendar_df['year'] == year]
        if len(year_data) > 0:
            codes = year_data.merge(fone_geography_df, how='left', left_on='geo_id', right_on='id')[['code_6','leg_emissions','outbound_route']]
            if verbose:
                logger.info(f"Year {year}: {len(codes)} codes")
            features = _generate_features(codes=codes['code_6'].tolist(), year=year, verbose=verbose)
            features = features.merge(codes, how='left', left_on='id', right_on='outbound_route')
            
            training_data = pd.concat([training_data, features], ignore_index=True)

    training_data = training_data.drop(columns=['id', 'outbound_route', 'code_6'])
    if verbose:
        logger.info(f"Total training samples: {len(training_data)}")
    
    training_set_air = training_data[training_data['truck_feasible'] == False].copy()
    training_set_truck = training_data[training_data['truck_feasible'] == True].copy()

    return training_set_air, training_set_truck


#### TRAIN DATA

@log_wrap
def train_regression_model(training_set=None, verbose=False, logger=None):
    """
    Train a linear regression model on the provided training data.

    This function selects relevant features, scales numerical columns, splits the data into
    training, validation, and test sets, fits a linear regression model, evaluates its performance,
    and plots feature importances, predicted vs actual values, and residuals.

    Parameters:
        training_set (pd.DataFrame): The dataset to train the model on. Must contain either
            'air_distance_km' or 'truck_distance_km' columns, and 'leg_emissions' as the target.
        verbose (bool, optional): If True, prints/logs detailed progress information.
        logger (logging.Logger, optional): Logger instance to use for logging if verbose is True.

    Returns:
        tuple:
            - lr_model (LinearRegression): The trained linear regression model.
            - feature_names (list): List of feature names used in the model.
            - feature_importance (np.ndarray): Absolute values of model coefficients.
            - performance (dict): Dictionary of performance metrics (MSE, R2, MAE).
            - feature_explanation (dict): Explanation of each feature.
    """
    if training_set is None:
        raise ValueError("Training set must be provided.")

    if verbose and logger:
        logger.info("Starting regression model training...")

    # Select features for the regression
    if 'air_distance_km' not in training_set.columns and 'truck_distance_km' not in training_set.columns:
        raise ValueError("Training set must contain 'air_distance_km' or 'truck_distance_km' columns.")
    if 'air_distance_km' in training_set.columns:
        selected_features = training_set[['from_circuit', 'to_circuit', 'air_distance_km', 'from_thirty','n_circuits']].copy()
        numerical_columns = ['air_distance_km', 'from_thirty', 'n_circuits']
    else:
        selected_features = training_set[['from_circuit', 'to_circuit', 'truck_distance_km', 'from_thirty','n_circuits']].copy()
        numerical_columns = ['truck_distance_km', 'from_thirty', 'n_circuits']

    selected_features = selected_features.fillna(0)

    # scale numerical features
    scaler = StandardScaler()
    selected_features[numerical_columns] = scaler.fit_transform(
        selected_features[numerical_columns]
    )

    # Assign feature and target variables
    X = selected_features.drop(['from_circuit', 'to_circuit'], axis='columns')
    y = training_set['leg_emissions'].values.flatten()

    # Split into test and remaining set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Split into train and val set
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2 of total
    )

    if verbose:
        msg = (
            f"Data split: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test samples."
        )
        if logger:
            logger.info(msg)
        else:
            print(msg)

    # Train the linear model
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    if verbose:
        msg = "Linear regression model trained."
        if logger:
            logger.info(msg)
        else:
            print(msg)

    # Run predictions on the model
    y_train_pred = lr_model.predict(X_train)
    y_val_pred = lr_model.predict(X_val)
    y_test_pred = lr_model.predict(X_test)

    # Evaluate the regression model
    train_mse = mean_squared_error(y_train, y_train_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    if verbose:
        msg = (
            f"Training MSE: {train_mse:.2f}\n"
            f"Validation MSE: {val_mse:.2f}\n"
            f"Testing MSE: {test_mse:.2f}\n"
            f"Training R²: {train_r2:.3f}\n"
            f"Validation R²: {val_r2:.2f}\n"
            f"Testing R²: {test_r2:.2f}\n"
            f"Testing MAE: {test_mae:.2f}"
        )
        if logger:
            logger.info(msg)
        else:
            print(msg)

    # Feature importance for linear regression is the absolute value of coefficients
    feature_names = X.columns.tolist()
    feature_importance = np.abs(lr_model.coef_)

    # Plotting: Feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, feature_importance)
    plt.xlabel('Absolute Coefficient Value')
    plt.title('Feature Importances (Linear Regression)')
    plt.tight_layout()
    plt.show()

    # Plotting: Predicted vs Actual for test set
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, y_test_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Emissions')
    plt.ylabel('Predicted Emissions')
    plt.title('Actual vs Predicted Emissions (Test Set)')
    plt.tight_layout()
    plt.show()

    # Plotting: Residuals
    residuals = y_test - y_test_pred
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=30, edgecolor='k')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution (Test Set)')
    plt.tight_layout()
    plt.show()

    # Explanation of features
    feature_explanation = {
        'air_distance_km': 'Great-circle distance between circuits (km)',
        'truck_distance_km': 'Truck route distance between circuits (km)',
        'truck_feasible': 'Whether truck transport is feasible (1=yes, 0=no)',
        'from_thirty': 'years until 2030 from the simulated year',
        'from_latitude': 'Latitude of from_circuit',
        'from_longitude': 'Longitude of from_circuit',
        'to_latitude': 'Latitude of to_circuit',
        'to_longitude': 'Longitude of to_circuit',
        'delta_latitude': 'Difference in latitude (to - from)',
        'delta_longitude': 'Difference in longitude (to - from)'
    }

    # Return all relevant outputs
    performance = {
        'train_mse': train_mse,
        'val_mse': val_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'val_r2': val_r2,
        'test_r2': test_r2,
        'test_mae': test_mae
    }

    if verbose and logger:
        logger.info("Regression model training complete.")

    return lr_model, feature_names, feature_importance, performance, feature_explanation


@log_wrap
def calendar_emissions(
    code_seq: list,
    season_year: int,
    air_model,
    truck_model,
    verbose: bool = False,
    logger=None
):
    """
    Predicts leg emissions for a sequence of circuits in a given season using trained regression models.

    This function generates features for the provided circuit sequence and year, applies the appropriate
    regression model (air or truck) to each leg based on feasibility, and returns the predicted emissions
    for each leg.

    Parameters:
        code_seq (list): List of circuit codes (at least two) representing the calendar sequence.
        season_year (int): The year for which to generate features and predictions.
        air_model: Trained regression model for air-feasible legs.
        truck_model: Trained regression model for truck-feasible legs.
        verbose (bool, optional): If True, prints/logs detailed progress information.
        logger (logging.Logger, optional): Logger instance to use for logging if verbose is True.

    Returns:
        tuple:
            - predictions_air (np.ndarray): Predicted emissions for air-feasible legs.
            - predictions_truck (np.ndarray): Predicted emissions for truck-feasible legs.
    """
    if air_model is None or truck_model is None:
        raise ValueError("Air and Truck models must be provided for prediction.")

    if not isinstance(code_seq, list) or len(code_seq) < 2:
        raise ValueError("Input must be a list of at least two circuit codes.")

    if not isinstance(season_year, int):
        raise ValueError("Season year must be an integer.")

    if verbose:
        msg = (
            f"Generating features for {len(code_seq)} circuits for year {season_year}."
        )
        if logger:
            logger.info(msg)
        else:
            print(msg)

    # Generate features for the sequence of circuits
    features = _generate_features(codes=code_seq, year=season_year, verbose=verbose)

    if verbose:
        msg = (
            f"Generated features for {len(features)} legs. "
            f"Splitting into air-feasible and truck-feasible subsets."
        )
        if logger:
            logger.info(msg)
        else:
            print(msg)

    # Split features by feasibility
    features_air = features[features['truck_feasible'] == False].copy()
    features_truck = features[features['truck_feasible'] == True].copy()

    # Drop unused columns for each model
    features_air = features_air.drop(
        columns=['from_latitude', 'from_longitude', 'to_latitude', 'to_longitude', 'truck_feasible', 'truck_distance_km'],
        errors='ignore'
    )
    features_truck = features_truck.drop(
        columns=['air_distance_km', 'truck_feasible', 'from_latitude', 'from_longitude', 'to_latitude', 'to_longitude'],
        errors='ignore'
    )

    # Select features for the regression
    if 'air_distance_km' not in features.columns and 'truck_distance_km' not in features.columns:
        raise ValueError("Features must contain 'air_distance_km' or 'truck_distance_km' columns.")

    selected_features_air = features_air[['from_circuit', 'to_circuit', 'air_distance_km', 'from_thirty', 'n_circuits']].copy()
    numerical_columns_air = ['air_distance_km', 'from_thirty', 'n_circuits']

    selected_features_truck = features_truck[['from_circuit', 'to_circuit', 'truck_distance_km', 'from_thirty', 'n_circuits']].copy()
    numerical_columns_truck = ['truck_distance_km', 'from_thirty', 'n_circuits']

    selected_features_air = selected_features_air.fillna(0)
    selected_features_truck = selected_features_truck.fillna(0)

    # Scale numerical features (note: this uses a new scaler, which may not match training-time scaling)
    scaler_air = StandardScaler()
    scaler_truck = StandardScaler()
    selected_features_air[numerical_columns_air] = scaler_air.fit_transform(selected_features_air[numerical_columns_air])
    selected_features_truck[numerical_columns_truck] = scaler_truck.fit_transform(selected_features_truck[numerical_columns_truck])

    # Drop non-numeric columns for prediction
    x_air = selected_features_air.drop(['from_circuit', 'to_circuit'], axis='columns')
    x_truck = selected_features_truck.drop(['from_circuit', 'to_circuit'], axis='columns')

    if verbose:
        msg = (
            f"Predicting emissions for {len(x_air)} air-feasible legs and {len(x_truck)} truck-feasible legs."
        )
        if logger:
            logger.info(msg)
        else:
            print(msg)

    # Make predictions
    predictions_air = air_model.predict(x_air) if len(x_air) > 0 else np.array([])
    predictions_truck = truck_model.predict(x_truck) if len(x_truck) > 0 else np.array([])

    if verbose:
        msg = (
            f"Prediction complete. Air: {sum(predictions_air)}, "
            f"Truck: {sum(predictions_truck)}."
        )
        if logger:
            logger.info(msg)
        else:
            print(msg)

    return predictions_air, predictions_truck