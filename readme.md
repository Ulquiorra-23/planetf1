# F1 Green Flag - A Sustainable Data Science Solution for F1 Calendar Management

## Description

This project analyzes Formula 1 logistics data to develop actionable recommendations for reducing the sport's environmental impact. By leveraging data science techniques, including clustering, regression analysis, and genetic algorithms, we aim to optimize the F1 race calendar, to minimize travel-related emissions for cars and equipments.

The core of the optimization uses a Genetic Algorithm (built with the DEAP library) to find near-optimal sequences of race locations, minimizing a fitness function based on travel distance or estimated CO2 Emsissions between consecutive events, while potentially considering other logistical constraints. The Genetic Algorithm fitness function relies on a Clustering K-Means model that find clusters for the set of circuits provided, and on a Regression model that estimates the emissions generated per leg.

## Features

* **Data Collection & Preparation:** Scripts and notebooks to gather F1 circuit, geographical, and logistical data, culminating in the `planet_fone.db` database.
* **Data Analysis:** Jupyter notebooks exploring the data using techniques like clustering (`clustering.py`) and regression (`regression.py`).
* **Genetic Algorithm Optimization:** Implementation of a custom Genetic Algorithm (`run_ga.py`, `genetic_ops.py`) using DEAP to find optimized race sequences based on minimizing travel distances or emissions if regression is enabled.
* **Modular Structure:** Code organized into utilities (`utils/`), models/operators (`models/`), and analysis notebooks.

## Analysis

* **Data Collection:**  `data_collection.ipynb` to gather initial data.
* **Data Preparation:**  `data_prep.ipynb` to clean and prepare data for analysis and modeling.
* **Clustering Analysis:** Explore `clustering.ipynb`.
* **Regression Analysis:** Explore `regression_eng.ipynb`.
* **Genetic Algorithm:**
    * `genetic.ipynb`: Contains development, testing, and tuning of the GA components.
    * `genetic_exe.ipynb`: Executes the tuned Genetic Algorithm to find optimized calendars.

## App

**Greenflag Streamlit App**
* Use this tool via the ([App](https://greenflag.streamlit.app/))

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **Install dependencies using Poetry:**
    Ensure you have [Poetry installed](https://python-poetry.org/docs/#installation). Then, navigate to the project directory and run:
    ```bash
    poetry install
    ```
    This will create a virtual environment for the project and install all necessary packages as defined in the `pyproject.toml` and `poetry.lock` files.

## Usage

Once the setup is complete, you can run the project scripts using `poetry run`. This command ensures that your script executes within the correct Poetry-managed virtual environment, using all the specified dependencies.

1.  **Run Historical Season:**
    * To optimize an historical calendar for a specific season (available from 2000 to 2025):
        ```bash
        poetry run python run_ga.py 1 <SeasonYear>
        ```
        *Example:*
        ```bash
        poetry run python run_ga.py 1 2023
        ```

2.  **Run Random Calendar:**
    * To optimize a random list of circuits based on a specified sample size:
        ```bash
        poetry run python run_ga.py 2 <SampleSize>
        ```
        *Example:*
        ```bash
        poetry run python run_ga.py 2 20
        ```

3.  **Run Custom Calendar:**
    * To optimize a custom list of circuits using their IDs (provide a comma-separated list of at least 15 IDs, e.g., `1,2,3,...,n`):
        ```bash
        poetry run python run_ga.py 3 <IDlist>
        ```
        *Example:*
        ```bash
        poetry run python run_ga.py 3 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
        ```

**Alternative: Using Poetry Shell**

If you prefer not to type `poetry run` before each command, you can activate the project's virtual environment shell once. In your terminal, navigate to the project directory and run:

```bash
poetry shell
```

## File Structure

```text
📦 
├─ .gitignore
├─ data
│  ├─ Costraints.xlsx
│  ├─ __init__.py
│  ├─ app_data.py
│  ├─ calendar.csv
│  ├─ fone_calendar.csv
│  ├─ fone_geography.csv
│  ├─ planet_fone.db
│  ├─ sqlite_sequence.csv
│  ├─ training_regression_calendar.csv
│  └─ travel_logistic.csv
├─ log
│  ├─ GOLD_20250525_173219.json
│  ├─ GOLD_EXP_20250525_173219.json
│  └─ GOLD_NC_20250525_173219.jso
├─ media
│  └─ app*.jpg
├─ models
│  ├─ __init__.py
│  ├─ clustering.py
│  ├─ fone_regression_air_model.pkl
│  ├─ fone_regression_truck_model.pkl
│  ├─ genetic_ops.py
│  └─ regression.py
├─ notebooks
│  ├─ clustering.ipynb
│  ├─ data_collection.ipynb
│  ├─ data_collection_mad.ipynb
│  ├─ data_prep.ipynb
│  ├─ fastf1_cache
│  │  └─ fastf1_http_cache.sqlite
│  ├─ genetic.ipynb
│  ├─ genetic_exe.ipynb
│  ├─ image.png
│  ├─ logs_debugging.ipynb
│  ├─ regression_eng.ipynb
│  └─ visuals_presentation.ipynb
├─ readme.md
├─ requirements.txt
├─ run_ga.py
├─ src
│  ├─ __init__.py
│  ├─ app.py
│  └─ math.ipynb
└─ utils
   ├─ __init__.py
   ├─ logging_test.ipynb
   ├─ logs.py
   ├─ sql.py
   └─ utilities.py
```
## Data

The primary data source is the SQLite database `planet_fone.db`, which is created and populated by the `data_collection.ipynb` and `data_prep.ipynb` notebooks. Additional raw or intermediate data files, as well as potential constraint definitions, are located in the `data/` directory.

## Contributing

This project is developed by:

* Jakob Jonas Spranger ([jakobjspranger](https://github.com/jakobjspranger))
* Juan Jose Montesinos ([juanjomontesinos](https://github.com/juanjomontesinos))
* Maximilian von Braun ([MaxBraun-EAE](https://github.com/MaxBraun-EAE))
* Massimiliano Napolitano ([Ulquiorra-23](https://github.com/Ulquiorra-23))
