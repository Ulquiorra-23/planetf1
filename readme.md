# F1 Green Flag - A Sustainable Data Science Solution for F1 Calendar Management

## Description

This project analyzes Formula 1 logistics data to develop actionable recommendations for reducing the sport's environmental impact. By leveraging data science techniques, including clustering, regression analysis, and genetic algorithms, we aim to optimize the F1 race calendar, specifically targeting the 2026 season, to minimize travel-related emissions.

The core of the optimization uses a Genetic Algorithm (built with the DEAP library) to find near-optimal sequences of race locations, minimizing a fitness function based on travel distance (as a proxy for emissions) between consecutive events, while potentially considering other logistical constraints.

## Project Goal

To provide data-driven insights and a computationally optimized race calendar proposal for the 2026 Formula 1 season, demonstrably reducing the carbon footprint associated with logistics and travel compared to traditional scheduling approaches.

## Features

* **Data Collection & Preparation:** Scripts and notebooks to gather F1 circuit, geographical, and logistical data, culminating in the `planet_fone.db` database.
* **Data Analysis:** Jupyter notebooks exploring the data using techniques like clustering (`clustering.py`) and regression **(WIP)**.
* **Genetic Algorithm Optimization:** Implementation of a custom Genetic Algorithm (`run_ga.py`, `genetic_ops.py`) using DEAP to find optimized race sequences based on minimizing travel distances or emissions if regression is enabled.
* **Modular Structure:** Code organized into utilities (`utils/`), models/operators (`models/`), and analysis notebooks.

## Installation

1.  **Prerequisites:**
    * Python 3.12.5 or later.
    * `pip` (Python package installer).

2.  **Clone the repository (if applicable):**
    ```bash
    git clone <your-repository-url>
    cd <your-project-directory>
    ```

3.  **Install Dependencies:**
    Install the required Python packages using the provided `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```


## Analysis

* **Data Collection:**  `data_collection.ipynb` to gather initial data.
* **Data Preparation:**  `data_prep.ipynb` to clean and prepare data for analysis and modeling.
* **Clustering Analysis:** Explore `clustering.ipynb`.
* **Genetic Algorithm:**
    * `genetic.ipynb`: Contains development, testing, and tuning of the GA components.
    * `genetic_exe.ipynb`: Executes the tuned Genetic Algorithm to find optimized calendars.

## Usage

1.  **Run Historical Season:**
    * run `run_ga.py` with arguments " 1 <SeasonYear>" *available from 2000 to 2025*
    * to optimize an historical calendar

2.  **Run Random Calendar**
    * run `run_ga.py` with arguments " 2 <SampleSize>" 
    * to optimize a random list of circuits

3.  **Run Custom Calendar:**
    * run `run_ga.py` with arguments " 3 <IDlist>" where IDlist has min 15 ids *i.e. 1,2,3,...,n*
    * to optimize a custom list of circuits

## File Structure

```text
./
├── .gitignore
├── genetic_exe.ipynb           # Genetic module execution (alternative to script)
├── readme.md                   # This README file
├── requirements.txt            # Python package dependencies
├── run_ga.py                   # Main script to execute the Genetic Algorithm
│
├── data/                       # Data files, backups, constraints
│   ├── calendar.csv
│   ├── Costraints.xlsx         # Constraints for scheduling
│   ├── fone_calendar.csv
│   ├── fone_geography.csv
│   ├── planet_fone.db          # main db
│   ├── sqlite_sequence.csv
│   ├── training_regression_calendar.csv # Dataset possibly for regression
│   └── travel_logistic.csv
│
├── models/                     # Custom model implementations (operators, etc.)
│   ├── clustering.py           # Clustering operators/functions
│   ├── genetic_ops.py          # Genetic Algorithm operators (crossover, mutation, fitness)
│   ├── __init__.py
│   └── __pycache__/            # Compiled Python files (auto-generated)
│       ├── clustering.cpython-312.pyc
│       ├── genetic_ops.cpython-312.pyc
│       └── __init__.cpython-312.pyc
│
├── notebooks/                  # Jupyter notebooks for analysis and development
│   ├── clustering.ipynb        # Clustering module testing and tuning
│   ├── genetic_exe.ipynb       # Genetic module execution (alternative to script)
│   ├── data_collection.ipynb   # For gathering data and creation of planet_fone.db
│   ├── data_prep.ipynb         # Data prep for regression, clustering, and genetic algo
│   ├── genetic.ipynb           # Genetic module building and tuning
│   ├── image.png               # Image used in a notebook
│   ├── math.ipynb              # Testing notebook
│   └── fastf1_cache/           # Cache for the FastF1 library
│       └── fastf1_http_cache.sqlite
│
└── utils/                      # Utility scripts
    ├── sql.py                  # SQL database utilities
    ├── utilities.py            # Generic utility functions
    ├── __init__.py
    └── __pycache__/            # Compiled Python files (auto-generated)
        ├── sql.cpython-312.pyc
        ├── utilities.cpython-312.pyc
        └── __init__.cpython-312.pyc
```
## Data

The primary data source is the SQLite database `planet_fone.db`, which is created and populated by the `data_collection.ipynb` and `data_prep.ipynb` notebooks. Additional raw or intermediate data files, as well as potential constraint definitions, are located in the `data/` directory.

## Contributing

This project is developed by:

* Jakob Jonas Spranger ([jakobjspranger](https://github.com/jakobjspranger))
* Juan Jose Montesinos ([juanjomontesinos](https://github.com/juanjomontesinos))
* Maximilian von Braun ([MaxBraun-EAE](https://github.com/MaxBraun-EAE))
* Massimiliano Napolitano ([Ulquiorra-23](https://github.com/Ulquiorra-23))
