# F1 Green Flag - A Sustainable Data Science Solution for F1 Calendar Management

## Description

This project analyzes Formula 1 logistics data to develop actionable recommendations for reducing the sport's environmental impact. By leveraging data science techniques, including clustering, regression analysis, and genetic algorithms, we aim to optimize the F1 race calendar, specifically targeting the 2026 season, to minimize travel-related emissions.

The core of the optimization uses a Genetic Algorithm (built with the DEAP library) to find near-optimal sequences of race locations, minimizing a fitness function based on travel distance (as a proxy for emissions) between consecutive events, while potentially considering other logistical constraints.

## Project Goal

To provide data-driven insights and a computationally optimized race calendar proposal for the 2026 Formula 1 season, demonstrably reducing the carbon footprint associated with logistics and travel compared to traditional scheduling approaches.

## Features

* **Data Collection & Preparation:** Scripts and notebooks to gather F1 circuit, geographical, and logistical data, culminating in the `planet_fone.db` database.
* **Data Analysis:** Jupyter notebooks exploring the data using techniques like clustering (`clustering.ipynb`) and potentially regression (`training_regression_calendar.csv` suggests this).
* **Genetic Algorithm Optimization:** Implementation of a custom Genetic Algorithm (`genetic.ipynb`, `genetic_ops.py`) using DEAP to find optimized race sequences based on minimizing travel distance/emissions.
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

## Usage

Currently, the project primarily consists of Jupyter notebooks for data processing, analysis, and model development/tuning.

* **Data Collection:** Run `data_collection.ipynb` to gather initial data.
* **Data Preparation:** Run `data_prep.ipynb` to clean and prepare data for analysis and modeling.
* **Clustering Analysis:** Explore `clustering.ipynb`.
* **Genetic Algorithm:**
    * `genetic.ipynb`: Contains development, testing, and tuning of the GA components.
    * `genetic_exe.ipynb`: Executes the tuned Genetic Algorithm to find optimized calendars.

*(Note: A main executable script for running the entire pipeline or specific tasks is under development.)*

## File Structure

```text
./
├── .gitignore                  # Ignored files for git
├── clustering.ipynb            # Clustering module testing and tuning
├── data_collection.ipynb       # For gathering data and creation of planet_fone.db
├── data_prep.ipynb             # Data prep for regression, clustering, and genetic algo
├── genetic.ipynb               # Genetic module building and tuning
├── genetic_exe.ipynb           # Genetic module execution
├── image.png                   # Used in data prep notebook
├── math.ipynb                  # Testing notebook
├── planet_fone.db              # Main datasource SQLite database
├── readme.md                   # This README file
├── requirements.txt            # Python package dependencies
├── training_regression_calendar.csv # Initial dataset possibly for regression
│
├── data/                       # Data files, backups, constraints
│   ├── calendar.csv
│   ├── Costraints.xlsx         # Constraints for scheduling?
│   ├── fone_calendar.csv
│   ├── fone_geography.csv
│   ├── sqlite_sequence.csv
│   └── travel_logistic.csv
│
├── fastf1_cache/               # Cache for the FastF1 library
│   └── fastf1_http_cache.sqlite
│
├── models/                     # Custom model implementations (operators, etc.)
│   ├── clustering.py           # Clustering operators/functions
│   ├── genetic_ops.py          # Genetic Algorithm operators (crossover, mutation, fitness)
│   └── __init__.py
│
└── utils/                      # Utility scripts
    ├── sql.py                  # SQL database utilities
    ├── utilities.py            # Generic utility functions
    └── __init__.py

## Data

The primary data source is the SQLite database `planet_fone.db`, which is created and populated by the `data_collection.ipynb` and `data_prep.ipynb` notebooks. Additional raw or intermediate data files, as well as potential constraint definitions, are located in the `data/` directory.

## Contributing

This project is developed by:

* Jakob Jonas Spranger ([jakobjspranger](https://github.com/jakobjspranger))
* Juan Jose Montesinos ([juanjomontesinos](https://github.com/juanjomontesinos))
* Maximilian von Braun ([MaxBraun-EAE](https://github.com/MaxBraun-EAE))
* Massimiliano Napolitano ([Ulquiorra-23](https://github.com/Ulquiorra-23))

*(Feel free to add contribution guidelines if desired)*

## License

*(Specify your project's license here, e.g., MIT License, Apache 2.0, or leave blank if undecided.)*

