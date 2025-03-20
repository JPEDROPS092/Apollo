# Genetic Syndrome Classification from Image Embeddings

This repository contains the solution for the Apollo Solutions Machine Learning Developer Test, focusing on the classification of genetic syndromes based on image embeddings. The project includes data processing, visualization, classification using KNN with both Cosine and Euclidean distances, manual implementation of key metrics (AUC, F1-score, Top-k accuracy), and a comprehensive report.

## Project Structure

The repository is organized as follows:

- `main.py`: The main script to run the entire pipeline.
- `reporting.py`:  Handles the generation of performance tables and summary statistics.
- `requirements.txt`: Lists the required Python packages.
- `README.md`: This file, providing an overview of the project.
- `results/`:  Directory containing generated plots and tables (t-SNE visualizations, ROC curves, performance tables, etc.).  This is populated after running `main.py`.
- `src/`: Directory containing the core modules:
    - `data_processing.py`:  Loads, preprocesses, and flattens the hierarchical data from the pickle file.  Provides functions for data exploration.
    - `classification.py`: Implements the KNN classifier with Cosine and Euclidean distance metrics, 10-fold cross-validation, and optimal k selection.
    - `metrics.py`:  Contains functions to manually calculate AUC, F1-score, and Top-k accuracy.
    - `visualization.py`:  Implements t-SNE dimensionality reduction and generates plots for visualization and analysis.
    - `utils.py`: Contains helper functions used across the project, including distance calculations.
    - `__init__.py`:  An empty file that makes the `src` directory a Python package.
- `processed_data.pkl`: The processed data after running `data_processing.py` in the `main.py` script.
- `uploads/`:  Contains the original `mini_gm_public_v0.1.p` pickle file (not included in the repository due to size, but expected to be placed here).
- `Genetic Syndrome Classification Report - Apollo Solutions.pdf`: The main report summarizing the methodology, results, analysis, challenges, and recommendations.
- `Interpretation.pdf`:  Answers to the interpretation questions provided in the test.
- `ML Junior Practical Test.docx`:  The original test description document (not included in the repository, but mentioned for context).
- `Task.md`: A Markdown version of the task description (likely a copy of the `.docx` file).

## Setup and Execution

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Place the data file:**

    Obtain the `mini_gm_public_v0.1.p` file and place it in the `uploads/` directory.

3.  **Install dependencies:**

    It is highly recommended to create a virtual environment before installing dependencies:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

    Then, install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the main script:**

    ```bash
    python main.py
    ```

    This script will:

    -   Load and preprocess the data.
    -   Perform exploratory data analysis.
    -   Generate t-SNE visualizations.
    -   Train and evaluate KNN models with both distance metrics.
    -   Calculate and report performance metrics (AUC, F1-score, Top-k accuracy).
    -   Generate ROC curves.
    -   Create performance tables.
    -   Save all plots and tables in the `results/` directory.

    The script is designed to run the entire pipeline sequentially.  You may need to adjust file paths within the scripts if you modify the directory structure.

## Key Files and Functionality

-   **`main.py`**: The entry point of the project.  It orchestrates the entire workflow, calling functions from the other modules. It checks if the processed data file exists and loads it if it does, otherwise it runs the data processing steps.

-   **`src/data_processing.py`**:  Handles data loading, cleaning, flattening, and basic statistical analysis. Key functions include loading the pickle file, flattening the hierarchical structure into a format suitable for machine learning (features and labels), and providing statistics like the number of syndromes and images per syndrome.

-   **`src/visualization.py`**:  Performs dimensionality reduction using t-SNE and generates plots for visualization.  This helps visualize the data distribution and identify potential clusters.

-   **`src/classification.py`**: Implements the KNN classifier. It includes functions for:
    -   Calculating Cosine and Euclidean distances.
    -   Performing 10-fold cross-validation.
    -   Finding the optimal *k* value for KNN.
    -   Predicting syndrome labels.

-   **`src/metrics.py`**:  Implements the performance metrics:
    -   `calculate_auc`: Calculates the Area Under the ROC Curve (AUC).
    -   `calculate_f1_score`: Calculates the F1-score.
    -   `calculate_top_k_accuracy`: Calculates the Top-k accuracy.

-   **`src/utils.py`**: Provides utility functions:
    -    `cosine_distance`: Calculates the cosine distance.
    -    `euclidean_distance`: Calculates the euclidean distance.

-   **`reporting.py`**: Generates summary tables and statistics related to the performance and data characteristics.

## Report

The `Genetic Syndrome Classification Report - Apollo Solutions.pdf` file contains a detailed explanation of the project, including:

-   **Methodology:**  A description of the data preprocessing steps, the choice of the KNN algorithm, the distance metrics used, and the cross-validation strategy.
-   **Results:**  Presentation of the findings, including t-SNE plots, ROC curves, and performance tables.
-   **Analysis:**  Interpretation of the results, comparison of the two distance metrics, and discussion of any insights gained.
-   **Challenges and Solutions:**  Description of any difficulties encountered during the project and the approaches taken to address them.
-   **Recommendations:**  Suggestions for potential improvements or further analysis.

## Interpretation Answers

The `Interpretation.pdf` file contains answers to the theoretical questions about data distributions, model selection, and evaluation metrics.

## Important Notes

-   The original pickle file (`mini_gm_public_v0.1.p`) is not included in the repository due to its size. It should be placed in the `uploads/` directory before running the code.
-   The code is written as standalone Python scripts, not as a Jupyter Notebook, as per the instructions.
-   The `results/` directory will be populated with plots and tables after running `main.py`.
-   The `requirements.txt` file lists all necessary Python packages. Make sure to install them before running the code.
