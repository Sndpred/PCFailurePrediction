**Project Overview**

The  objective is to build a predictive model that can identify potential PC failures in a production environment based on historical performance metrics. By analyzing key indicators such as CPU usage, memory consumption, disk I/O, and temperature, the model can provide an early warning system to prevent downtime and data loss.

**Methodology**

The analysis notebook is organized into the following key steps:

  **Data Loading and Initial Exploration:** The process begins with loading the datapc.csv file and performing an initial check on its structure, data types, and missing values.

  **Data Cleaning and Failure Event Detection:** Since the dataset lacks explicit failure labels, a robust strategy is employed to label failure events synthetically. This involves identifying large time gaps in data collection and periods of anomalous system metric behavior.

  **Exploratory Data Analysis (EDA):** This step focuses on understanding the distribution of failure events and identifying potential correlations between system metrics and the likelihood of failure.

  **Feature Engineering:** A comprehensive set of predictive features is created from the raw data. This includes:

        Time-based features (e.g., hour of day, day of week).

        Rolling window features to capture trends and volatility.

        Lag features represent past states of the system.

        Interaction and domain-specific features.

  **Model Development and Training:** Several machine learning models (Random Forest, Gradient Boosting, Logistic Regression, and SVM) are trained and evaluated using different class imbalance handling techniques (SMOTE, Undersampling).

  **Model Evaluation and Optimization:** The best-performing model is identified based on its AUC (Area Under the Curve) score. A detailed evaluation, including a confusion matrix and key metrics like precision and recall, is performed. The prediction threshold is also optimized to improve performance.

**Key Findings**

   **Failure Patterns:** Failures were found to exhibit temporal patterns, occurring more frequently at specific hours and on certain days of the week.

   **Top Predictors:** The most important features for predicting failure were found to be anomaly_count, win_cpu_Percent_Processor_Time, and mem_used_percent.

   **Best Model:** The Random Forest Classifier trained with undersampling achieved the highest performance, with an AUC of 0.946 on the validation set.

**Streamlit Application**

A Streamlit application (streamlit_app.py) is included to demonstrate the model in a user-friendly interface. It allows users to upload new data, run the predictive model, and visualize the results in real time.

**How to Use**

    Clone this repository: git clone [repository_url]

    Install the required libraries:

    pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn

    Ensure your dataset (datapc.csv) is in the same directory as the notebook.

    Open the PCFailureAnalysis.ipynb Notebook in Jupyter and run the cells in sequence to reproduce the analysis.

**Dataset**

The dataset used for this project contains system performance telemetry from multiple PCs, including metrics related to CPU, memory, disk, and temperature.
