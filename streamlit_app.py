import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, RocCurveDisplay
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up the Streamlit app layout
st.set_page_config(layout="wide", page_title="PC Failure Analysis Dashboard")

st.title("PC Failure Analysis Dashboard")
st.write("This application analyzes system metrics to predict PC failures, based on a Jupyter Notebook workflow.")
st.write("You can either use the pre-loaded `datapc.csv` or upload your own CSV file.")

# Caching data loading to improve performance for the default file
@st.cache_data
def load_default_data():
    """Loads the default dataset."""
    try:
        df = pd.read_csv('datapc.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    except FileNotFoundError:
        return None

# Helper function to clean data values
def clean_data_values(X_array):
    """Replace inf with NaN, then fill NaN with column median."""
    X_cleaned = np.where(np.isinf(X_array), np.nan, X_array)
    for col in range(X_cleaned.shape[1]):
        col_data = X_cleaned[:, col]
        median_val = np.nanmedian(col_data)
        mask = np.isnan(col_data)
        if median_val is not None and not np.isnan(median_val):
            X_cleaned[mask, col] = median_val
        else:
            X_cleaned[mask, col] = 0 # Fallback to 0 if median is also NaN
    return X_cleaned

# Caching the full analysis pipeline
@st.cache_data
def run_full_pipeline(df_input):
    """Runs the full data cleaning, feature engineering, and model training pipeline."""
    # ------------------------------------------------------------------
    # Step 2: Data Cleaning and Failure Event Detection
    # ------------------------------------------------------------------
    df_clean = df_input.copy()
    high_missing_cols = ['disk_total', 'disk_used', 'disk_used_percent']
    
    # Check if columns exist before dropping
    df_clean = df_clean.drop(columns=[col for col in high_missing_cols if col in df_clean.columns])
    
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')

    df_clean['time_diff'] = df_clean['timestamp'].diff()
    df_clean['time_diff_seconds'] = df_clean['time_diff'].dt.total_seconds()
    
    normal_interval = 60
    gap_threshold = normal_interval * 5
    df_clean['large_gap'] = df_clean['time_diff_seconds'] > gap_threshold
    
    cpu_threshold = df_clean['win_cpu_Percent_Processor_Time'].quantile(0.95)
    mem_threshold = df_clean['mem_used_percent'].quantile(0.95)
    queue_threshold = df_clean['win_system_Processor_Queue_Length'].quantile(0.95)
    temp_threshold = df_clean['smart_device_temp_c'].quantile(0.95)
    
    df_clean['cpu_spike'] = df_clean['win_cpu_Percent_Processor_Time'] > cpu_threshold
    df_clean['memory_spike'] = df_clean['mem_used_percent'] > mem_threshold
    df_clean['queue_spike'] = df_clean['win_system_Processor_Queue_Length'] > queue_threshold
    df_clean['temp_spike'] = df_clean['smart_device_temp_c'] > temp_threshold
    
    df_clean['failure_next'] = df_clean['large_gap'].shift(-1).fillna(False)
    
    anomaly_cols = ['cpu_spike', 'memory_spike', 'queue_spike', 'temp_spike']
    df_clean['anomaly_count'] = df_clean[anomaly_cols].sum(axis=1)
    df_clean['multiple_anomalies'] = df_clean['anomaly_count'] >= 2
    
    df_clean['failure_event'] = (df_clean['failure_next'] | df_clean['multiple_anomalies'])
    
    prediction_window_minutes = 10
    df_clean['failure_in_10min'] = df_clean['failure_event'].shift(-int(prediction_window_minutes)).fillna(False).astype(int)
    
    # ------------------------------------------------------------------
    # Step 3: Feature Engineering
    # ------------------------------------------------------------------
    df_features = df_clean.copy()
    
    df_features['hour'] = df_features['timestamp'].dt.hour
    df_features['day_of_week'] = df_features['timestamp'].dt.dayofweek
    df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)
    
    df_features['peak_failure_hour'] = ((df_features['hour'] == 6) | (df_features['hour'] == 12)).astype(int)
    df_features['high_risk_day'] = ((df_features['day_of_week'] == 1) | (df_features['day_of_week'] == 4)).astype(int)
    
    df_features['minutes_since_start'] = (df_features['timestamp'] - df_features['timestamp'].min()).dt.total_seconds() / 60
    
    key_metrics = [
        'win_cpu_Percent_Processor_Time', 'mem_used_percent', 
        'win_system_Processor_Queue_Length', 'smart_device_temp_c',
        'win_disk_Percent_Disk_Read_Time'
    ]
    
    df_features = df_features.sort_values('timestamp').reset_index(drop=True)
    
    for metric in key_metrics:
        df_features[f'{metric}_5min_avg'] = df_features[metric].rolling(window=5, min_periods=1).mean()
        df_features[f'{metric}_15min_avg'] = df_features[metric].rolling(window=15, min_periods=1).mean()
        df_features[f'{metric}_5min_std'] = df_features[metric].rolling(window=5, min_periods=1).std()
        df_features[f'{metric}_trend_5min'] = df_features[metric] - df_features[f'{metric}_5min_avg']
        df_features[f'{metric}_trend_15min'] = df_features[metric] - df_features[f'{metric}_15min_avg']
        df_features[f'{metric}_rate_change'] = df_features[metric].pct_change(periods=5).fillna(0)
    
    important_metrics = ['anomaly_count', 'win_cpu_Percent_Processor_Time', 'win_system_Processor_Queue_Length']
    
    for metric in important_metrics:
        df_features[f'{metric}_lag_1'] = df_features[metric].shift(1)
        df_features[f'{metric}_lag_5'] = df_features[metric].shift(5)
        df_features[f'{metric}_lag_10'] = df_features[metric].shift(10)
        for lag in [1, 5, 10]:
            df_features[f'{metric}_lag_{lag}'] = df_features[f'{metric}_lag_{lag}'].fillna(method='bfill')
    
    df_features['cpu_memory_stress'] = df_features['win_cpu_Percent_Processor_Time'] * df_features['mem_used_percent'] / 100
    df_features['system_overload'] = df_features['win_system_Processor_Queue_Length'] * df_features['win_cpu_Percent_Processor_Time']
    df_features['thermal_cpu_stress'] = df_features['smart_device_temp_c'] * df_features['win_cpu_Percent_Processor_Time']
    df_features['disk_io_total'] = df_features['win_disk_Percent_Disk_Read_Time'] + df_features['win_disk_Percent_Disk_Write_Time']
    df_features['resource_ratio'] = (df_features['win_cpu_Percent_Processor_Time'] + df_features['mem_used_percent']) / 2
    
    for metric in key_metrics:
        df_features[f'{metric}_10min_min'] = df_features[metric].rolling(window=10, min_periods=1).min()
        df_features[f'{metric}_10min_max'] = df_features[metric].rolling(window=10, min_periods=1).max()
        df_features[f'{metric}_10min_range'] = df_features[f'{metric}_10min_max'] - df_features[f'{metric}_10min_min']
        df_features[f'{metric}_10min_p75'] = df_features[metric].rolling(window=10, min_periods=1).quantile(0.75)
        df_features[f'{metric}_10min_p25'] = df_features[metric].rolling(window=10, min_periods=1).quantile(0.25)
    
    for metric in key_metrics:
        rolling_mean = df_features[metric].rolling(window=30, min_periods=1).mean()
        rolling_std = df_features[metric].rolling(window=30, min_periods=1).std()
        df_features[f'{metric}_zscore'] = (df_features[metric] - rolling_mean) / (rolling_std + 1e-8)
        df_features[f'{metric}_is_anomaly'] = (abs(df_features[f'{metric}_zscore']) > 2).astype(int)

    anomaly_cols = [col for col in df_features.columns if col.endswith('_is_anomaly')]
    df_features['recent_anomalies_5min'] = df_features[anomaly_cols].rolling(window=5, min_periods=1).sum().sum(axis=1)
    df_features['recent_anomalies_15min'] = df_features[anomaly_cols].rolling(window=15, min_periods=1).sum().sum(axis=1)
    
    df_features['system_stability'] = 1 / (1 + df_features['win_system_Processor_Queue_Length'] + df_features['anomaly_count'])
    baseline_cpu = df_features['win_cpu_Percent_Processor_Time'].quantile(0.1)
    df_features['performance_degradation'] = np.maximum(0, df_features['win_cpu_Percent_Processor_Time'] - baseline_cpu)
    
    cpu_critical = df_features['win_cpu_Percent_Processor_Time'].quantile(0.90)
    memory_critical = df_features['mem_used_percent'].quantile(0.90)
    temp_critical = df_features['smart_device_temp_c'].quantile(0.90)
    
    df_features['cpu_critical'] = (df_features['win_cpu_Percent_Processor_Time'] > cpu_critical).astype(int)
    df_features['memory_critical'] = (df_features['mem_used_percent'] > memory_critical).astype(int)
    df_features['temp_critical'] = (df_features['smart_device_temp_c'] > temp_critical).astype(int)
    df_features['critical_count'] = df_features['cpu_critical'] + df_features['memory_critical'] + df_features['temp_critical']
    
    # Fill remaining missing values created by feature engineering
    for col in df_features.columns:
        if df_features[col].isnull().any():
            df_features[col] = df_features[col].fillna(df_features[col].median() if df_features[col].dtype in ['float64', 'int64'] else 0)

    # ------------------------------------------------------------------
    # Step 4: Model Training
    # ------------------------------------------------------------------
    exclude_cols = [
        'timestamp', 'time_diff', 'time_diff_seconds', 'large_gap', 
        'cpu_spike', 'memory_spike', 'queue_spike', 'temp_spike',
        'failure_next', 'failure_event', 'multiple_anomalies'
    ]
    
    feature_cols = [col for col in df_features.columns 
                    if col not in exclude_cols and not col.endswith('_is_anomaly') 
                    and col != 'failure_in_10min' and df_features[col].dtype in ['float64', 'int64']]
    
    X = df_features[feature_cols]
    y = df_features['failure_in_10min']

    # Chronological split
    n_samples = len(X)
    train_end = int(0.6 * n_samples)
    val_end = int(0.8 * n_samples)
    
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]
    
    # Clean inf values before scaling
    X_train_cleaned = clean_data_values(X_train.values)
    X_val_cleaned = clean_data_values(X_val.values)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_cleaned)
    X_val_scaled = scaler.transform(X_val_cleaned)
    
    undersampler = RandomUnderSampler(random_state=42)
    X_under, y_under = undersampler.fit_resample(X_train_scaled, y_train)
    
    best_model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    best_model.fit(X_under, y_under)
    
    y_pred_proba = best_model.predict_proba(X_val_scaled)[:, 1]
    
    precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    # Get feature importances for the best model
    importances = best_model.feature_importances_
    importance_df = pd.DataFrame({'feature': feature_cols, 'importance': importances}).sort_values('importance', ascending=False)
    
    # Attach scaler to the model and store all feature names for later use
    best_model.scaler_ = scaler
    best_model.feature_names_ = feature_cols

    return df_clean, best_model, X_val, y_val, X_val_scaled, y_pred_proba, importance_df, optimal_threshold

# ---
# Main App Logic
# ---

uploaded_file = st.file_uploader("Upload your own PC metrics CSV file", type=['csv'])

if uploaded_file is not None:
    # Read uploaded file
    try:
        df = pd.read_csv(uploaded_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        st.success("File uploaded and processed successfully!")
    except Exception as e:
        st.error(f"Error loading file: {e}")
        df = None
else:
    # Use default data if no file is uploaded
    df = load_default_data()
    if df is None:
        st.warning("Please upload a CSV file to begin, or ensure 'datapc.csv' is in the app directory.")


if df is not None:
    # Run the full pipeline with the selected data
    df_clean, best_model, X_val, y_val, X_val_scaled, y_pred_proba, importance_df, optimal_threshold = run_full_pipeline(df)

    # ---
    ## 1. Data Overview
    st.dataframe(df.head())
    st.write(f"**Dataset Shape:** `{df.shape}`")
    st.write(df.describe())

    # ---
    ## 2. Failure Pattern Analysis
    st.write("Visualizations from the notebook analysis show key failure patterns over time.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Failures by Hour of Day")
        failure_records = df_clean[df_clean['failure_event']]
        if not failure_records.empty:
            hourly_failures = failure_records.groupby(failure_records['timestamp'].dt.hour).size()
            fig_hourly, ax_hourly = plt.subplots()
            ax_hourly.bar(hourly_failures.index, hourly_failures.values, color='lightcoral', alpha=0.7)
            ax_hourly.set_xlabel('Hour')
            ax_hourly.set_ylabel('Number of Failures')
            ax_hourly.set_xticks(hourly_failures.index)
            st.pyplot(fig_hourly)
        else:
            st.info("No failure events detected in the data to plot.")
    
    with col2:
        st.subheader("Failures by Day of Week")
        if not failure_records.empty:
            daily_failures = failure_records.groupby(failure_records['timestamp'].dt.day_name()).size()
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_failures = daily_failures.reindex(day_order)
            fig_daily, ax_daily = plt.subplots()
            ax_daily.bar(range(len(daily_failures)), daily_failures.values, color='lightblue', alpha=0.7)
            ax_daily.set_xlabel('Day of Week')
            ax_daily.set_ylabel('Number of Failures')
            ax_daily.set_xticks(range(len(daily_failures)))
            ax_daily.set_xticklabels([day[:3] for day in day_order], rotation=45)
            st.pyplot(fig_daily)
        else:
            st.info("No failure events detected in the data to plot.")

    # ---
    ## 3. Model Performance and Evaluation
    st.write("The model was trained using a **Random Forest Classifier** on an **undersampled** dataset, which achieved the best performance.")
    
    # Display key metrics using Streamlit's new metric component
    optimal_predictions = (y_pred_proba >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_val, optimal_predictions)
    tn, fp, fn, tp = cm.ravel()
    
    col3, col4, col5 = st.columns(3)
    col3.metric("Best Model", "Random Forest")
    col4.metric("Best AUC Score", f"{roc_auc_score(y_val, y_pred_proba):.3f}")
    col5.metric("Optimal Threshold", f"{optimal_threshold:.3f}")

    # Display confusion matrix and classification report
    st.subheader("Confusion Matrix (Validation Set)")
    st.write(cm)
    st.subheader("Classification Report (Validation Set)")
    report_df = pd.DataFrame(classification_report(y_val, optimal_predictions, output_dict=True)).T
    st.dataframe(report_df.round(2))

    # ---
    ## 4. Top Feature Importances
    st.write("These are the most important features the model used to make its predictions.")

    fig_importances, ax_importances = plt.subplots(figsize=(12, 8))
    top_features = importance_df.head(20)
    ax_importances.barh(range(len(top_features)), top_features['importance'], color='teal', alpha=0.8)
    ax_importances.set_yticks(range(len(top_features)))
    ax_importances.set_yticklabels(top_features['feature'], fontsize=12)
    ax_importances.set_xlabel('Feature Importance', fontsize=14)
    ax_importances.set_title(f'Top 20 Feature Importances - Random Forest', fontsize=16)
    ax_importances.invert_yaxis()
    st.pyplot(fig_importances)

    # ---
    ## 5. Live Prediction (Demo)
    st.header("5. Live Prediction (Demo)")
    st.write("Use the sliders below to see how different metric values affect the failure prediction.")

    demo_features = top_features['feature'].tolist()
    input_data = {}
    
    # Use columns to organize the sliders for a cleaner UI
    num_columns = 2
    cols = st.columns(num_columns)
    
    for i, feature in enumerate(demo_features):
        col_idx = i % num_columns
        with cols[col_idx]:
            min_val = X_val[feature].min()
            max_val = X_val[feature].max()
            mean_val = X_val[feature].mean()
            # Create a unique key for each slider
            slider_key = f"slider_{feature}_{i}"
            input_data[feature] = st.slider(
                feature.replace('_', ' ').title(),
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(mean_val),
                key=slider_key
            )

    # Make prediction with user input
    if st.button("Predict Failure"):
        # Get the full list of features the model was trained on
        full_feature_list = best_model.feature_names_
        
        # Create a dictionary with all features, using slider values for the top features
        # and the mean from the validation set for the rest.
        user_input_full = {
            feature: input_data.get(feature, X_val[feature].mean())
            for feature in full_feature_list
        }
        
        user_df = pd.DataFrame([user_input_full])
        
        # Make a temporary copy of user_df and clean it
        user_df_cleaned = pd.DataFrame(clean_data_values(user_df.values), columns=user_df.columns)
        
        # Ensure the DataFrame has all columns in the correct order for the scaler
        user_df_cleaned = user_df_cleaned[full_feature_list]
        user_scaled = best_model.scaler_.transform(user_df_cleaned)
        
        prediction_proba = best_model.predict_proba(user_scaled)[:, 1][0]
        
        st.subheader("Prediction Result")
        st.write(f"The model predicts a **{prediction_proba * 100:.2f}%** probability of failure in the next 10 minutes.")

        if prediction_proba >= optimal_threshold:
            st.error("🚨 **FAILURE ALERT:** The system is at high risk of failure!")
        else:
            st.success("✅ **STATUS:** The system is operating normally.")
