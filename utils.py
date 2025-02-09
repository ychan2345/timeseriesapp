import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime

def load_and_preprocess_data(file):
    """Load and preprocess the uploaded CSV file."""
    try:
        # Read CSV file
        df = pd.read_csv(file)

        # Basic data cleaning
        df = df.replace([np.inf, -np.inf], np.nan)

        # Handle missing values
        null_counts = df.isnull().sum()
        if null_counts.any():
            st.warning("Found missing values, removing them...")
            df = df.dropna()

        # Store initial numeric columns
        initial_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Detect and convert date columns
        date_columns = []
        for col in df.columns:
            if col not in initial_numeric_cols:  # Don't try to convert numeric columns to dates
                try:
                    pd.to_datetime(df[col])
                    date_columns.append(col)
                    df[col] = pd.to_datetime(df[col])
                except:
                    continue

        if not date_columns:
            st.warning("No date columns detected. Some features may be limited.")

        # Get final numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            st.error("No numeric columns found in the dataset.")
            raise ValueError("No numeric columns found in the dataset.")

        return df

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        raise e

def create_time_features(df, date_column):
    """Create time-based features from date column."""
    df = df.copy()
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    df['dayofweek'] = df[date_column].dt.dayofweek
    df['quarter'] = df[date_column].dt.quarter
    return df

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error (MAPE)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_wmape(y_true, y_pred):
    """Calculate Weighted Mean Absolute Percentage Error (wMAPE)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100

def calculate_metrics(y_true, y_pred):
    """Calculate model performance metrics."""
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    metrics = {
        "R2": r2_score(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": calculate_mape(y_true, y_pred),
        "wMAPE": calculate_wmape(y_true, y_pred)
    }

    return metrics