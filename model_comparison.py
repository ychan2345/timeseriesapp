import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import io

def add_model_to_comparison(model_results, model_name):
    """Add model results to the comparison session state."""
    if 'model_comparison' not in st.session_state:
        st.session_state.model_comparison = {}

    st.session_state.model_comparison[model_name] = model_results

def reset_comparison():
    """Reset the model comparison session state."""
    if 'model_comparison' in st.session_state:
        del st.session_state.model_comparison

def save_model_to_pickle(model_results):
    """Save model results to a pickle file."""
    try:
        # Create a bytes buffer for the pickle data
        buffer = io.BytesIO()
        pickle.dump(model_results, buffer)
        buffer.seek(0)
        return buffer
    except Exception as e:
        st.error(f"Error saving model: {str(e)}")
        return None

def create_metric_comparison(models_dict):
    """Create a comparison table of metrics across models, excluding CV metrics."""
    metrics_df = pd.DataFrame()

    for model_name, results in models_dict.items():
        metrics = results.get('metrics', {})
        # Ensure consistent metric names and round values
        formatted_metrics = {}
        for k, v in metrics.items():
            # Skip any cross-validation metrics
            if "CV_" in k:
                continue
            try:
                # Convert to float and round
                float_val = float(v)
                # Handle percentage metrics
                if 'MAPE' in k or 'wMAPE' in k:
                    formatted_metrics[k] = f"{float_val:.2f}%"
                else:
                    formatted_metrics[k] = f"{float_val:.3f}"
            except (ValueError, TypeError):
                formatted_metrics[k] = v

        metrics_df[model_name] = pd.Series(formatted_metrics)

    return metrics_df

def create_prediction_comparison_plot(models_dict):
    """Create an overlay plot of predictions from different models."""
    fig = go.Figure()

    for model_name, results in models_dict.items():
        predictions = results.get('predictions', [])
        actual_values = results.get('actual_values', [])
        dates = pd.date_range(
            start=pd.Timestamp.now() - pd.Timedelta(days=len(actual_values)),
            periods=len(actual_values),
            freq='D'
        )

        # Add predictions
        fig.add_trace(go.Scatter(
            x=dates,
            y=predictions,
            name=f'{model_name} Predictions',
            mode='lines'
        ))

    # Add actual values only once
    if actual_values:
        fig.add_trace(go.Scatter(
            x=dates,
            y=actual_values,
            name='Actual Values',
            mode='lines',
            line=dict(color='black', dash='dash')
        ))

    fig.update_layout(
        title='Model Predictions Comparison',
        xaxis_title='Date',
        yaxis_title='Value',
        showlegend=True,
        xaxis=dict(rangeslider=dict(visible=True))
    )

    return fig

def create_feature_importance_comparison(models_dict):
    """Create a comparison plot of feature importance across models."""
    # Collect all unique features
    all_features = set()
    for results in models_dict.values():
        features = results.get('feature_importance', {})
        all_features.update(features.keys())

    # Create subplots for each model
    n_models = len(models_dict)
    fig = make_subplots(
        rows=n_models, cols=1,
        subplot_titles=list(models_dict.keys()),
        vertical_spacing=0.1
    )

    # Add feature importance bars for each model
    for i, (model_name, results) in enumerate(models_dict.items(), start=1):
        features = results.get('feature_importance', {})
        if features:
            # Sort features by importance
            sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)
            feature_names = [f[0] for f in sorted_features]
            importance_values = [f[1] for f in sorted_features]

            fig.add_trace(
                go.Bar(x=feature_names, y=importance_values, name=model_name),
                row=i, col=1
            )

    fig.update_layout(
        height=300 * n_models,
        title='Feature Importance Comparison',
        showlegend=False
    )

    return fig

# Define metrics where higher is better and lower is better
higher_better = ['R2_Train', 'R2_Test']  # R² metrics
lower_better = ['MSE_Train', 'MSE_Test', 'RMSE_Train', 'RMSE_Test',
               'MAE_Train', 'MAE_Test', 'MAPE_Train', 'MAPE_Test',
               'wMAPE_Train', 'wMAPE_Test']  # Error metrics

def highlight_best(row):
    """
    Highlights the best values in each row based on metric type:
    - Yellow background for best values.
    - For R² metrics (in `higher_better`), the highest value is best.
    - For error metrics (in `lower_better`), the lowest value is best.
    """
    try:
        # Remove any '%' symbols and convert to numeric values.
        row_numeric = pd.to_numeric(row.astype(str).str.rstrip('%'), errors='coerce')
        metric_name = row.name  # The row index is the metric name

        # Check which metric category this row belongs to.
        if metric_name in higher_better and not row_numeric.isnull().all():
            best_val = row_numeric.max()
            return ['background-color: yellow' if v == best_val else '' for v in row_numeric]
        elif metric_name in lower_better and not row_numeric.isnull().all():
            best_val = row_numeric.min()
            return ['background-color: yellow' if v == best_val else '' for v in row_numeric]
        else:
            return ['' for _ in row]
    except Exception as e:
        print(f"Error in highlight_best: {e}")
        return ['' for _ in row]

def show_comparison_dashboard():
    """Display the model comparison dashboard."""
    if 'model_comparison' not in st.session_state or not st.session_state.model_comparison:
        st.warning("No models available for comparison. Please run some models first.")
        return

    st.title("🔄 Model Performance Comparison Dashboard")

    # Metric comparison
    st.header("Performance Metrics Comparison")
    metrics_df = create_metric_comparison(st.session_state.model_comparison)

    # Apply the styling and format the display (apply row-wise with axis=1)
    styled_df = metrics_df.style.apply(highlight_best, axis=1)
    st.dataframe(styled_df)

    # Model saving section
    st.header("Save Model")
    st.info("""
    Select a model to download as a pickle (.pkl) file. The downloaded model will be saved to your computer's downloads folder.
    You can later load this model file using Python's pickle library for future predictions.
    """)

    available_models = list(st.session_state.model_comparison.keys())
    selected_model_to_save = st.selectbox(
        "Select model to save:",
        available_models,
        key="model_save_selector"
    )

    if st.button("Save Selected Model"):
        model_data = st.session_state.model_comparison[selected_model_to_save]
        pickle_buffer = save_model_to_pickle(model_data)

        if pickle_buffer is not None:
            st.success(f"Model {selected_model_to_save} is ready for download!")
            st.download_button(
                label="Download Model",
                data=pickle_buffer,
                file_name=f"{selected_model_to_save}.pkl",
                mime="application/octet-stream",
                help="Click to download the model to your computer"
            )

    # Prediction plots comparison
    st.header("Predictions Comparison")
    pred_fig = create_prediction_comparison_plot(st.session_state.model_comparison)
    st.plotly_chart(pred_fig, use_container_width=True)

    # Feature importance comparison
    st.header("Feature Importance Comparison")
    importance_fig = create_feature_importance_comparison(st.session_state.model_comparison)
    st.plotly_chart(importance_fig, use_container_width=True)

    # Add option to reset comparison
    if st.button("Reset Comparison"):
        reset_comparison()
        st.rerun()