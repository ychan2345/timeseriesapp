import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import plotly.graph_objects as go
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from utils import create_time_features, calculate_metrics

# Define exports before any other code
__all__ = ['run_ml_model']

def get_model_and_space(model_type):
    """Get model class and hyperparameter space based on model type."""
    if model_type == "RandomForest":
        model_class = RandomForestRegressor
        space = {
            'n_estimators': hp.quniform('n_estimators', 10, 300, 10),
            'max_depth': hp.quniform('max_depth', 3, 20, 1),
            'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
            'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 5, 1)
        }
    elif model_type == "XGBoost":
        model_class = XGBRegressor
        space = {
            'n_estimators': hp.quniform('n_estimators', 10, 300, 10),
            'max_depth': hp.quniform('max_depth', 3, 20, 1),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
            'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
            'subsample': hp.uniform('subsample', 0.5, 1.0)
        }
    return model_class, space

def create_model(model_class, params):
    """Create model instance with given parameters."""
    params = {k: int(v) if k in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'min_child_weight']
             else v for k, v in params.items()}
    return model_class(**params, random_state=42)

def objective(params, model_class, X, y, cv_splits):
    """Objective function for hyperopt optimization."""
    model = create_model(model_class, params)
    kf = KFold(n_splits=cv_splits, shuffle=False)  # No shuffle for time series
    scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
    return {'loss': -scores.mean(), 'status': STATUS_OK}

def run_ml_model(df, date_col, target_col, feature_cols, model_type, n_folds=5):
    """Run machine learning time series analysis with hyperparameter optimization."""
    try:
        # Store original dates for plotting and metadata
        dates = df[date_col].copy()
        training_period_start = dates.min()
        training_period_end = dates.max()

        # Create time features from date column
        df = create_time_features(df, date_col)

        # Combine original features with time features
        time_features = ['year', 'month', 'day', 'dayofweek', 'quarter']
        all_features = feature_cols + time_features

        # Prepare data
        X = df[all_features].copy()
        y = df[target_col].copy()

        # Train-test split (preserving time order)
        train_size = int(len(df) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        dates_train, dates_test = dates[:train_size], dates[train_size:]

        # Get model class and hyperparameter space
        model_class, space = get_model_and_space(model_type)

        # Run hyperparameter optimization on training data
        with st.spinner('Optimizing hyperparameters...'):
            trials = Trials()
            best = fmin(
                fn=lambda params: objective(params, model_class, X_train, y_train, n_folds),
                space=space,
                algo=tpe.suggest,
                max_evals=50,
                trials=trials,
                rstate=np.random.default_rng(42)
            )

        # Create and train model with best parameters
        best_model = create_model(model_class, best)

        # Train on training set
        best_model.fit(X_train, y_train)

        # Generate predictions for both sets
        train_predictions = best_model.predict(X_train)
        test_predictions = best_model.predict(X_test)

        # Calculate metrics for both sets
        train_metrics = calculate_metrics(y_train, train_predictions)
        test_metrics = calculate_metrics(y_test, test_predictions)

        # Combine metrics with appropriate suffixes
        metrics = {}
        for key, value in train_metrics.items():
            metrics[f'{key}_Train'] = value
        for key, value in test_metrics.items():
            metrics[f'{key}_Test'] = value

        # Calculate feature importance
        feature_importance = (
            best_model.feature_importances_ if hasattr(best_model, 'feature_importances_')
            else np.zeros(len(all_features))
        )

        # Create prediction plot
        fig_pred = go.Figure()

        # Add training data
        fig_pred.add_trace(go.Scatter(
            x=dates_train,
            y=y_train,
            name='Training Actual',
            mode='lines',
            line=dict(color='blue')
        ))
        fig_pred.add_trace(go.Scatter(
            x=dates_train,
            y=train_predictions,
            name='Training Predictions',
            mode='lines',
            line=dict(color='lightblue')
        ))

        # Add test data
        fig_pred.add_trace(go.Scatter(
            x=dates_test,
            y=y_test,
            name='Test Actual',
            mode='lines',
            line=dict(color='red')
        ))
        fig_pred.add_trace(go.Scatter(
            x=dates_test,
            y=test_predictions,
            name='Test Predictions',
            mode='lines',
            line=dict(color='orange')
        ))

        fig_pred.update_layout(
            title=f'{model_type} Predictions vs Actual Values',
            xaxis_title='Date',
            yaxis_title=target_col,
            showlegend=True
        )

        # Create feature importance plot
        fig_importance = go.Figure()
        fig_importance.add_trace(go.Bar(
            x=all_features,
            y=feature_importance,
            name='Feature Importance'
        ))

        fig_importance.update_layout(
            title='Feature Importance',
            xaxis_title='Features',
            yaxis_title='Importance Score'
        )

        # Store results with additional metadata
        st.session_state.model_results = {
            "model_type": f"ML_{model_type}",
            "metrics": metrics,
            "predictions": test_predictions.tolist(),
            "actual_values": y_test.tolist(),
            "feature_importance": dict(zip(all_features, feature_importance.tolist())),
            "training_period_start": training_period_start.strftime("%Y-%m-%d"),
            "training_period_end": training_period_end.strftime("%Y-%m-%d"),
            "model_object": best_model
        }

        return {
            "success": True,
            "metrics": metrics,
            "plot": fig_pred,
            "feature_importance": fig_importance
        }

    except Exception as e:
        st.error(f"Error in ML modeling: {str(e)}")
        return {"success": False}