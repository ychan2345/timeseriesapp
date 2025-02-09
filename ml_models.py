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

    # Use k-fold cross validation
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')

    # Return the negative mean squared error (hyperopt minimizes)
    return {'loss': -scores.mean(), 'status': STATUS_OK}

def calculate_prediction_intervals(y_test, predictions, confidence=0.95):
    """Calculate prediction intervals using residuals."""
    residuals = y_test - predictions
    std_residuals = np.std(residuals)
    z_score = 1.96  # for 95% confidence interval

    confidence_interval = z_score * std_residuals
    lower_bound = predictions - confidence_interval
    upper_bound = predictions + confidence_interval

    return lower_bound, upper_bound

def run_ml_model(df, date_col, target_col, feature_cols, model_type, n_folds=5):
    """Run machine learning time series analysis with hyperparameter optimization."""
    try:
        # Store original dates for plotting
        dates = df[date_col].copy()

        # Create time features from date column
        df = create_time_features(df, date_col)

        # Combine original features with time features
        time_features = ['year', 'month', 'day', 'dayofweek', 'quarter']
        all_features = feature_cols + time_features

        # Prepare data - exclude the original date column from features
        X = df[all_features].copy()  # Only use processed features
        y = df[target_col].copy()

        # Get model class and hyperparameter space
        model_class, space = get_model_and_space(model_type)

        # Train-test split (before hyperparameter optimization)
        train_size = int(len(df) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        dates_test = dates[train_size:]  # Keep dates aligned with test set

        # Run hyperparameter optimization on training data only
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

        # Create model with best parameters
        best_model = create_model(model_class, best)

        # Train on full training set
        best_model.fit(X_train, y_train)

        # Generate predictions for both training and test sets
        train_predictions = best_model.predict(X_train)
        test_predictions = best_model.predict(X_test)

        # Calculate metrics for training set
        train_metrics = calculate_metrics(y_train, train_predictions)

        # Calculate metrics for test set
        test_metrics = calculate_metrics(y_test, test_predictions)

        # Cross-validation scores
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=kf, scoring='r2')

        # Calculate confidence intervals
        lower_bound, upper_bound = calculate_prediction_intervals(y_test, test_predictions)

        # Calculate feature importance
        feature_importance = (
            best_model.feature_importances_ if hasattr(best_model, 'feature_importances_')
            else np.zeros(len(all_features))
        )

        # Add suffixes to metrics
        metrics = {}
        for key, value in train_metrics.items():
            metrics[f'{key}_Train'] = value
        for key, value in test_metrics.items():
            metrics[f'{key}_Test'] = value
        metrics['CV_R2_Mean'] = cv_scores.mean()
        metrics['CV_R2_Std'] = cv_scores.std()

        # Create prediction plot with confidence intervals
        fig_pred = go.Figure()

        # Add actual values
        fig_pred.add_trace(go.Scatter(
            x=dates_test,
            y=y_test,
            name='Actual Values',
            mode='lines',
            line=dict(color='blue')
        ))

        # Add predictions
        fig_pred.add_trace(go.Scatter(
            x=dates_test,
            y=test_predictions,
            name='Predictions',
            mode='lines',
            line=dict(color='red')
        ))

        # Add confidence intervals
        fig_pred.add_trace(go.Scatter(
            x=pd.concat([dates_test, dates_test[::-1]]),
            y=np.concatenate([upper_bound, lower_bound[::-1]]),
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence Interval',
            showlegend=True
        ))

        fig_pred.update_layout(
            title=f'{model_type} Predictions vs Actual Values',
            xaxis_title='Date',
            yaxis_title=target_col,
            showlegend=True,
            xaxis=dict(
                rangeslider=dict(visible=True),
                type='date'
            )
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

        # Store results for LLM interpretation
        st.session_state.model_results = {
            "model_type": f"ML_{model_type}",
            "metrics": metrics,
            "predictions": test_predictions.tolist(),
            "actual_values": y_test.tolist(),
            "feature_importance": dict(zip(all_features, feature_importance.tolist()))
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