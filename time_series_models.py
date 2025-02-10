import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import plotly.graph_objects as go
from utils import calculate_metrics, create_time_features
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pmdarima as pm  # For auto_arima

def calculate_prediction_intervals(y_test, predictions, confidence=0.95):
    """Calculate prediction intervals using residuals."""
    residuals = y_test - predictions
    std_residuals = np.std(residuals)
    z_score = 1.96  # for 95% confidence interval

    confidence_interval = z_score * std_residuals
    lower_bound = predictions - confidence_interval
    upper_bound = predictions + confidence_interval

    return lower_bound, upper_bound

def calculate_feature_importance(model, feature_cols, model_type, fitted_model=None):
    """Calculate feature importance based on model type."""
    importance_scores = {}

    try:
        if model_type in ["ARIMA", "SARIMA"] and fitted_model and (feature_cols is not None and len(feature_cols) > 0):
            # Get p-values from model summary
            try:
                summary = fitted_model.summary()
                for i, col in enumerate(feature_cols):
                    try:
                        # Add 1 to skip intercept
                        p_value = fitted_model.pvalues[i + 1]
                        # Convert p-value to importance (1 - p_value)
                        importance_scores[col] = 1 - p_value
                    except:
                        importance_scores[col] = 0.0
            except:
                for col in feature_cols:
                    importance_scores[col] = 1.0 / len(feature_cols)

        elif model_type == "Prophet" and (feature_cols is not None and len(feature_cols) > 0):
            # For Prophet, calculate importance based on regressor effects
            for col in feature_cols:
                try:
                    # Get the coefficient for this regressor
                    coef = model.params[f'{col}_beta']
                    importance_scores[col] = abs(coef)
                except:
                    importance_scores[col] = 0.0

        # Normalize scores to [0, 1] range
        if importance_scores:
            max_score = max(importance_scores.values())
            if max_score > 0:
                importance_scores = {k: v / max_score for k, v in importance_scores.items()}
            else:
                importance_scores = {k: 1.0 / len(importance_scores) for k in importance_scores.keys()}

    except Exception as e:
        print(f"Error calculating feature importance: {str(e)}")
        if feature_cols and len(feature_cols) > 0:
            importance_scores = {col: 1.0 / len(feature_cols) for col in feature_cols}

    return importance_scores

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def calculate_wmape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100

def calculate_metrics(y_true, y_pred):
    """Calculate model performance metrics."""
    # Ensure arrays are 1D and have matching dimensions
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true ({len(y_true)}) != y_pred ({len(y_pred)})")

    # Remove any infinite or NaN values
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    # Check if we have valid data
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            "R2": 0,
            "MSE": float('inf'),
            "RMSE": float('inf'),
            "MAE": float('inf'),
            "MAPE": float('inf'),
            "wMAPE": float('inf')
        }

    try:
        r2 = r2_score(y_true, y_pred)
        # Clip R² to a reasonable range (-1 to 1)
        r2 = max(min(r2, 1.0), -1.0)

        metrics = {
            "R2": r2,
            "MSE": mean_squared_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAE": mean_absolute_error(y_true, y_pred),
            "MAPE": calculate_mape(y_true, y_pred),
            "wMAPE": calculate_wmape(y_true, y_pred)
        }
        return metrics
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return {
            "R2": 0,
            "MSE": float('inf'),
            "RMSE": float('inf'),
            "MAE": float('inf'),
            "MAPE": float('inf'),
            "wMAPE": float('inf')
        }

def calculate_time_series_r2(y_true, y_pred):
    """Calculate R² score specifically for time series data."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Remove any NaN or infinite values
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return 0

    # Calculate naive baseline (using previous value as prediction)
    y_baseline = np.roll(y_true, 1)
    y_baseline[0] = y_true[0]  # Handle first value

    # Calculate residual sum of squares (RSS)
    rss = np.sum((y_true - y_pred) ** 2)

    # Calculate total sum of squares (TSS) using naive baseline
    tss = np.sum((y_true - y_baseline) ** 2)

    if tss == 0:
        return 0

    r2 = 1 - (rss / tss)

    # Clip R² to reasonable range
    return max(min(r2, 1.0), -1.0)

def scale_features(train_data, val_data, feature_cols):
    """Scale features using robust scaling."""
    scaled_train = train_data.copy()
    scaled_val = val_data.copy()

    if not (feature_cols and len(feature_cols) > 0):
        return scaled_train, scaled_val

    for col in feature_cols:
        # Calculate robust statistics from training data
        median = train_data[col].median()
        iqr = train_data[col].quantile(0.75) - train_data[col].quantile(0.25)
        if iqr == 0:
            iqr = 1  # Prevent division by zero

        # Scale both training and validation data using training statistics
        scaled_train[col] = (train_data[col] - median) / iqr
        scaled_val[col] = (val_data[col] - median) / iqr

    return scaled_train, scaled_val

def run_time_series_model(df, date_col, target_col, feature_cols, model_type, n_folds=5):
    """Run time series analysis with rolling window cross-validation and fixed test set,
       tuning hyperparameters for ARIMA, SARIMA (using auto_arima), and Prophet (grid search)."""
    try:
        # Sort data by date
        df = df.sort_values(date_col)

        # Store training period information
        training_period_start = df[date_col].min()
        training_period_end = df[date_col].max()

        # Reserve last 20% of data for final testing
        test_size = int(len(df) * 0.2)
        train_full = df[:-test_size]
        test_data = df[-test_size:]

        # Set a minimum window for validation; for SARIMA use at least 24, else 12
        min_window = 24 if model_type == "SARIMA" else 12
        window_size = max(len(train_full) // n_folds, min_window)

        # ---------------------------
        # Hyperparameter Tuning Step using auto_arima for ARIMA/SARIMA
        tuning_train = train_full[:-window_size]
        # Note: tuning_val is not explicitly used here because auto_arima does its own internal CV
        best_params = {}
        if model_type == "ARIMA":
            if feature_cols is not None and len(feature_cols) > 0:
                auto_model = pm.auto_arima(tuning_train[target_col],
                                           exogenous=tuning_train[feature_cols],
                                           seasonal=False,
                                           trace=True,
                                           error_action='ignore',
                                           suppress_warnings=True)
            else:
                auto_model = pm.auto_arima(tuning_train[target_col],
                                           seasonal=False,
                                           trace=True,
                                           error_action='ignore',
                                           suppress_warnings=True)
            best_params['order'] = auto_model.order

        elif model_type == "SARIMA":
            if feature_cols is not None and len(feature_cols) > 0:
                auto_model = pm.auto_arima(tuning_train[target_col],
                                           exogenous=tuning_train[feature_cols],
                                           seasonal=True,
                                           m=12,
                                           trace=True,
                                           error_action='ignore',
                                           suppress_warnings=True)
            else:
                auto_model = pm.auto_arima(tuning_train[target_col],
                                           seasonal=True,
                                           m=12,
                                           trace=True,
                                           error_action='ignore',
                                           suppress_warnings=True)
            best_params['order'] = auto_model.order
            best_params['seasonal_order'] = auto_model.seasonal_order

        elif model_type == "Prophet":
            # For Prophet, we continue to use a grid search approach
            candidate_params = [{'changepoint_prior_scale': cps, 'seasonality_mode': sm}
                                for cps in [0.001, 0.01, 0.1, 0.5]
                                for sm in ['additive', 'multiplicative']]
            best_params_prophet = None
            best_score = np.inf
            for params in candidate_params:
                try:
                    prophet_data = pd.DataFrame({
                        'ds': tuning_train[date_col],
                        'y': tuning_train[target_col]
                    })
                    if feature_cols is not None and len(feature_cols) > 0:
                        for col in feature_cols:
                            prophet_data[col] = tuning_train[col]
                    model = Prophet(
                        yearly_seasonality=True,
                        weekly_seasonality=True,
                        daily_seasonality=False,
                        changepoint_prior_scale=params['changepoint_prior_scale'],
                        seasonality_mode=params['seasonality_mode']
                    )
                    if feature_cols is not None and len(feature_cols) > 0:
                        for col in feature_cols:
                            model.add_regressor(col)
                    model.fit(prophet_data)
                    future_dates = pd.DataFrame({'ds': tuning_train[date_col].iloc[-window_size:]})
                    if feature_cols is not None and len(feature_cols) > 0:
                        for col in feature_cols:
                            future_dates[col] = tuning_train[col].iloc[-window_size:]
                    forecast = model.predict(future_dates)
                    predictions = forecast['yhat'].values
                    actual = tuning_train[target_col].iloc[-window_size:].values
                    score = np.mean(np.abs(actual - predictions))
                    if score < best_score:
                        best_score = score
                        best_params_prophet = params
                except Exception as e:
                    continue
            if best_params_prophet is None:
                best_params_prophet = {'changepoint_prior_scale': 0.05, 'seasonality_mode': 'additive'}
            best_params['prophet'] = best_params_prophet

        st.write("Best hyperparameters found:", best_params)
        # ---------------------------
        # Now perform rolling window cross-validation using the tuned hyperparameters
        cv_scores = []
        final_predictions = None
        final_actuals = None
        final_dates = None
        fitted_model = None
        prophet_model = None

        for fold in range(n_folds):
            try:
                start_idx = fold * window_size
                end_idx = len(train_full) - window_size  # keep last window_size for validation
                if start_idx >= end_idx:
                    break

                train_data = train_full[start_idx:end_idx]
                val_data = train_full[end_idx:end_idx + window_size]

                if len(train_data) < min_window or len(val_data) < min_window:
                    print(f"Skipping fold {fold} due to insufficient data")
                    continue

                predictions = None

                if model_type == "ARIMA":
                    order = best_params.get('order', (1, 1, 1))
                    try:
                        if feature_cols is not None and len(feature_cols) > 0:
                            model = SARIMAX(train_data[target_col],
                                            exog=train_data[feature_cols],
                                            order=order)
                        else:
                            model = ARIMA(train_data[target_col], order=order)
                        fitted_model = model.fit()
                        if feature_cols is not None and len(feature_cols) > 0:
                            predictions = fitted_model.forecast(steps=len(val_data),
                                                                exog=val_data[feature_cols])
                        else:
                            predictions = fitted_model.forecast(steps=len(val_data))
                        predictions = np.array(predictions).flatten()
                    except Exception as e:
                        print(f"ARIMA error in fold {fold}: {str(e)}")
                        continue

                elif model_type == "SARIMA":
                    order = best_params.get('order', (1, 1, 1))
                    seasonal_order = best_params.get('seasonal_order', (1, 1, 1, 12))
                    try:
                        if feature_cols is not None and len(feature_cols) > 0:
                            model = SARIMAX(train_data[target_col],
                                            exog=train_data[feature_cols],
                                            order=order,
                                            seasonal_order=seasonal_order)
                        else:
                            model = SARIMAX(train_data[target_col],
                                            order=order,
                                            seasonal_order=seasonal_order)
                        fitted_model = model.fit()
                        if feature_cols is not None and len(feature_cols) > 0:
                            predictions = fitted_model.forecast(steps=len(val_data),
                                                                exog=val_data[feature_cols])
                        else:
                            predictions = fitted_model.forecast(steps=len(val_data))
                        predictions = np.array(predictions).flatten()
                    except Exception as e:
                        print(f"SARIMA error in fold {fold}: {str(e)}")
                        continue

                elif model_type == "Prophet":
                    try:
                        prophet_data = pd.DataFrame({
                            'ds': train_data[date_col],
                            'y': train_data[target_col]
                        })
                        if feature_cols is not None and len(feature_cols) > 0:
                            for col in feature_cols:
                                prophet_data[col] = train_data[col]
                        params = best_params.get('prophet', {'changepoint_prior_scale': 0.05, 'seasonality_mode': 'additive'})
                        prophet_model = Prophet(
                            yearly_seasonality=True,
                            weekly_seasonality=True,
                            daily_seasonality=False,
                            changepoint_prior_scale=params['changepoint_prior_scale'],
                            seasonality_mode=params['seasonality_mode']
                        )
                        if feature_cols is not None and len(feature_cols) > 0:
                            for col in feature_cols:
                                prophet_model.add_regressor(col)
                        prophet_model.fit(prophet_data)
                        future_dates = pd.DataFrame({'ds': val_data[date_col]})
                        if feature_cols is not None and len(feature_cols) > 0:
                            for col in feature_cols:
                                future_dates[col] = val_data[col]
                        forecast = prophet_model.predict(future_dates)
                        predictions = forecast['yhat'].values
                    except Exception as e:
                        print(f"Prophet error in fold {fold}: {str(e)}")
                        continue

                # Optionally scale features (though scaling is done before model fitting in many cases)
                scaled_train, scaled_val = scale_features(train_data, val_data, feature_cols)

                if predictions is not None:
                    mask = np.isfinite(val_data[target_col].values) & np.isfinite(predictions)
                    y_true = val_data[target_col].values[mask]
                    y_pred = predictions[mask]
                    if len(y_true) > 0:
                        ts_r2 = calculate_time_series_r2(y_true, y_pred)
                        fold_metrics = calculate_metrics(y_true, y_pred)
                        fold_metrics["R2"] = ts_r2  # Override standard R² with time-series R²
                        if -1 <= ts_r2 <= 1:
                            cv_scores.append(fold_metrics)
                        else:
                            print(f"Skipping fold {fold} due to invalid R² score: {ts_r2}")

            except Exception as fold_error:
                print(f"Error in fold {fold}: {str(fold_error)}")
                continue

        # Fallback: if no valid CV scores were collected, use final training metrics as CV scores.
        if not cv_scores:
            st.warning("No valid cross-validation scores were calculated. Falling back to final training metrics.")
            fallback_model = ARIMA(train_full[target_col], order=best_params.get('order', (1, 1, 1))).fit()
            cv_scores = [calculate_metrics(train_full[target_col], fallback_model.fittedvalues)]

        # Train final model on all training data using tuned hyperparameters and get predictions
        if model_type == "Prophet":
            prophet_data = pd.DataFrame({
                'ds': train_full[date_col],
                'y': train_full[target_col]
            })
            if feature_cols is not None and len(feature_cols) > 0:
                for col in feature_cols:
                    prophet_data[col] = train_full[col]
            params = best_params.get('prophet', {'changepoint_prior_scale': 0.05, 'seasonality_mode': 'additive'})
            prophet_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_prior_scale=params['changepoint_prior_scale'],
                seasonality_mode=params['seasonality_mode']
            )
            if feature_cols is not None and len(feature_cols) > 0:
                for col in feature_cols:
                    prophet_model.add_regressor(col)
            prophet_model.fit(prophet_data)
            future_dates = pd.DataFrame({'ds': test_data[date_col]})
            if feature_cols is not None and len(feature_cols) > 0:
                for col in feature_cols:
                    future_dates[col] = test_data[col]
            forecast = prophet_model.predict(future_dates)
            final_predictions = forecast['yhat'].values

            train_future = pd.DataFrame({'ds': train_full[date_col]})
            if feature_cols is not None and len(feature_cols) > 0:
                for col in feature_cols:
                    train_future[col] = train_full[col]
            train_forecast = prophet_model.predict(train_future)
            train_predictions = train_forecast['yhat'].values

        else:  # ARIMA or SARIMA
            if feature_cols is not None and len(feature_cols) > 0:
                if model_type == "ARIMA":
                    order = best_params.get('order', (1, 1, 1))
                    model = SARIMAX(train_full[target_col],
                                    exog=train_full[feature_cols],
                                    order=order)
                else:  # SARIMA
                    order = best_params.get('order', (1, 1, 1))
                    seasonal_order = best_params.get('seasonal_order', (1, 1, 1, 12))
                    model = SARIMAX(train_full[target_col],
                                    exog=train_full[feature_cols],
                                    order=order,
                                    seasonal_order=seasonal_order)
            else:
                if model_type == "ARIMA":
                    order = best_params.get('order', (1, 1, 1))
                    model = ARIMA(train_full[target_col], order=order)
                else:  # SARIMA
                    order = best_params.get('order', (1, 1, 1))
                    seasonal_order = best_params.get('seasonal_order', (1, 1, 1, 12))
                    model = SARIMAX(train_full[target_col],
                                    order=order,
                                    seasonal_order=seasonal_order)
            fitted_model = model.fit()
            if feature_cols is not None and len(feature_cols) > 0:
                final_predictions = fitted_model.forecast(
                    steps=len(test_data),
                    exog=test_data[feature_cols]
                )
                train_predictions = fitted_model.get_prediction(
                    start=0,
                    end=len(train_full) - 1,
                    exog=train_full[feature_cols]
                ).predicted_mean
            else:
                final_predictions = fitted_model.forecast(steps=len(test_data))
                train_predictions = fitted_model.get_prediction(
                    start=0,
                    end=len(train_full) - 1
                ).predicted_mean

        final_actuals = test_data[target_col]
        final_dates = test_data[date_col]

        # Calculate metrics
        cv_metrics = {}

        train_metrics = calculate_metrics(train_full[target_col], train_predictions)
        for key, value in train_metrics.items():
            cv_metrics[f'{key}_Train'] = value

        test_metrics = calculate_metrics(final_actuals, final_predictions)
        for key, value in test_metrics.items():
            cv_metrics[f'{key}_Test'] = value

        metric_keys = cv_scores[0].keys()
        for key in metric_keys:
            if key == "R2":
                values = [score[key] for score in cv_scores if -1 <= score[key] <= 1]
            else:
                values = [score[key] for score in cv_scores if score[key] >= 0]
            if values:
                cv_metrics[f'{key}_CV_Mean'] = np.mean(values)
                cv_metrics[f'{key}_CV_Std'] = np.std(values)
            else:
                cv_metrics[f'{key}_CV_Mean'] = 0
                cv_metrics[f'{key}_CV_Std'] = 0

        lower_bound, upper_bound = calculate_prediction_intervals(final_actuals, final_predictions)

        # Create prediction plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df[date_col][:-test_size],
            y=df[target_col][:-test_size],
            name='Training Data',
            mode='lines',
            line=dict(color='lightblue', width=1)
        ))
        fig.add_trace(go.Scatter(
            x=final_dates,
            y=final_actuals,
            name='Test Data (Actual)',
            mode='lines',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=final_dates,
            y=final_predictions,
            name='Predictions',
            mode='lines',
            line=dict(color='red', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([final_dates, final_dates[::-1]]),
            y=np.concatenate([upper_bound, lower_bound[::-1]]),
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% Confidence Interval',
            showlegend=True
        ))
        first_test_date = df.iloc[-test_size][date_col]
        fig.add_vline(
            x=first_test_date,
            line_dash="dash",
            line_color="gray",
            opacity=0.7
        )
        fig.add_annotation(
            x=first_test_date,
            y=1.05,
            yref="paper",
            text="Test Set Start",
            showarrow=False,
            font=dict(size=12)
        )
        fig.update_layout(
            title=f'{model_type} Model Predictions',
            xaxis_title='Date',
            yaxis_title=target_col,
            showlegend=True,
            xaxis=dict(
                rangeslider=dict(visible=True),
                type='date'
            )
        )

        importance_fig = None
        if feature_cols is not None and len(feature_cols) > 0:
            importance_scores = calculate_feature_importance(
                prophet_model if model_type == "Prophet" else None,
                feature_cols,
                model_type,
                fitted_model if model_type in ["ARIMA", "SARIMA"] else None
            )
            importance_fig = go.Figure()
            sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
            feature_names = [f[0] for f in sorted_features]
            importance_values = [f[1] for f in sorted_features]
            importance_fig.add_trace(go.Bar(
                x=feature_names,
                y=importance_values,
                name='Feature Importance'
            ))
            importance_fig.update_layout(
                title=f'Feature Importance ({model_type} model)',
                xaxis_title='Features',
                yaxis_title='Importance Score',
                showlegend=False
            )

        st.session_state.model_results = {
            "model_type": model_type,
            "metrics": cv_metrics,
            "predictions": final_predictions.tolist(),
            "actual_values": final_actuals.tolist(),
            "feature_importance": importance_scores if (feature_cols is not None and len(feature_cols) > 0) else {},
            "training_period_start": training_period_start.strftime("%Y-%m-%d"),
            "training_period_end": training_period_end.strftime("%Y-%m-%d"),
            "model_object": prophet_model if model_type == "Prophet" else fitted_model
        }

        return {
            "success": True,
            "metrics": cv_metrics,
            "plot": fig,
            "feature_importance": importance_fig
        }

    except Exception as e:
        st.error(f"Error in time series modeling: {str(e)}")
        return {"success": False}