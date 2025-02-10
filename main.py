import streamlit as st
import pandas as pd
from utils import load_and_preprocess_data
from visualization import create_visualizations
from time_series_models import run_time_series_model
from ml_models import run_ml_model
from llm_insights import generate_insights
from model_comparison import add_model_to_comparison, show_comparison_dashboard

st.set_page_config(
    page_title="Time Series Forecasting App",
    page_icon="📈",
    layout="wide"
)

def main():
    st.title("📈 Time Series Forecasting Application")

    # File upload section
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

    if uploaded_file is not None:
        try:
            # Load and preprocess data
            df = load_and_preprocess_data(uploaded_file)

            if df is not None:
                # Store processed data in session state
                st.session_state["processed_data"] = df.copy()

                # Navigation
                analysis_type = st.sidebar.selectbox(
                    "Select Analysis Type",
                    ["Data Visualization", "Time Series Model", "ML Time Series", 
                     "Model Comparison", "LLM Interpretations"]
                )

                # Handle different analysis types
                if analysis_type == "Data Visualization":
                    # Only show data overview in visualization section
                    st.subheader("Data Overview")
                    st.dataframe(df.head())
                    st.write(f"Total records: {len(df)}")
                    create_visualizations(df)

                elif analysis_type == "Time Series Model":
                    date_col = st.selectbox("Select Date Column", df.columns)
                    # Exclude the selected date column from the target variable options
                    target_col = st.selectbox("Select Target Variable", [col for col in df.columns if col != date_col])

                    # Add optional feature selection
                    use_additional_features = st.checkbox("Use Additional Features")
                    feature_cols = []
                    if use_additional_features:
                        feature_cols = st.multiselect(
                            "Select Additional Feature Columns",
                            [col for col in df.columns if col not in [date_col, target_col]]
                        )

                    model_type = st.selectbox("Select Model Type", ["ARIMA", "SARIMA", "Prophet"])

                    # Add cross-validation fold selection
                    n_folds = st.slider("Number of Cross-validation Folds", 
                                        min_value=2, max_value=10, value=5)

                    if st.button("Run Time Series Analysis"):
                        with st.spinner('Training model...'):
                            results = run_time_series_model(df, date_col, target_col, 
                                                             feature_cols, model_type, n_folds)

                            if results.get("success"):
                                # Add to comparison
                                model_name = f"{model_type}_Model"
                                existing_runs = len([k for k in st.session_state.get('model_comparison', {}).keys() if model_type in k])
                                if existing_runs > 0:
                                    model_name = f"{model_type}_Model_Run_{existing_runs + 1}"
                                add_model_to_comparison(st.session_state.model_results, model_name)
                                st.success(f"Added {model_name} to comparison!")

                                st.success("Model training completed!")

                                # Display metrics in a more organized way
                                st.header("Model Performance Metrics")

                                # Training Set Metrics
                                st.subheader("Full Training Set Metrics (First 80% of Data)")
                                st.caption("These metrics are calculated on the entire training set, not cross-validation results")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("R² (Full Training)", f"{results['metrics'].get('R2_Train', 0):.3f}")
                                    st.metric("MAE (Full Training)", f"{results['metrics'].get('MAE_Train', 0):.3f}")
                                with col2:
                                    st.metric("RMSE (Full Training)", f"{results['metrics'].get('RMSE_Train', 0):.3f}")
                                    st.metric("MAPE (Full Training)", f"{results['metrics'].get('MAPE_Train', 0):.2f}%")
                                with col3:
                                    st.metric("MSE (Full Training)", f"{results['metrics'].get('MSE_Train', 0):.3f}")
                                    st.metric("wMAPE (Full Training)", f"{results['metrics'].get('wMAPE_Train', 0):.2f}%")

                                # Test Set Metrics
                                st.subheader("Test Set Metrics (Last 20% of Data)")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("R² (Test)", f"{results['metrics'].get('R2_Test', 0):.3f}")
                                    st.metric("MAE (Test)", f"{results['metrics'].get('MAE_Test', 0):.3f}")
                                with col2:
                                    st.metric("RMSE (Test)", f"{results['metrics'].get('RMSE_Test', 0):.3f}")
                                    st.metric("MAPE (Test)", f"{results['metrics'].get('MAPE_Test', 0):.2f}%")
                                with col3:
                                    st.metric("MSE (Test)", f"{results['metrics'].get('MSE_Test', 0):.3f}")
                                    st.metric("wMAPE (Test)", f"{results['metrics'].get('wMAPE_Test', 0):.2f}%")

                                # Plots
                                st.subheader("Predictions vs Actual Values")
                                st.plotly_chart(results["plot"])

                                # Feature importance plot if available
                                if results.get("feature_importance") is not None:
                                    st.subheader("Feature Importance")
                                    st.plotly_chart(results["feature_importance"])

                elif analysis_type == "ML Time Series":
                    date_col = st.selectbox("Select Date Column", df.columns)
                    # Exclude the selected date column from the target variable options
                    target_col = st.selectbox("Select Target Variable", 
                                                [col for col in df.columns if col != date_col])
                    # Store target column in session state
                    st.session_state["target_col"] = target_col
                    feature_cols = st.multiselect("Select Feature Columns", 
                                                   [col for col in df.columns if col not in [date_col, target_col]])
                    model_type = st.selectbox("Select Model Type", 
                                              ["RandomForest", "XGBoost"])
                    n_folds = st.slider("Number of Cross-validation Folds", 
                                        min_value=2, max_value=10, value=5)

                    if st.button("Run ML Analysis"):
                        with st.spinner('Training model with hyperparameter optimization...'):
                            results = run_ml_model(df, date_col, target_col, 
                                                  feature_cols, model_type, n_folds)

                            if results.get("success"):
                                # Add to comparison
                                model_name = f"{model_type}_ML_Model"
                                existing_runs = len([k for k in st.session_state.get('model_comparison', {}).keys() if model_type in k])
                                if existing_runs > 0:
                                    model_name = f"{model_type}_ML_Model_Run_{existing_runs + 1}"
                                add_model_to_comparison(st.session_state.model_results, model_name)
                                st.success(f"Added {model_name} to comparison!")

                                st.success("Model training completed!")

                                # Display metrics in a more organized way
                                st.header("Model Performance Metrics")

                                # Training Set Metrics
                                st.subheader("Training Set Metrics (First 80% of Data)")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("R² (Train)", f"{results['metrics'].get('R2_Train', 0):.3f}")
                                    st.metric("MAE (Train)", f"{results['metrics'].get('MAE_Train', 0):.3f}")
                                with col2:
                                    st.metric("RMSE (Train)", f"{results['metrics'].get('RMSE_Train', 0):.3f}")
                                    st.metric("MAPE (Train)", f"{results['metrics'].get('MAPE_Train', 0):.2f}%")
                                with col3:
                                    st.metric("MSE (Train)", f"{results['metrics'].get('MSE_Train', 0):.3f}")
                                    st.metric("wMAPE (Train)", f"{results['metrics'].get('wMAPE_Train', 0):.2f}%")

                                # Test Set Metrics
                                st.subheader("Test Set Metrics (Last 20% of Data)")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("R² (Test)", f"{results['metrics'].get('R2_Test', 0):.3f}")
                                    st.metric("MAE (Test)", f"{results['metrics'].get('MAE_Test', 0):.3f}")
                                with col2:
                                    st.metric("RMSE (Test)", f"{results['metrics'].get('RMSE_Test', 0):.3f}")
                                    st.metric("MAPE (Test)", f"{results['metrics'].get('MAPE_Test', 0):.2f}%")
                                with col3:
                                    st.metric("MSE (Test)", f"{results['metrics'].get('MSE_Test', 0):.3f}")
                                    st.metric("wMAPE (Test)", f"{results['metrics'].get('wMAPE_Test', 0):.2f}%")

                                # Plots
                                st.subheader("Predictions vs Actual Values")
                                st.plotly_chart(results["plot"])

                                st.subheader("Feature Importance")
                                st.plotly_chart(results["feature_importance"])

                elif analysis_type == "Model Comparison":
                    show_comparison_dashboard()

                elif analysis_type == "LLM Interpretations":
                    st.header("LLM Model Interpretations")

                    # API key management section
                    st.subheader("API Key Management")
                    if 'OPENAI_API_KEY' in st.session_state:
                        st.success("OpenAI API key is set")
                        if st.button("Change API Key"):
                            del st.session_state.OPENAI_API_KEY
                            st.rerun()
                    else:
                        api_key = st.text_input("Enter your OpenAI API key:", type="password")
                        if api_key:
                            st.session_state.OPENAI_API_KEY = api_key
                            st.success("API key saved!")
                            st.rerun()

                    # Check for available models in comparison
                    if 'model_comparison' not in st.session_state or not st.session_state.model_comparison:
                        st.warning("Please run at least one model first to get LLM interpretations.")
                    else:
                        # Model selection dropdown
                        available_models = list(st.session_state.model_comparison.keys())
                        selected_model = st.selectbox(
                            "Select model to analyze:",
                            available_models,
                            index=0
                        )

                        st.info(f"Selected model for analysis: {selected_model}")

                        # Only show generate button if we have API key
                        if 'OPENAI_API_KEY' in st.session_state:
                            if st.button("Generate Model Insights"):
                                with st.spinner('Generating insights...'):
                                    # Use the selected model's results for analysis
                                    model_results = st.session_state.model_comparison[selected_model]
                                    insights = generate_insights(model_results)

                                    if insights:
                                        st.write("Model Insights and Recommendations:")
                                        # Display insights in a more structured way
                                        if "performance_interpretation" in insights:
                                            st.subheader("Model Performance")
                                            st.write(insights["performance_interpretation"])
                                        if "key_insights" in insights:
                                            st.subheader("Key Insights")
                                            st.write(insights["key_insights"])
                                        if "recommendations" in insights:
                                            st.subheader("Recommendations")
                                            st.write(insights["recommendations"])
                                        if "potential_issues" in insights:
                                            st.subheader("Potential Issues")
                                            st.write(insights["potential_issues"])
                        else:
                            st.warning("Please enter your OpenAI API key above to generate insights.")

        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
    else:
        st.info("Please upload a CSV file to begin analysis.")

if __name__ == "__main__":
    main()