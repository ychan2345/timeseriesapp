import os
from openai import OpenAI
import json
import streamlit as st

# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
def generate_insights(model_results):
    """Generate insights using GPT-4o model."""
    try:
        # Check if API key exists in session state
        if 'OPENAI_API_KEY' not in st.session_state:
            st.error("OpenAI API key not found. Please add your API key first.")
            return None

        client = OpenAI(api_key=st.session_state.OPENAI_API_KEY)

        # Convert model results to a JSON-safe format
        safe_metrics = {}
        for key, value in model_results['metrics'].items():
            if isinstance(value, (int, float)):
                safe_metrics[key] = float(value)  # Convert numpy types to native Python types
            else:
                safe_metrics[key] = str(value)

        # Create a more structured prompt with sanitized data
        prompt = {
            "model_type": str(model_results['model_type']),
            "metrics": safe_metrics,
            "analysis_request": """
                Please analyze these time series model results and provide insights in a structured format.
                Focus on:
                1. Model Performance: Evaluate the metrics and overall model fit
                2. Key Insights: Important patterns and findings from the predictions
                3. Recommendations: Specific suggestions for improvement
                4. Potential Issues: Any concerns or limitations identified

                Return the analysis in this exact JSON format:
                {
                    "performance_interpretation": "detailed analysis of model performance metrics",
                    "key_insights": "main findings from the predictions",
                    "recommendations": "specific suggestions for improvement",
                    "potential_issues": "identified concerns and limitations"
                }
            """
        }

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}],
            response_format={"type": "json_object"}
        )

        insights = json.loads(response.choices[0].message.content)

        # Verify that all expected keys are present
        expected_keys = ["performance_interpretation", "key_insights", 
                        "recommendations", "potential_issues"]

        if not all(key in insights for key in expected_keys):
            st.warning("Incomplete analysis received. Some sections may be missing.")
            # Fill in any missing keys with placeholder text
            for key in expected_keys:
                if key not in insights:
                    insights[key] = "No analysis available for this section."

        return insights

    except json.JSONDecodeError as e:
        st.error(f"Error parsing model results or API response: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Failed to generate insights: {str(e)}")
        if "Incorrect API key" in str(e):
            st.error("Invalid OpenAI API key. Please check your API key and try again.")
        return None