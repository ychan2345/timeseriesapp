import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def create_visualizations(df):
    """Create various visualizations for the data."""
    st.subheader("Data Visualization")

    # Scatter plot
    st.write("### Scatter Plot")
    col1, col2 = st.columns(2)
    x_col = col1.selectbox("Select X-axis", df.select_dtypes(include=[np.number]).columns)
    y_col = col2.selectbox("Select Y-axis", df.select_dtypes(include=[np.number]).columns)

    fig_scatter = px.scatter(df, x=x_col, y=y_col, 
                             title=f"Scatter Plot: {x_col} vs {y_col}")

    # Add correlation coefficient
    correlation = df[x_col].corr(df[y_col])
    fig_scatter.add_annotation(
        text=f"Correlation: {correlation:.2f}",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False
    )
    st.plotly_chart(fig_scatter)

    # Correlation matrix
    st.write("### Correlation Matrix")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()

    # Use a diverging color scale with the midpoint set to 0 so that
    # low (negative) and high (positive) correlations are clearly distinguished.
    fig_corr = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        color_continuous_scale="RdBu_r",      # Diverging color scale
        color_continuous_midpoint=0,          # Set midpoint to zero
        aspect="auto"
    )

    # Update layout for better readability
    fig_corr.update_layout(
        title="Correlation Matrix",
        xaxis_title="",
        yaxis_title="",
        xaxis=dict(tickangle=45),
        coloraxis_colorbar=dict(
            title=dict(
                text="Correlation",
                side="right"
            ),
            ticks="outside"
        )
    )
    st.plotly_chart(fig_corr)

    # Time series plot with multiple y-axes
    st.write("### Time Series Plot")
    date_cols = df.select_dtypes(include=['datetime64']).columns
    if len(date_cols) > 0:
        date_col = st.selectbox("Select Date Column", date_cols)

        # Default to 'Sales' if it exists, otherwise use first numeric column
        default_target = 'Sales' if 'Sales' in df.columns else df.select_dtypes(include=[np.number]).columns[0]

        # Allow multiple column selection with default target
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        selected_cols = st.multiselect(
            "Select Columns to Plot",
            options=numeric_cols,
            default=[default_target]
        )

        if selected_cols:
            # Create figure with secondary y-axis
            fig_ts = go.Figure()

            # Add traces with alternating y-axes
            for i, col in enumerate(selected_cols):
                # Normalize the data to decide which axis to use
                data_range = df[col].max() - df[col].min()
                data_mean = df[col].mean()

                if i == 0:  # First column always on primary axis
                    fig_ts.add_trace(
                        go.Scatter(
                            x=df[date_col],
                            y=df[col],
                            name=col,
                            mode='lines',
                            line=dict(width=2)
                        )
                    )
                else:  # Additional columns on secondary axis if scale is significantly different
                    prev_range = df[selected_cols[0]].max() - df[selected_cols[0]].min()
                    prev_mean = df[selected_cols[0]].mean()

                    # Check if scales are significantly different
                    scale_ratio = data_range / prev_range if prev_range != 0 else 1
                    mean_ratio = data_mean / prev_mean if prev_mean != 0 else 1

                    if scale_ratio > 5 or scale_ratio < 0.2 or mean_ratio > 5 or mean_ratio < 0.2:
                        # Use secondary axis
                        fig_ts.add_trace(
                            go.Scatter(
                                x=df[date_col],
                                y=df[col],
                                name=col,
                                mode='lines',
                                line=dict(width=2),
                                yaxis="y2"
                            )
                        )
                    else:
                        # Use primary axis
                        fig_ts.add_trace(
                            go.Scatter(
                                x=df[date_col],
                                y=df[col],
                                name=col,
                                mode='lines',
                                line=dict(width=2)
                            )
                        )

            # Update layout for dual axes
            fig_ts.update_layout(
                title="Time Series Plot",
                xaxis_title="Date",
                yaxis_title=selected_cols[0],
                yaxis2=dict(
                    title="Other Variables",
                    overlaying="y",
                    side="right"
                ),
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.05
                ),
                margin=dict(r=100)  # Add right margin for secondary axis
            )
            st.plotly_chart(fig_ts, use_container_width=True)
