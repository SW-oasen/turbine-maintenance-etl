import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re

st.set_page_config(page_title="Turbofan Maintenance Dashboard", layout="wide")

@st.cache_data
def load_data():
    conn = sqlite3.connect('db-sqlite/turbofan.db')

    # Load main features
    features_df = pd.read_sql("""
        SELECT * FROM fct_cycles_features 
        ORDER BY unit_nr, time_cycles
    """, conn)
    
    # Clean and convert data types for Arrow compatibility
    if not features_df.empty:
        # Convert object columns to appropriate types
        for col in features_df.columns:
            if features_df[col].dtype == 'object':
                # Try to convert to numeric first
                try:
                    features_df[col] = pd.to_numeric(features_df[col], errors='ignore')
                except:
                    # If not numeric, keep as string
                    features_df[col] = features_df[col].astype(str)
        
        # Ensure key columns are correct types
        if 'unit_nr' in features_df.columns:
            features_df['unit_nr'] = features_df['unit_nr'].astype(int)
        if 'time_cycles' in features_df.columns:
            features_df['time_cycles'] = features_df['time_cycles'].astype(int)
        if 'rul' in features_df.columns:
            features_df['rul'] = pd.to_numeric(features_df['rul'], errors='coerce')
    
    # Load predictions if available
    try:
        predictions_df = pd.read_sql("""
            SELECT unit_nr, cycle, dataset, model_name, predicted_rul, actual_rul, prediction_date, model_version, confidence_score
            FROM ml_predictions 
            WHERE dataset != 'VALIDATION'
            ORDER BY prediction_date DESC
            LIMIT 1000
        """, conn)
        # Clean predictions data types
        if not predictions_df.empty:
            predictions_df['unit_nr'] = predictions_df['unit_nr'].astype(int)
            predictions_df['cycle'] = predictions_df['cycle'].astype(int)
            predictions_df['predicted_rul'] = pd.to_numeric(predictions_df['predicted_rul'], errors='coerce')
            predictions_df['actual_rul'] = pd.to_numeric(predictions_df['actual_rul'], errors='coerce')
            predictions_df['confidence_score'] = pd.to_numeric(predictions_df['confidence_score'], errors='coerce')
            
    except Exception as e:
        st.write(f"No predictions table found: {e}")
        predictions_df = pd.DataFrame()
    
    conn.close()
    return features_df, predictions_df

def main():
    st.title(" Turbofan Engine Maintenance Dashboard")
    st.markdown("Real-time monitoring and predictive maintenance analytics")
    
    # Load data
    features_df, predictions_df = load_data()
    
    # Data validation
    if features_df.empty:
        st.error("No feature data found. Please run the ETL pipeline first.")
        return
    
    # Display data info
    with st.expander("Data Info"):
        st.write(f"Data shape: {features_df.shape}")
        st.write("Column types:")
        st.write(features_df.dtypes)

    # Sidebar filters
    st.sidebar.header("Filters")
    selected_units = st.sidebar.multiselect(
        "Select Engine Units", 
        options=sorted(features_df['unit_nr'].unique()),
        default=sorted(features_df['unit_nr'].unique())[:5]
    )
    
    # Filter data
    filtered_df = features_df[features_df['unit_nr'].isin(selected_units)]
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Engines", len(features_df['unit_nr'].unique()))

    with col2:
        st.metric("Active Monitoring", len(selected_units))
    
    with col3:
        # Average current RUL calculation
        try:
            if 'rul' in filtered_df.columns and len(filtered_df) > 0:
                # Get the latest RUL for each engine
                max_rul = filtered_df.groupby('unit_nr')['rul'].max()
                avg_rul = max_rul.mean()
                st.metric("Avg RUL", f"{avg_rul:.0f} cycles" if not pd.isna(avg_rul) else "N/A")
            else:
                st.metric("Avg RUL", "N/A")
        except Exception as e:
            st.metric("Avg RUL", "Error")
    
    with col4:
        try:
            if 'rul' in filtered_df.columns and len(filtered_df) > 0:
                max_rul = filtered_df.groupby('unit_nr')['rul'].max()
                median_life = max_rul.median()
                st.metric("Median RUL", f"{median_life:.0f} cycles")
            else:
                st.metric("Median RUL", "N/A")
        except Exception as e:
            st.metric("Median RUL", "Error")

    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Engine Health Over Time")
        fig = px.line(
            filtered_df, 
            x='time_cycles', 
            y='rul',
            color='unit_nr',
            title="Remaining Useful Life by Engine"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Sensor Readings Distribution")
        sensor_pattern = r'(mean.*sensor.*)'
        sensor_cols = [col for col in filtered_df.columns 
                       if re.search(sensor_pattern, col, re.IGNORECASE)]
        selected_sensor = st.selectbox("Select Sensor", sensor_cols)
        
    if sensor_cols:
        # selected_sensor = st.selectbox("Select Sensor", sensor_cols)
        
        # Clean and convert sensor data
        sensor_data = pd.to_numeric(filtered_df[selected_sensor], errors='coerce')
        sensor_data = sensor_data.dropna()
        
        if len(sensor_data) > 0:
            # Use plotly graph_objects for better control
            fig = go.Figure(data=[go.Histogram(
                x=sensor_data,
                nbinsx=30,
                name=selected_sensor
            )])
            
            fig.update_layout(
                title=f"Distribution of {selected_sensor}",
                xaxis_title=selected_sensor,
                yaxis_title="Frequency",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No valid data available for {selected_sensor}")
    else:
        st.warning("No sensor columns found in the data")
    
    # Predictions table
    if not predictions_df.empty:
        st.subheader("Recent Predictions")
        st.dataframe(predictions_df.head(10), use_container_width=True)
    
    # Raw data
    with st.expander("View Raw Data"):
        st.dataframe(filtered_df, use_container_width=True)

if __name__ == "__main__":
    main()