"""
Real-time Traffic Dashboard with Streamlit
Interactive visualization for traffic congestion monitoring and prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🚦 Traffic Congestion Monitor",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .alert-high {
        background-color: #ff4b4b;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .alert-medium {
        background-color: #ffa500;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .alert-low {
        background-color: #00cc00;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load training data"""
    try:
        df = pd.read_csv('Train.csv')
        df['datetime'] = pd.to_datetime(df['datetimestamp_start'])
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['date'] = df['datetime'].dt.date
        return df
    except Exception as e:
        st.error(f"Veri yüklenemedi: {e}")
        return None


@st.cache_resource
def load_model():
    """Load trained prediction model"""
    try:
        # Try loading ensemble models first
        if Path('voting_ensemble_enter_model.pkl').exists():
            enter_model = joblib.load('voting_ensemble_enter_model.pkl')
            exit_model = joblib.load('voting_ensemble_exit_model.pkl')
            return enter_model, exit_model, 'Ensemble Model'
        elif Path('tuned_enter_model.pkl').exists():
            enter_model = joblib.load('tuned_enter_model.pkl')
            exit_model = joblib.load('tuned_exit_model.pkl')
            return enter_model, exit_model, 'Tuned Model'
        else:
            return None, None, 'No Model'
    except Exception as e:
        st.warning(f"Model yüklenemedi: {e}")
        return None, None, 'No Model'


def plot_congestion_distribution(df):
    """Plot congestion level distribution"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Enter congestion
    enter_counts = df['congestion_enter_rating'].value_counts().sort_index()
    ax1.bar(enter_counts.index, enter_counts.values, color=['green', 'yellow', 'orange', 'red'])
    ax1.set_xlabel('Congestion Level')
    ax1.set_ylabel('Count')
    ax1.set_title('Enter Congestion Distribution')
    ax1.set_xticks(range(len(enter_counts)))
    ax1.set_xticklabels(['Free\nFlowing', 'Light\nDelay', 'Moderate\nDelay', 'Heavy\nDelay'])
    
    # Exit congestion
    exit_counts = df['congestion_exit_rating'].value_counts().sort_index()
    ax2.bar(exit_counts.index, exit_counts.values, color=['green', 'yellow', 'orange', 'red'])
    ax2.set_xlabel('Congestion Level')
    ax2.set_ylabel('Count')
    ax2.set_title('Exit Congestion Distribution')
    ax2.set_xticks(range(len(exit_counts)))
    ax2.set_xticklabels(['Free\nFlowing', 'Light\nDelay', 'Moderate\nDelay', 'Heavy\nDelay'])
    
    plt.tight_layout()
    return fig


def plot_hourly_pattern(df):
    """Plot hourly congestion patterns"""
    hourly_stats = df.groupby('hour').agg({
        'congestion_enter_rating': lambda x: (x != 'free flowing').mean() * 100,
        'congestion_exit_rating': lambda x: (x != 'free flowing').mean() * 100
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(hourly_stats['hour'], hourly_stats['congestion_enter_rating'], 
            marker='o', linewidth=2, label='Enter', color='#1f77b4')
    ax.plot(hourly_stats['hour'], hourly_stats['congestion_exit_rating'], 
            marker='s', linewidth=2, label='Exit', color='#ff7f0e')
    
    # Highlight rush hours
    rush_hours = [7, 8, 9, 16, 17, 18]
    for hour in rush_hours:
        ax.axvspan(hour - 0.5, hour + 0.5, alpha=0.2, color='red')
    
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Congestion Percentage (%)', fontsize=12)
    ax.set_title('Hourly Traffic Congestion Patterns', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(24))
    
    plt.tight_layout()
    return fig


def plot_heatmap(df):
    """Plot congestion heatmap by day and hour"""
    pivot = df.pivot_table(
        values='congestion_enter_rating',
        index='day_of_week',
        columns='hour',
        aggfunc=lambda x: (x != 'free flowing').mean() * 100
    )
    
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(pivot, annot=False, fmt='.0f', cmap='RdYlGn_r', 
                cbar_kws={'label': 'Congestion %'}, ax=ax)
    
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Day of Week', fontsize=12)
    ax.set_title('Traffic Congestion Heatmap', fontsize=14, fontweight='bold')
    ax.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=0)
    
    plt.tight_layout()
    return fig


def predict_congestion(enter_model, exit_model, features):
    """Make congestion predictions"""
    try:
        enter_pred = enter_model.predict(features)[0]
        exit_pred = exit_model.predict(features)[0]
        
        # Get probabilities if available
        if hasattr(enter_model, 'predict_proba'):
            enter_proba = enter_model.predict_proba(features)[0]
            exit_proba = exit_model.predict_proba(features)[0]
        else:
            enter_proba = None
            exit_proba = None
        
        return enter_pred, exit_pred, enter_proba, exit_proba
    except Exception as e:
        st.error(f"Tahmin hatası: {e}")
        return None, None, None, None


def main():
    # Header
    st.markdown('<h1 class="main-header">🚦 Traffic Congestion Monitor</h1>', unsafe_allow_html=True)
    st.markdown("**Norman Niles Roundabout - Real-time Traffic Analysis**")
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Load model
    enter_model, exit_model, model_name = load_model()
    
    # Sidebar
    st.sidebar.header("⚙️ Dashboard Controls")
    
    # Date filter
    st.sidebar.subheader("📅 Date Filter")
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(df['date'].min(), df['date'].max()),
        min_value=df['date'].min(),
        max_value=df['date'].max()
    )
    
    # Hour filter
    st.sidebar.subheader("🕐 Hour Filter")
    hour_range = st.sidebar.slider("Select Hour Range", 0, 23, (0, 23))
    
    # Filter data
    if len(date_range) == 2:
        mask = (df['date'] >= date_range[0]) & (df['date'] <= date_range[1])
        mask &= (df['hour'] >= hour_range[0]) & (df['hour'] <= hour_range[1])
        filtered_df = df[mask]
    else:
        filtered_df = df
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Analytics", "🔮 Prediction", "📋 Data"])
    
    # TAB 1: Overview
    with tab1:
        st.header("Traffic Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_records = len(filtered_df)
        congested_enter = (filtered_df['congestion_enter_rating'] != 'free flowing').sum()
        congested_exit = (filtered_df['congestion_exit_rating'] != 'free flowing').sum()
        avg_congestion = (congested_enter + congested_exit) / (2 * total_records) * 100
        
        with col1:
            st.metric("📊 Total Records", f"{total_records:,}")
        
        with col2:
            st.metric("🚗 Enter Congestion", f"{congested_enter:,}", 
                     f"{congested_enter/total_records*100:.1f}%")
        
        with col3:
            st.metric("🚦 Exit Congestion", f"{congested_exit:,}", 
                     f"{congested_exit/total_records*100:.1f}%")
        
        with col4:
            st.metric("📈 Avg Congestion", f"{avg_congestion:.1f}%")
        
        # Current status (simulated)
        st.subheader("🔴 Current Traffic Status")
        
        current_hour = datetime.now().hour
        is_rush_hour = current_hour in [7, 8, 9, 16, 17, 18]
        
        if is_rush_hour:
            st.markdown('<div class="alert-high">⚠️ RUSH HOUR - Heavy Traffic Expected</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-low">✅ Normal Traffic Conditions</div>', 
                       unsafe_allow_html=True)
        
        # Plots
        st.subheader("📊 Congestion Distribution")
        fig_dist = plot_congestion_distribution(filtered_df)
        st.pyplot(fig_dist)
    
    # TAB 2: Analytics
    with tab2:
        st.header("Traffic Analytics")
        
        # Hourly pattern
        st.subheader("⏰ Hourly Patterns")
        fig_hourly = plot_hourly_pattern(filtered_df)
        st.pyplot(fig_hourly)
        
        # Heatmap
        st.subheader("🗓️ Weekly Heatmap")
        fig_heatmap = plot_heatmap(filtered_df)
        st.pyplot(fig_heatmap)
        
        # Signal usage
        st.subheader("🚦 Signal Usage Analysis")
        signal_counts = filtered_df['signaling'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(signal_counts.values, labels=signal_counts.index, autopct='%1.1f%%',
                  startangle=90, colors=['#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
            ax.set_title('Signal Usage Distribution')
            st.pyplot(fig)
        
        with col2:
            st.dataframe(signal_counts.reset_index(), use_container_width=True)
    
    # TAB 3: Prediction
    with tab3:
        st.header("🔮 Real-time Congestion Prediction")
        
        if enter_model is None or exit_model is None:
            st.warning("⚠️ Prediction model not available. Please train a model first.")
            st.info("Run: `python traffic_ensemble.py` or `python hyperparameter_tuning.py`")
        else:
            st.success(f"✅ Model Loaded: {model_name}")
            
            st.subheader("Input Features")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                vehicle_count = st.slider("Vehicle Count", 0, 60, 25)
                avg_speed = st.slider("Avg Speed (km/h)", 0, 80, 35)
                traffic_density = st.slider("Traffic Density", 0.0, 1.0, 0.5)
            
            with col2:
                vehicle_variance = st.slider("Vehicle Variance", 0.0, 15.0, 5.0)
                speed_variance = st.slider("Speed Variance", 0.0, 20.0, 8.0)
                hour_input = st.slider("Hour", 0, 23, datetime.now().hour)
            
            with col3:
                is_rush_hour_input = st.checkbox("Rush Hour", 
                                                hour_input in [7, 8, 9, 16, 17, 18])
                day_of_week_input = st.selectbox("Day of Week", 
                                                 ['Monday', 'Tuesday', 'Wednesday', 
                                                  'Thursday', 'Friday', 'Saturday', 'Sunday'])
                is_weekend_input = day_of_week_input in ['Saturday', 'Sunday']
            
            # Create feature vector
            features = pd.DataFrame({
                'vehicle_count': [vehicle_count],
                'avg_speed': [avg_speed],
                'traffic_density': [traffic_density],
                'vehicle_variance': [vehicle_variance],
                'speed_variance': [speed_variance],
                'hour': [hour_input],
                'is_rush_hour': [int(is_rush_hour_input)],
                'day_of_week': [['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                'Friday', 'Saturday', 'Sunday'].index(day_of_week_input)],
                'is_weekend': [int(is_weekend_input)]
            })
            
            # Predict button
            if st.button("🎯 Predict Congestion", type="primary"):
                with st.spinner("Predicting..."):
                    enter_pred, exit_pred, enter_proba, exit_proba = predict_congestion(
                        enter_model, exit_model, features
                    )
                    
                    if enter_pred is not None:
                        st.subheader("Prediction Results")
                        
                        col1, col2 = st.columns(2)
                        
                        congestion_labels = ['free flowing', 'light delay', 'moderate delay', 'heavy delay']
                        congestion_colors = ['green', 'yellow', 'orange', 'red']
                        
                        with col1:
                            st.markdown("### 🚗 Enter Congestion")
                            pred_label = congestion_labels[enter_pred]
                            pred_color = congestion_colors[enter_pred]
                            st.markdown(f'<div class="alert-{pred_color}" style="background-color:{pred_color};">'
                                      f'{pred_label.upper()}</div>', unsafe_allow_html=True)
                            
                            if enter_proba is not None:
                                st.write("Confidence:")
                                for i, prob in enumerate(enter_proba):
                                    st.write(f"  {congestion_labels[i]}: {prob*100:.1f}%")
                        
                        with col2:
                            st.markdown("### 🚦 Exit Congestion")
                            pred_label = congestion_labels[exit_pred]
                            pred_color = congestion_colors[exit_pred]
                            st.markdown(f'<div class="alert-{pred_color}" style="background-color:{pred_color};">'
                                      f'{pred_label.upper()}</div>', unsafe_allow_html=True)
                            
                            if exit_proba is not None:
                                st.write("Confidence:")
                                for i, prob in enumerate(exit_proba):
                                    st.write(f"  {congestion_labels[i]}: {prob*100:.1f}%")
    
    # TAB 4: Data
    with tab4:
        st.header("📋 Raw Data")
        
        st.subheader(f"Filtered Data ({len(filtered_df)} records)")
        
        # Display options
        show_columns = st.multiselect(
            "Select Columns to Display",
            filtered_df.columns.tolist(),
            default=['datetime', 'congestion_enter_rating', 'congestion_exit_rating', 
                    'signaling', 'hour']
        )
        
        if show_columns:
            st.dataframe(filtered_df[show_columns], use_container_width=True, height=400)
        
        # Download button
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv,
            file_name=f"traffic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        f"📊 **Data Statistics**\n\n"
        f"Total Records: {len(df):,}\n\n"
        f"Date Range: {df['date'].min()} to {df['date'].max()}\n\n"
        f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


if __name__ == "__main__":
    main()
