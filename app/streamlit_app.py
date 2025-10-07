import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import time
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_generator import HospitalDataGenerator
from src.anomaly_detector import OutbreakDetector

# Page configuration
st.set_page_config(
    page_title="Disease Outbreak Detection System",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .alert-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        animation: pulse 2s infinite;
    }
    .critical { 
        background: linear-gradient(135deg, #ff4444, #cc0000); 
        color: white; 
        font-weight: bold;
        border: 2px solid #ff0000;
    }
    .high { 
        background: linear-gradient(135deg, #ff8800, #ff6600); 
        color: white; 
        font-weight: bold;
    }
    .medium { 
        background: linear-gradient(135deg, #ffbb33, #ff9900); 
        color: black; 
    }
    .low { 
        background: linear-gradient(135deg, #00C851, #00aa44); 
        color: white; 
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'streaming' not in st.session_state:
    st.session_state.streaming = False
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = None

def generate_data():
    """Generate data with MULTIPLE visible outbreaks"""
    generator = HospitalDataGenerator()
    
    # Use the enhanced method that creates many outbreaks
    with st.spinner("🦠 Injecting multiple outbreak patterns across 20+ cities..."):
        data = generator.generate_multiple_outbreaks(
            start_date=datetime.now() - timedelta(days=30),
            days=30
        )
    
    # Show summary
    st.success(f"✅ Generated {len(data):,} records with 20+ outbreak patterns!")
    
    return data

def create_outbreak_map(detection_results):
    """Create interactive map with outbreak locations"""
    # Get outbreak data
    outbreaks = detection_results[detection_results['is_outbreak']].copy()
    
    # Create base map centered on US
    m = folium.Map(
        location=[39.8283, -98.5795], 
        zoom_start=4,
        tiles='CartoDB dark_matter'  # Dark theme for better contrast
    )
    
    if len(outbreaks) == 0:
        # No outbreaks - add single marker
        folium.Marker(
            [39.8283, -98.5795],
            popup="No outbreaks detected",
            icon=folium.Icon(color='green', icon='check', prefix='fa')
        ).add_to(m)
    else:
        # Group by city and severity for better visualization
        city_outbreaks = outbreaks.groupby(['city', 'lat', 'lon', 'severity']).agg({
            'case_count': 'sum',
            'disease': lambda x: ', '.join(x.unique()),
            'anomaly_score': 'mean'
        }).reset_index()
        
        # Sort by case count to add biggest outbreaks last (on top)
        city_outbreaks = city_outbreaks.sort_values('case_count')
        
        # Color and size mapping
        color_map = {
            'Critical': '#ff0000',
            'High': '#ff6600',
            'Medium': '#ffaa00',
            'Low': '#00ff00',
            'None': '#0099ff'
        }
        
        icon_map = {
            'Critical': 'exclamation-triangle',
            'High': 'exclamation-circle',
            'Medium': 'exclamation',
            'Low': 'info-circle'
        }
        
        # Add markers and circles for each outbreak
        for _, outbreak in city_outbreaks.iterrows():
            color = color_map.get(outbreak['severity'], '#0099ff')
            icon = icon_map.get(outbreak['severity'], 'info')
            
            # Calculate radius based on case count (more visible)
            radius = min(outbreak['case_count'] * 500, 100000)  # Increased multiplier
            
            # Create detailed popup
            popup_html = f"""
            <div style='font-family: Arial; width: 250px;'>
                <h4 style='color: {color}; margin: 5px 0;'>{outbreak['city']}</h4>
                <hr style='margin: 5px 0;'>
                <b>Severity:</b> <span style='color: {color}'>{outbreak['severity']}</span><br>
                <b>Total Cases:</b> {outbreak['case_count']:,}<br>
                <b>Diseases:</b> {outbreak['disease']}<br>
                <b>Anomaly Score:</b> {outbreak['anomaly_score']:.2f}<br>
                <hr style='margin: 5px 0;'>
                <small>Click circle to see affected area</small>
            </div>
            """
            
            # Add pulsing circle for affected area
            folium.Circle(
                location=[outbreak['lat'], outbreak['lon']],
                radius=radius,
                popup=f"Affected area: {outbreak['case_count']:,} cases",
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.4,
                weight=2,
                tooltip=f"{outbreak['city']}: {outbreak['case_count']:,} cases"
            ).add_to(m)
            
            # Add marker with custom icon
            folium.Marker(
                location=[outbreak['lat'], outbreak['lon']],
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(
                    color='red' if outbreak['severity'] == 'Critical' else 
                          'orange' if outbreak['severity'] == 'High' else
                          'yellow' if outbreak['severity'] == 'Medium' else 'green',
                    icon=icon,
                    prefix='fa'
                ),
                tooltip=f"{outbreak['city']}: {outbreak['severity']} outbreak"
            ).add_to(m)
        
        # Add heatmap layer for density
        from folium.plugins import HeatMap
        heat_data = [[row['lat'], row['lon'], row['case_count']] 
                     for _, row in outbreaks.iterrows()]
        HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(m)
    
    return m

def create_time_series_plot(detection_results):
    """Create enhanced time series visualization"""
    # Aggregate by date and severity
    daily_data = detection_results.groupby(['date', 'severity']).agg({
        'case_count': 'sum'
    }).reset_index()
    
    # Pivot for stacked area chart
    daily_pivot = daily_data.pivot(index='date', columns='severity', values='case_count').fillna(0)
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Total Cases by Severity Over Time", "Daily New Cases"),
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3]
    )
    
    # Define colors
    colors = {
        'Critical': '#ff0000',
        'High': '#ff6600',
        'Medium': '#ffaa00',
        'Low': '#00ff00',
        'None': '#0099ff'
    }
    
    # Add stacked area chart
    for severity in ['None', 'Low', 'Medium', 'High', 'Critical']:
        if severity in daily_pivot.columns:
            fig.add_trace(
                go.Scatter(
                    x=daily_pivot.index,
                    y=daily_pivot[severity],
                    name=severity,
                    mode='lines',
                    stackgroup='one',
                    fillcolor=colors.get(severity, '#999999'),
                    line=dict(width=0.5, color=colors.get(severity, '#999999'))
                ),
                row=1, col=1
            )
    
    # Add daily new cases
    daily_total = detection_results.groupby('date')['case_count'].sum().reset_index()
    daily_total['new_cases'] = daily_total['case_count'].diff().fillna(0)
    
    fig.add_trace(
        go.Bar(
            x=daily_total['date'],
            y=daily_total['new_cases'],
            name='New Cases',
            marker_color='lightblue'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Cases", row=1, col=1)
    fig.update_yaxes(title_text="New Cases", row=2, col=1)
    
    fig.update_layout(
        height=600,
        showlegend=True,
        hovermode='x unified',
        title_text="Outbreak Progression Analysis"
    )
    
    return fig

def create_disease_distribution(detection_results):
    """Create enhanced disease distribution chart"""
    # Get top diseases by outbreak status
    disease_data = detection_results.groupby(['disease', 'is_outbreak']).agg({
        'case_count': 'sum'
    }).reset_index()
    
    # Create grouped bar chart
    fig = px.bar(
        disease_data,
        x='case_count',
        y='disease',
        color='is_outbreak',
        orientation='h',
        color_discrete_map={True: '#ff4444', False: '#4444ff'},
        labels={'is_outbreak': 'Outbreak Status', 'case_count': 'Total Cases'},
        title='Disease Distribution: Outbreak vs Normal Cases'
    )
    
    fig.update_layout(
        height=400,
        showlegend=True,
        legend_title_text='Outbreak Detected'
    )
    
    return fig

def create_severity_gauge(severity_score):
    """Create enhanced severity gauge"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=severity_score,
        title={'text': "System Risk Level", 'font': {'size': 24}},
        delta={'reference': 30, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue", 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 25], 'color': '#00ff00'},
                {'range': [25, 50], 'color': '#ffff00'},
                {'range': [50, 75], 'color': '#ff9900'},
                {'range': [75, 100], 'color': '#ff0000'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def main():
    # Header with animation
    st.markdown("""
        <h1 style='text-align: center; color: #ff4444;'>
            🦠 Disease Outbreak Detection System
        </h1>
        <p style='text-align: center; color: #888;'>
            Real-time monitoring and early warning system for disease outbreaks
        </p>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Control Panel")
        
        # Data generation with emphasis
        st.markdown("### 📊 Data Generation")
        if st.button("🔄 Generate New Data with Outbreaks", 
                    type="primary", 
                    use_container_width=True,
                    help="Generate data with 20+ outbreak patterns across US cities"):
            with st.spinner("Generating data with multiple outbreaks..."):
                st.session_state.data = generate_data()
        
        # Detection settings
        st.markdown("### 🎯 Detection Settings")
        
        contamination = st.slider(
            "Anomaly Sensitivity",
            min_value=0.05,
            max_value=0.3,
            value=0.15,  # Higher default for more detections
            step=0.05,
            help="Higher = More sensitive (more detections)"
        )
        
        min_cases = st.number_input(
            "Minimum Cases for Alert",
            min_value=3,
            max_value=50,
            value=5,  # Lower default
            help="Lower = More alerts"
        )
        
        # Real-time streaming
        st.markdown("### 📡 Real-time Mode")
        streaming = st.checkbox("Enable Real-time Streaming", value=False)
        
        if streaming:
            refresh_rate = st.slider(
                "Refresh Rate (seconds)",
                min_value=1,
                max_value=10,
                value=5
            )
        
        # System status with metrics
        st.markdown("### 📊 System Status")
        if st.session_state.data is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Records", f"{len(st.session_state.data):,}")
                st.metric("Cities", st.session_state.data['city'].nunique())
            with col2:
                st.metric("Diseases", st.session_state.data['disease'].nunique())
                st.metric("🚨 Alerts", len(st.session_state.alerts))
        else:
            st.info("No data loaded. Click 'Generate New Data' to start.")
        
        # Quick stats
        if st.session_state.data is not None:
            st.markdown("### 🏥 Top Affected Cities")
            top_cities = st.session_state.data['city'].value_counts().head(5)
            for city, count in top_cities.items():
                st.caption(f"{city}: {count:,} cases")
    
    # Main content area
    if st.session_state.data is not None:
        # Run detection
        detector = OutbreakDetector()
        detector.thresholds['min_cases'] = min_cases
        
        with st.spinner("🔍 Analyzing data for outbreaks..."):
            detection_results = detector.detect_outbreaks(st.session_state.data)
            st.session_state.detection_results = detection_results
            st.session_state.alerts = detector.get_alerts(detection_results)
        
        # Alert Section with enhanced styling
        if len(st.session_state.alerts) > 0:
            st.markdown(f"""
                <div style='padding: 1rem; background: linear-gradient(135deg, #ff4444, #cc0000); 
                            border-radius: 10px; color: white; margin-bottom: 1rem;'>
                    <h2 style='margin: 0;'>⚠️ {len(st.session_state.alerts)} Potential Outbreaks Detected!</h2>
                    <p style='margin: 0.5rem 0 0 0;'>Immediate attention required for critical outbreaks</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Display top alerts in grid
            alert_cols = st.columns(min(3, len(st.session_state.alerts)))
            for idx, alert in enumerate(st.session_state.alerts[:6]):  # Show more alerts
                with alert_cols[idx % 3]:
                    severity_class = alert['severity'].lower()
                    st.markdown(f"""
                        <div class="alert-box {severity_class}">
                            <h4 style='margin: 0;'>{alert['city']}</h4>
                            <hr style='margin: 5px 0; opacity: 0.5;'>
                            <b>🦠 Disease:</b> {alert['disease']}<br>
                            <b>📊 Cases:</b> {alert['case_count']:,}<br>
                            <b>⚠️ Severity:</b> {alert['severity']}<br>
                            <b>📈 Score:</b> {alert['anomaly_score']:.2f}
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.success("✅ No significant outbreaks detected. System operating normally.")
        
        # Enhanced Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🗺️ Outbreak Map", 
            "📈 Time Analysis", 
            "📊 Disease Analytics", 
            "🏙️ City Analysis",
            "📋 Alert Details"
        ])
        
        with tab1:
            st.subheader("Geographic Distribution of Outbreaks")
            
            # Add summary metrics above map
            map_cols = st.columns(4)
            outbreak_cities = detection_results[detection_results['is_outbreak']]['city'].nunique()
            total_outbreak_cases = detection_results[detection_results['is_outbreak']]['case_count'].sum()
            critical_count = len([a for a in st.session_state.alerts if a['severity'] == 'Critical'])
            high_count = len([a for a in st.session_state.alerts if a['severity'] == 'High'])
            
            with map_cols[0]:
                st.metric("Affected Cities", outbreak_cities)
            with map_cols[1]:
                st.metric("Outbreak Cases", f"{total_outbreak_cases:,}")
            with map_cols[2]:
                st.metric("Critical Alerts", critical_count)
            with map_cols[3]:
                st.metric("High Alerts", high_count)
            
            # Display map
            outbreak_map = create_outbreak_map(detection_results)
            st_folium(outbreak_map, height=600, width=None, returned_objects=[])
            
            # Legend
            st.markdown("""
                **Map Legend:**
                - 🔴 **Critical**: Immediate action required (100+ cases)
                - 🟠 **High**: Significant outbreak (50-100 cases)
                - 🟡 **Medium**: Monitoring required (20-50 cases)
                - 🟢 **Low**: Minor anomaly (10-20 cases)
                - Heat overlay shows outbreak density
            """)
        
        with tab2:
            st.subheader("Temporal Analysis")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                time_series_fig = create_time_series_plot(detection_results)
                st.plotly_chart(time_series_fig, use_container_width=True)
            
            with col2:
                st.subheader("Risk Assessment")
                
                # Calculate risk score
                if len(st.session_state.alerts) > 0:
                    critical_alerts = len([a for a in st.session_state.alerts if a['severity'] == 'Critical'])
                    high_alerts = len([a for a in st.session_state.alerts if a['severity'] == 'High'])
                    risk_score = min(critical_alerts * 20 + high_alerts * 10 + len(st.session_state.alerts) * 2, 100)
                else:
                    risk_score = 0
                
                gauge_fig = create_severity_gauge(risk_score)
                st.plotly_chart(gauge_fig, use_container_width=True)
                
                # Risk interpretation
                if risk_score >= 75:
                    st.error("🚨 CRITICAL RISK LEVEL")
                elif risk_score >= 50:
                    st.warning("⚠️ HIGH RISK LEVEL")
                elif risk_score >= 25:
                    st.warning("⚠️ MEDIUM RISK LEVEL")
                else:
                    st.success("✅ LOW RISK LEVEL")
        
        with tab3:
            st.subheader("Disease Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                disease_fig = create_disease_distribution(detection_results)
                st.plotly_chart(disease_fig, use_container_width=True)
            
            with col2:
                # Disease severity heatmap
                disease_severity = detection_results.pivot_table(
                    index='disease',
                    columns='severity',
                    values='case_count',
                    aggfunc='sum',
                    fill_value=0
                )
                
                fig_heatmap = px.imshow(
                    disease_severity,
                    labels=dict(x="Severity", y="Disease", color="Cases"),
                    title="Disease-Severity Heatmap",
                    color_continuous_scale="Reds"
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with tab4:
            st.subheader("City-Level Analysis")
            
            # City statistics
            city_stats = detection_results.groupby('city').agg({
                'case_count': 'sum',
                'is_outbreak': 'any',
                'severity': lambda x: x[x != 'None'].mode()[0] if len(x[x != 'None']) > 0 else 'None'
            }).sort_values('case_count', ascending=False).head(15)
            
            # Create city bar chart
            fig_city = go.Figure()
            
            for severity in ['Critical', 'High', 'Medium', 'Low', 'None']:
                cities_with_severity = city_stats[city_stats['severity'] == severity]
                if len(cities_with_severity) > 0:
                    color = {'Critical': '#ff0000', 'High': '#ff6600', 
                            'Medium': '#ffaa00', 'Low': '#00ff00', 'None': '#0099ff'}[severity]
                    
                    fig_city.add_trace(go.Bar(
                        y=cities_with_severity.index,
                        x=cities_with_severity['case_count'],
                        name=severity,
                        orientation='h',
                        marker_color=color
                    ))
            
            fig_city.update_layout(
                title='Top 15 Cities by Case Count',
                xaxis_title='Total Cases',
                yaxis_title='City',
                height=500,
                barmode='stack'
            )
            
            st.plotly_chart(fig_city, use_container_width=True)
            
            # City metrics
            st.subheader("📊 City Statistics")
            metric_cols = st.columns(4)
            
            with metric_cols[0]:
                total_cities = detection_results['city'].nunique()
                st.metric("Total Cities", total_cities)
            
            with metric_cols[1]:
                affected_cities = detection_results[detection_results['is_outbreak']]['city'].nunique()
                st.metric("Cities with Outbreaks", affected_cities)
            
            with metric_cols[2]:
                avg_cases = detection_results.groupby('city')['case_count'].sum().mean()
                st.metric("Avg Cases per City", f"{avg_cases:.0f}")
            
            with metric_cols[3]:
                max_city = city_stats.iloc[0].name if len(city_stats) > 0 else "N/A"
                max_cases = city_stats.iloc[0]['case_count'] if len(city_stats) > 0 else 0
                st.metric("Most Affected", f"{max_city} ({max_cases:,})")
        
        with tab5:
            st.subheader("📋 Detailed Alert Information")
            
            if len(st.session_state.alerts) > 0:
                # Summary stats
                st.markdown("### Alert Summary")
                summary_cols = st.columns(4)
                
                severity_counts = pd.DataFrame(st.session_state.alerts)['severity'].value_counts()
                for col, (severity, count) in zip(summary_cols, severity_counts.items()):
                    with col:
                        st.metric(f"{severity} Alerts", count)
                
                # Alert table
                st.markdown("### All Alerts")
                alerts_df = pd.DataFrame(st.session_state.alerts)
                alerts_df = alerts_df.sort_values('anomaly_score', ascending=False)
                
                # Format the dataframe for display
                alerts_display = alerts_df[['timestamp', 'city', 'disease', 'case_count', 'severity', 'anomaly_score']].copy()
                alerts_display['timestamp'] = pd.to_datetime(alerts_display['timestamp']).dt.strftime('%Y-%m-%d')
                alerts_display['anomaly_score'] = alerts_display['anomaly_score'].round(3)
                
                # Color code by severity
                def highlight_severity(row):
                    color_map = {
                        'Critical': 'background-color: #ff4444; color: white',
                        'High': 'background-color: #ff8800; color: white',
                        'Medium': 'background-color: #ffbb33; color: black',
                        'Low': 'background-color: #00C851; color: white'
                    }
                    return [color_map.get(row['severity'], '')] * len(row)
                
                styled_df = alerts_display.style.apply(highlight_severity, axis=1)
                st.dataframe(styled_df, use_container_width=True, height=400)
                
                # Download button
                csv = alerts_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Alert Report (CSV)",
                    data=csv,
                    file_name=f"outbreak_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No alerts to display. System is operating normally.")
        
        # Auto-refresh for streaming mode
        if streaming:
            time.sleep(refresh_rate)
            st.rerun()
    
    else:
        # Welcome screen with instructions
        st.info("👈 Click 'Generate New Data with Outbreaks' in the sidebar to start the detection system")
        
        # Display system capabilities
        st.markdown("""
        ### 🎯 System Capabilities
        
        This advanced outbreak detection system features:
        
        #### 🤖 Detection Algorithms
        - **Isolation Forest**: Multivariate anomaly detection
        - **Statistical Analysis**: Z-score and EWMA-based methods
        - **DBSCAN Clustering**: Geographic outbreak identification
        - **Ensemble Approach**: Combined detection for accuracy
        
        #### 📊 Real-time Monitoring
        - Tracks 20+ major US cities simultaneously
        - Monitors 8+ different diseases
        - Processes 25,000+ patient records
        - Generates alerts with severity classification
        
        #### 🗺️ Visualization Features
        - Interactive outbreak map with heatmap overlay
        - Time series analysis with trend detection
        - City and disease-level analytics
        - Risk assessment gauges
        
        #### 🚨 Alert System
        - **Critical**: Immediate action required (100+ cases)
        - **High**: Significant outbreak (50-100 cases)
        - **Medium**: Monitoring required (20-50 cases)
        - **Low**: Minor anomaly detected (10-20 cases)
        
        ### 📋 Getting Started
        1. Click **'Generate New Data with Outbreaks'** to create synthetic outbreak data
        2. System will automatically detect and classify outbreaks
        3. Explore the different tabs to analyze outbreak patterns
        4. Download alert reports for further analysis
        """)

if __name__ == "__main__":
    main()