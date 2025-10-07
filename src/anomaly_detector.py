import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class OutbreakDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {
            'isolation_forest': None,
            'statistical': None,
            'dbscan': None
        }
        self.thresholds = {
            'statistical_zscore': 3,
            'rate_change': 2.0,  # 200% increase
            'min_cases': 10      # Minimum cases to consider
        }
        
    def prepare_features(self, df):
        """Prepare features for anomaly detection"""
        # Aggregate by city and date
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        
        features = df.groupby(['city', 'date', 'disease']).agg({
            'patient_id': 'count',
            'symptoms_severity': 'mean',
            'latitude': 'mean',
            'longitude': 'mean'
        }).reset_index()
        
        features.columns = ['city', 'date', 'disease', 'case_count', 
                           'avg_severity', 'lat', 'lon']
        
        # Add temporal features
        features['day_of_week'] = pd.to_datetime(features['date']).dt.dayofweek
        features['month'] = pd.to_datetime(features['date']).dt.month
        
        # Calculate rolling statistics
        for window in [3, 7]:
            features[f'rolling_mean_{window}d'] = features.groupby(['city', 'disease'])['case_count'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            features[f'rolling_std_{window}d'] = features.groupby(['city', 'disease'])['case_count'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
        
        # Calculate rate of change
        features['rate_change'] = features.groupby(['city', 'disease'])['case_count'].transform(
            lambda x: x.pct_change(periods=1).fillna(0)
        )
        
        return features
    
    def detect_isolation_forest(self, features):
        """Detect anomalies using Isolation Forest"""
        # Prepare data
        feature_cols = ['case_count', 'avg_severity', 'rolling_mean_7d', 
                       'rolling_std_7d', 'rate_change']
        
        X = features[feature_cols].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest
        iso_forest = IsolationForest(
            contamination=0.1,  # Expected proportion of outliers
            random_state=42,
            n_estimators=100
        )
        
        # Predict anomalies (-1 for anomaly, 1 for normal)
        features['iso_forest_anomaly'] = iso_forest.fit_predict(X_scaled)
        features['iso_forest_score'] = iso_forest.score_samples(X_scaled)
        
        return features
    
    def detect_statistical(self, features):
        """Detect anomalies using statistical methods"""
        # Z-score based detection
        features['zscore'] = features.groupby(['city', 'disease'])['case_count'].transform(
            lambda x: np.abs(stats.zscore(x.fillna(x.mean())))
        )
        
        # Mark as anomaly if z-score exceeds threshold
        features['statistical_anomaly'] = (
            (features['zscore'] > self.thresholds['statistical_zscore']) & 
            (features['case_count'] > self.thresholds['min_cases'])
        ).astype(int)
        
        # Exponential Weighted Moving Average (EWMA) detection
        features['ewma'] = features.groupby(['city', 'disease'])['case_count'].transform(
            lambda x: x.ewm(span=7, adjust=False).mean()
        )
        
        features['ewma_deviation'] = np.abs(features['case_count'] - features['ewma'])
        
        return features
    
    def detect_clustering(self, features):
        """Detect spatial-temporal clusters using DBSCAN"""
        # Prepare spatial-temporal features
        features['date_numeric'] = (pd.to_datetime(features['date']) - 
                                   pd.to_datetime(features['date']).min()).dt.days
        
        # Normalize coordinates and time
        X_cluster = features[['lat', 'lon', 'date_numeric']].values
        X_cluster_scaled = StandardScaler().fit_transform(X_cluster)
        
        # Apply DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        features['cluster'] = dbscan.fit_predict(X_cluster_scaled)
        
        # Mark clusters with high case counts as anomalies
        cluster_stats = features.groupby('cluster')['case_count'].agg(['mean', 'sum'])
        anomaly_clusters = cluster_stats[
            cluster_stats['mean'] > features['case_count'].mean() * 2
        ].index
        
        features['cluster_anomaly'] = features['cluster'].isin(anomaly_clusters).astype(int)
        
        return features
    
    def detect_outbreaks(self, df):
        """Main method to detect outbreaks using multiple methods"""
        # Prepare features
        features = self.prepare_features(df)
        
        # Apply different detection methods
        features = self.detect_isolation_forest(features)
        features = self.detect_statistical(features)
        features = self.detect_clustering(features)
        
        # Combine detection results (ensemble approach)
        features['anomaly_score'] = (
            -features['iso_forest_anomaly'] +  # Convert to 0/1
            features['statistical_anomaly'] * 2 +  # Weight statistical higher
            features['cluster_anomaly']
        ) / 4  # Normalize to 0-1
        
        # Final outbreak detection
        features['is_outbreak'] = features['anomaly_score'] >= 0.5
        
        # Calculate outbreak severity
        features['severity'] = features.apply(self._calculate_severity, axis=1)
        
        return features
    
    def _calculate_severity(self, row):
        """Calculate outbreak severity level"""
        if not row['is_outbreak']:
            return 'None'
        
        if row['case_count'] > 100 or row['rate_change'] > 5:
            return 'Critical'
        elif row['case_count'] > 50 or row['rate_change'] > 3:
            return 'High'
        elif row['case_count'] > 20 or row['rate_change'] > 2:
            return 'Medium'
        else:
            return 'Low'
    
    def get_alerts(self, detection_results):
        """Generate alerts for detected outbreaks"""
        alerts = []
        
        outbreaks = detection_results[detection_results['is_outbreak']]
        
        for _, outbreak in outbreaks.iterrows():
            alert = {
                'timestamp': outbreak['date'],
                'city': outbreak['city'],
                'disease': outbreak['disease'],
                'case_count': outbreak['case_count'],
                'severity': outbreak['severity'],
                'anomaly_score': outbreak['anomaly_score'],
                'location': (outbreak['lat'], outbreak['lon']),
                'message': f"Potential {outbreak['disease']} outbreak in {outbreak['city']}: "
                          f"{outbreak['case_count']} cases detected"
            }
            alerts.append(alert)
        
        return alerts

# Test the detector
if __name__ == "__main__":
    from data_generator import HospitalDataGenerator
    
    # Generate test data
    generator = HospitalDataGenerator()
    normal_data = generator.generate_normal_data(
        start_date=datetime.now() - timedelta(days=30),
        days=30
    )
    
    # Inject outbreak
    outbreak_config = {
        'disease': 'Influenza',
        'city': 'New York',
        'start_day': 20,
        'duration_days': 10,
        'intensity': 3
    }
    
    test_data = generator.inject_outbreak(normal_data, outbreak_config)
    
    # Detect outbreaks
    detector = OutbreakDetector()
    results = detector.detect_outbreaks(test_data)
    
    # Get alerts
    alerts = detector.get_alerts(results)
    print(f"Detected {len(alerts)} potential outbreaks")
    
    for alert in alerts[:5]:
        print(f"\n{alert['severity']} Alert: {alert['message']}")
        print(f"  Score: {alert['anomaly_score']:.2f}")