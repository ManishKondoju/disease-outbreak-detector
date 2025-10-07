import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
from src.data_generator import HospitalDataGenerator
from src.anomaly_detector import OutbreakDetector

def train_models():
    """Train and save anomaly detection models"""
    print("="*50)
    print("🦠 OUTBREAK DETECTION SYSTEM - MODEL TRAINING")
    print("="*50)
    
    # Generate training data
    print("\n📊 Generating training data...")
    generator = HospitalDataGenerator()
    
    # Generate normal data
    normal_data = generator.generate_normal_data(
        start_date=datetime.now() - timedelta(days=90),
        days=60,
        records_per_day=500
    )
    print(f"✅ Generated {len(normal_data)} normal records")
    
    # Inject various outbreaks for training
    print("\n💉 Injecting synthetic outbreaks...")
    outbreaks = [
        {'disease': 'Influenza', 'city': 'New York', 'start_day': 30, 'duration_days': 10, 'intensity': 3},
        {'disease': 'COVID-19', 'city': 'Los Angeles', 'start_day': 40, 'duration_days': 14, 'intensity': 2},
        {'disease': 'Food Poisoning', 'city': 'Chicago', 'start_day': 50, 'duration_days': 3, 'intensity': 5},
        {'disease': 'Dengue', 'city': 'Houston', 'start_day': 35, 'duration_days': 7, 'intensity': 2},
        {'disease': 'Measles', 'city': 'Phoenix', 'start_day': 45, 'duration_days': 10, 'intensity': 3}
    ]
    
    training_data = normal_data
    for outbreak in outbreaks:
        training_data = generator.inject_outbreak(training_data, outbreak)
        print(f"  ✅ Added {outbreak['disease']} outbreak in {outbreak['city']}")
    
    # Save raw training data
    os.makedirs('data/processed', exist_ok=True)
    training_data.to_csv('data/processed/training_data.csv', index=False)
    print(f"\n💾 Saved training data: {len(training_data)} records")
    
    # Train detector
    print("\n🤖 Training anomaly detection models...")
    detector = OutbreakDetector()
    
    # Prepare features and train
    detection_results = detector.detect_outbreaks(training_data)
    
    # Calculate performance metrics
    print("\n📊 Performance Metrics:")
    total_outbreaks = detection_results['is_outbreak'].sum()
    total_records = len(detection_results)
    print(f"  Total aggregated records: {total_records}")
    print(f"  Detected outbreaks: {total_outbreaks}")
    print(f"  Detection rate: {(total_outbreaks/total_records)*100:.2f}%")
    
    # Analyze by disease
    print("\n🦠 Detection by Disease:")
    disease_stats = detection_results[detection_results['is_outbreak']].groupby('disease')['case_count'].sum()
    for disease, count in disease_stats.items():
        print(f"  {disease}: {count} cases")
    
    # Save models and configurations
    print("\n💾 Saving models and configurations...")
    os.makedirs('models', exist_ok=True)
    
    # Save the detector object
    joblib.dump(detector, 'models/outbreak_detector.pkl')
    
    # Save thresholds and parameters
    config = {
        'thresholds': detector.thresholds,
        'training_date': datetime.now().isoformat(),
        'training_records': len(training_data),
        'detected_outbreaks': total_outbreaks
    }
    joblib.dump(config, 'models/config.pkl')
    
    print("✅ Models saved successfully!")
    
    # Generate alerts for testing
    alerts = detector.get_alerts(detection_results)
    print(f"\n🚨 Sample Alerts Generated: {len(alerts)} alerts")
    
    if len(alerts) > 0:
        print("\nTop 3 Alerts:")
        for alert in alerts[:3]:
            print(f"  - {alert['severity']}: {alert['message']}")
    
    print("\n" + "="*50)
    print("✅ TRAINING COMPLETE!")
    print("="*50)
    print("\n📌 Next step: Run 'streamlit run app/streamlit_app.py'")
    
    return detector, detection_results

if __name__ == "__main__":
    train_models()