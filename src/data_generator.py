import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker
import json

class HospitalDataGenerator:
    def __init__(self):
        self.fake = Faker()
        
        # Define diseases with HIGHER base rates for more visibility
        self.diseases = {
            'Influenza': {'base_rate': 0.20, 'seasonal': True, 'contagious': True},
            'COVID-19': {'base_rate': 0.15, 'seasonal': False, 'contagious': True},
            'Food Poisoning': {'base_rate': 0.10, 'seasonal': False, 'contagious': False},
            'Dengue': {'base_rate': 0.08, 'seasonal': True, 'contagious': False},
            'Measles': {'base_rate': 0.07, 'seasonal': False, 'contagious': True},
            'Common Cold': {'base_rate': 0.15, 'seasonal': True, 'contagious': True},
            'Gastroenteritis': {'base_rate': 0.12, 'seasonal': False, 'contagious': True},
            'Norovirus': {'base_rate': 0.08, 'seasonal': False, 'contagious': True},
            'Other': {'base_rate': 0.05, 'seasonal': False, 'contagious': False}
        }
        
        # Define US cities with coordinates
        self.cities = [
            {'name': 'New York', 'lat': 40.7128, 'lon': -74.0060, 'population': 8336817},
            {'name': 'Los Angeles', 'lat': 34.0522, 'lon': -118.2437, 'population': 3979576},
            {'name': 'Chicago', 'lat': 41.8781, 'lon': -87.6298, 'population': 2693976},
            {'name': 'Houston', 'lat': 29.7604, 'lon': -95.3698, 'population': 2320268},
            {'name': 'Phoenix', 'lat': 33.4484, 'lon': -112.0740, 'population': 1680992},
            {'name': 'Philadelphia', 'lat': 39.9526, 'lon': -75.1652, 'population': 1584064},
            {'name': 'San Antonio', 'lat': 29.4241, 'lon': -98.4936, 'population': 1547253},
            {'name': 'San Diego', 'lat': 32.7157, 'lon': -117.1611, 'population': 1423851},
            {'name': 'Dallas', 'lat': 32.7767, 'lon': -96.7970, 'population': 1343573},
            {'name': 'San Jose', 'lat': 37.3382, 'lon': -121.8863, 'population': 1021795},
            {'name': 'Austin', 'lat': 30.2672, 'lon': -97.7431, 'population': 978908},
            {'name': 'Jacksonville', 'lat': 30.3322, 'lon': -81.6557, 'population': 911507},
            {'name': 'Fort Worth', 'lat': 32.7555, 'lon': -97.3308, 'population': 909585},
            {'name': 'Columbus', 'lat': 39.9612, 'lon': -82.9988, 'population': 898553},
            {'name': 'Charlotte', 'lat': 35.2271, 'lon': -80.8431, 'population': 885708},
            {'name': 'San Francisco', 'lat': 37.7749, 'lon': -122.4194, 'population': 881549},
            {'name': 'Indianapolis', 'lat': 39.7684, 'lon': -86.1581, 'population': 876384},
            {'name': 'Seattle', 'lat': 47.6062, 'lon': -122.3321, 'population': 753675},
            {'name': 'Denver', 'lat': 39.7392, 'lon': -104.9903, 'population': 727211},
            {'name': 'Boston', 'lat': 42.3601, 'lon': -71.0589, 'population': 692600},
            {'name': 'Miami', 'lat': 25.7617, 'lon': -80.1918, 'population': 453579},
            {'name': 'Atlanta', 'lat': 33.7490, 'lon': -84.3880, 'population': 498715},
            {'name': 'Portland', 'lat': 45.5152, 'lon': -122.6784, 'population': 653115},
            {'name': 'Las Vegas', 'lat': 36.1699, 'lon': -115.1398, 'population': 651319},
            {'name': 'Detroit', 'lat': 42.3314, 'lon': -83.0458, 'population': 670031}
        ]
        
        self.hospitals_per_city = 5
        
    def generate_normal_data(self, start_date, days=30, records_per_day=800):
        """Generate normal (baseline) hospital data with higher baseline"""
        data = []
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            
            # MORE records per day for better visibility
            daily_records = records_per_day + random.randint(-100, 200)
            
            for _ in range(daily_records):
                city = random.choice(self.cities)
                
                # Add some noise to coordinates
                lat = city['lat'] + random.uniform(-0.1, 0.1)
                lon = city['lon'] + random.uniform(-0.1, 0.1)
                
                # Select disease based on base rates
                disease = self._select_disease(current_date, is_outbreak=False)
                
                # Generate patient data
                age = int(np.random.gamma(4, 15))
                age = min(max(age, 1), 95)
                
                record = {
                    'timestamp': current_date + timedelta(
                        hours=random.randint(0, 23),
                        minutes=random.randint(0, 59)
                    ),
                    'city': city['name'],
                    'latitude': lat,
                    'longitude': lon,
                    'hospital_id': f"{city['name']}_H{random.randint(1, self.hospitals_per_city)}",
                    'patient_id': self.fake.uuid4(),
                    'age': age,
                    'gender': random.choice(['M', 'F']),
                    'disease': disease,
                    'symptoms_severity': random.randint(1, 10),
                    'admission_type': random.choice(['Emergency', 'Urgent', 'Routine']),
                    'test_result': random.choice(['Positive', 'Negative', 'Pending'])
                }
                
                data.append(record)
        
        return pd.DataFrame(data)
    
    def inject_outbreak(self, normal_data, outbreak_config):
        """Inject an outbreak pattern into normal data"""
        outbreak_data = normal_data.copy()
        
        # Outbreak parameters
        disease = outbreak_config['disease']
        city = outbreak_config['city']
        start_day = outbreak_config['start_day']
        duration = outbreak_config['duration_days']
        intensity = outbreak_config['intensity']  # multiplier for cases
        
        # Find city coordinates
        city_info = next(c for c in self.cities if c['name'] == city)
        
        # Generate MORE outbreak cases for visibility
        outbreak_records = []
        
        for day in range(duration):
            current_date = normal_data['timestamp'].min() + timedelta(days=start_day + day)
            
            # STRONGER exponential growth
            if day < duration / 2:
                daily_cases = int(intensity * 30 * (1.8 ** day))  # Increased base and growth
            else:
                daily_cases = int(intensity * 30 * (0.6 ** (day - duration/2)))
            
            # Cap at reasonable maximum but keep it high
            daily_cases = min(daily_cases, 500)
            
            for _ in range(daily_cases):
                # Cluster cases around outbreak epicenter
                lat = city_info['lat'] + np.random.normal(0, 0.03)
                lon = city_info['lon'] + np.random.normal(0, 0.03)
                
                record = {
                    'timestamp': current_date + timedelta(
                        hours=random.randint(0, 23),
                        minutes=random.randint(0, 59)
                    ),
                    'city': city,
                    'latitude': lat,
                    'longitude': lon,
                    'hospital_id': f"{city}_H{random.randint(1, self.hospitals_per_city)}",
                    'patient_id': self.fake.uuid4(),
                    'age': int(np.random.gamma(4, 15)),
                    'gender': random.choice(['M', 'F']),
                    'disease': disease,
                    'symptoms_severity': random.randint(6, 10),  # Higher severity
                    'admission_type': random.choice(['Emergency', 'Emergency', 'Urgent']),
                    'test_result': 'Positive'  # Confirmed cases
                }
                
                outbreak_records.append(record)
        
        # Combine with normal data
        outbreak_df = pd.DataFrame(outbreak_records)
        combined_data = pd.concat([outbreak_data, outbreak_df], ignore_index=True)
        
        return combined_data.sort_values('timestamp').reset_index(drop=True)
    
    def generate_multiple_outbreaks(self, start_date, days=30):
        """Generate data with MANY outbreaks across different cities"""
        # Start with normal data
        normal_data = self.generate_normal_data(start_date, days, records_per_day=800)
        
        # Define MANY outbreaks - at least one in each major city
        outbreaks = [
            # Major outbreaks
            {'disease': 'Influenza', 'city': 'New York', 'start_day': 5, 'duration_days': 15, 'intensity': 4},
            {'disease': 'COVID-19', 'city': 'Los Angeles', 'start_day': 8, 'duration_days': 20, 'intensity': 5},
            {'disease': 'Gastroenteritis', 'city': 'Chicago', 'start_day': 10, 'duration_days': 12, 'intensity': 4},
            {'disease': 'Measles', 'city': 'Houston', 'start_day': 12, 'duration_days': 10, 'intensity': 3},
            {'disease': 'Food Poisoning', 'city': 'Phoenix', 'start_day': 15, 'duration_days': 5, 'intensity': 6},
            
            # Secondary outbreaks
            {'disease': 'Dengue', 'city': 'Miami', 'start_day': 3, 'duration_days': 14, 'intensity': 4},
            {'disease': 'Norovirus', 'city': 'Seattle', 'start_day': 7, 'duration_days': 8, 'intensity': 5},
            {'disease': 'Influenza', 'city': 'Boston', 'start_day': 9, 'duration_days': 12, 'intensity': 3},
            {'disease': 'COVID-19', 'city': 'San Francisco', 'start_day': 11, 'duration_days': 15, 'intensity': 4},
            {'disease': 'Common Cold', 'city': 'Denver', 'start_day': 14, 'duration_days': 10, 'intensity': 3},
            
            # Additional outbreaks for more coverage
            {'disease': 'Influenza', 'city': 'Atlanta', 'start_day': 6, 'duration_days': 11, 'intensity': 4},
            {'disease': 'Food Poisoning', 'city': 'Las Vegas', 'start_day': 18, 'duration_days': 4, 'intensity': 7},
            {'disease': 'COVID-19', 'city': 'Portland', 'start_day': 13, 'duration_days': 14, 'intensity': 3},
            {'disease': 'Gastroenteritis', 'city': 'San Diego', 'start_day': 16, 'duration_days': 7, 'intensity': 4},
            {'disease': 'Measles', 'city': 'Dallas', 'start_day': 20, 'duration_days': 9, 'intensity': 3},
            
            # Late-stage outbreaks
            {'disease': 'Influenza', 'city': 'Philadelphia', 'start_day': 22, 'duration_days': 8, 'intensity': 5},
            {'disease': 'Norovirus', 'city': 'San Antonio', 'start_day': 24, 'duration_days': 6, 'intensity': 4},
            {'disease': 'COVID-19', 'city': 'Detroit', 'start_day': 19, 'duration_days': 11, 'intensity': 4},
            {'disease': 'Dengue', 'city': 'Jacksonville', 'start_day': 21, 'duration_days': 7, 'intensity': 3},
            {'disease': 'Common Cold', 'city': 'Indianapolis', 'start_day': 23, 'duration_days': 5, 'intensity': 4}
        ]
        
        # Apply all outbreaks
        data_with_outbreaks = normal_data
        for outbreak in outbreaks:
            if outbreak['start_day'] < days:  # Only add if within date range
                data_with_outbreaks = self.inject_outbreak(data_with_outbreaks, outbreak)
                print(f"Added {outbreak['disease']} outbreak in {outbreak['city']}")
        
        return data_with_outbreaks
    
    def _select_disease(self, date, is_outbreak=False):
        """Select a disease based on probabilities"""
        month = date.month
        
        weights = []
        for disease, info in self.diseases.items():
            weight = info['base_rate']
            
            # Seasonal adjustment
            if info['seasonal']:
                if disease in ['Influenza', 'Common Cold']:
                    if month in [12, 1, 2]:
                        weight *= 2.5
                    elif month in [6, 7, 8]:
                        weight *= 0.5
                elif disease == 'Dengue':
                    if month in [6, 7, 8, 9]:
                        weight *= 3.0
                    elif month in [12, 1, 2]:
                        weight *= 0.3
            
            weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        
        return np.random.choice(list(self.diseases.keys()), p=weights)
    
    def generate_streaming_data(self, records_per_batch=100):
        """Generate real-time streaming data with periodic outbreaks"""
        batch_count = 0
        while True:
            batch = []
            
            # Every 5th batch, inject outbreak-like pattern
            is_outbreak_batch = batch_count % 5 == 0
            
            if is_outbreak_batch:
                # Pick a random city for mini-outbreak
                outbreak_city = random.choice(self.cities)
                outbreak_disease = random.choice(list(self.diseases.keys()))
                records_this_batch = records_per_batch * 3  # Triple the records
            else:
                records_this_batch = records_per_batch
            
            for _ in range(records_this_batch):
                if is_outbreak_batch:
                    city = outbreak_city
                    disease = outbreak_disease
                else:
                    city = random.choice(self.cities)
                    disease = self._select_disease(datetime.now())
                
                record = {
                    'timestamp': datetime.now(),
                    'city': city['name'],
                    'latitude': city['lat'] + random.uniform(-0.1, 0.1),
                    'longitude': city['lon'] + random.uniform(-0.1, 0.1),
                    'hospital_id': f"{city['name']}_H{random.randint(1, self.hospitals_per_city)}",
                    'patient_id': self.fake.uuid4(),
                    'age': int(np.random.gamma(4, 15)),
                    'gender': random.choice(['M', 'F']),
                    'disease': disease,
                    'symptoms_severity': random.randint(5, 10) if is_outbreak_batch else random.randint(1, 10),
                    'admission_type': random.choice(['Emergency', 'Urgent', 'Routine']),
                    'test_result': 'Positive' if is_outbreak_batch else random.choice(['Positive', 'Negative', 'Pending'])
                }
                batch.append(record)
            
            batch_count += 1
            yield pd.DataFrame(batch)

# Test the generator
if __name__ == "__main__":
    from datetime import datetime, timedelta
    
    generator = HospitalDataGenerator()
    
    # Generate data with MULTIPLE outbreaks
    print("Generating data with multiple outbreaks...")
    outbreak_data = generator.generate_multiple_outbreaks(
        start_date=datetime.now() - timedelta(days=30),
        days=30
    )
    
    print(f"\n📊 Generated {len(outbreak_data)} total records")
    
    # Show outbreak statistics
    print("\n🦠 Disease Distribution:")
    disease_counts = outbreak_data['disease'].value_counts()
    for disease, count in disease_counts.items():
        print(f"  {disease}: {count} cases")
    
    print("\n🏙️ City Distribution (Top 10):")
    city_counts = outbreak_data['city'].value_counts().head(10)
    for city, count in city_counts.items():
        print(f"  {city}: {count} cases")
    
    print("\n✅ Data generation complete!")