"""
Test the frontend API with sample data
"""

import requests
import json

def test_frontend_api():
    """Test the Flask API with sample data"""
    
    base_url = 'http://127.0.0.1:5000'
    
    print("🧪 Testing Frontend API...")
    
    # Test 1: Health check
    try:
        response = requests.get(f'{base_url}/health')
        health_data = response.json()
        print(f"✅ Health Check: {health_data}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test 2: Sample data endpoint
    try:
        response = requests.get(f'{base_url}/sample_data')
        sample_data = response.json()
        print(f"✅ Sample data: Got {len(sample_data['sample_data'])} features")
    except Exception as e:
        print(f"❌ Sample data failed: {e}")
    
    # Test 3: Manual prediction with COVID-19 sample
    covid_input = "21.130000, 12.280000, 11.010000, 7.980000, 7.810000, 7.760000, 5.470000, 3.470000, 2.310000, 1.850000, 1.420000, 1.150000, 0.980000, 0.870000, 0.760000, 0.650000, 0.540000, 0.430000, 0.320000, 0.280000, 0.250000, 0.220000, 0.190000, 0.160000, 0.130000, 0.110000, 0.090000, 0.080000, 0.070000, 0.060000, 0.050000, 0.040000, 0.030000, 0.030000, 0.020000, 0.020000, 0.010000, 0.010000, 0.010000, 0.010000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000"
    
    try:
        response = requests.post(f'{base_url}/predict', data={'manual_input': covid_input})
        if response.status_code == 200:
            result = response.json()
            print(f"✅ COVID-19 Sample Prediction:")
            if 'results' in result:
                for sample in result['results']:
                    print(f"   Sample {sample['sample_id']}:")
                    for pred in sample['predictions']:
                        print(f"     • {pred['disease']}: {pred['probability']}%")
            else:
                print(f"   Raw result: {result}")
        else:
            print(f"❌ Prediction failed with status {response.status_code}: {response.text}")
    except Exception as e:
        print(f"❌ Manual prediction failed: {e}")
    
    # Test 4: File upload (simulate)
    print(f"\n📄 CSV File Test:")
    print(f"   Upload 'sample_1_covid-19.csv' through the web interface")
    print(f"   Expected: COVID-19 as top prediction")
    
    print(f"\n🌐 Frontend is ready at: {base_url}")
    print(f"   • Model loaded: {health_data.get('model_loaded', False)}")
    print(f"   • Disease classes: {health_data.get('disease_classes', 0)}")

if __name__ == '__main__':
    test_frontend_api()