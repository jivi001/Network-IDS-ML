import requests
import json

url = 'http://localhost:8000/predict'
# Sample feature vector (Normal traffic-like)
# KDD99/NSL-KDD typically has 41 features.
# Indexes: 0=duration, 1=protocol_type, 2=service, 3=flag, 4=src_bytes, ...
features = [0] * 41
features[1] = 'tcp'   # protocol_type
features[2] = 'http'  # service
features[3] = 'SF'    # flag

payload = {'features': features}

try:
    response = requests.post(url, json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")
