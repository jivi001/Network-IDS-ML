"""
REST API for NIDS inference service.
"""

from flask import Flask, request, jsonify
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nids.pipelines import InferencePipeline

app = Flask(__name__)

# Initialize inference pipeline
MODEL_VERSION = os.getenv('MODEL_VERSION', 'v1.0.0')
pipeline = InferencePipeline(model_version=MODEL_VERSION)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_version': MODEL_VERSION
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict attack type for network traffic sample.
    
    Request Body:
    {
        "features": [0.5, 1.2, ..., 3.4]  # Feature vector
    }
    
    Response:
    {
        "prediction": "DoS",
        "confidence": 0.95,
        "tier_used": 1,
        "anomaly_score": -0.12
    }
    """
    try:
        data = request.json
        
        if 'features' not in data:
            return jsonify({'error': 'Missing features field'}), 400
        
        features = data['features']
        result = pipeline.predict_single(features)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint.
    
    Request Body:
    {
        "features": [[...], [...], ...]  # List of feature vectors
    }
    """
    try:
        data = request.json
        
        if 'features' not in data:
            return jsonify({'error': 'Missing features field'}), 400
        
        features = np.array(data['features'])
        result = pipeline.predict_batch(features)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
