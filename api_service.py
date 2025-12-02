"""
FastAPI REST API Service for Traffic Congestion Prediction
Production-ready API with batch and single prediction endpoints
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
import uvicorn
import warnings
warnings.filterwarnings('ignore')

# Initialize FastAPI app
app = FastAPI(
    title="Traffic Congestion Prediction API",
    description="Real-time traffic congestion prediction for Norman Niles Roundabout",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global variables for model caching
MODELS = {
    'enter': None,
    'exit': None,
    'feature_names': None,
    'model_type': 'unknown'
}


# Pydantic models for request/response
class TrafficFeatures(BaseModel):
    """Single prediction features"""
    vehicle_count: float = Field(..., description="Number of vehicles", ge=0, le=100)
    avg_speed: float = Field(..., description="Average speed (km/h)", ge=0, le=100)
    traffic_density: float = Field(..., description="Traffic density", ge=0, le=1)
    vehicle_variance: float = Field(0.0, description="Vehicle count variance", ge=0)
    speed_variance: float = Field(0.0, description="Speed variance", ge=0)
    hour: int = Field(..., description="Hour of day", ge=0, le=23)
    is_rush_hour: int = Field(..., description="Rush hour flag (0 or 1)", ge=0, le=1)
    day_of_week: int = Field(..., description="Day of week (0=Mon, 6=Sun)", ge=0, le=6)
    is_weekend: int = Field(..., description="Weekend flag (0 or 1)", ge=0, le=1)

    class Config:
        json_schema_extra = {
            "example": {
                "vehicle_count": 25.0,
                "avg_speed": 35.0,
                "traffic_density": 0.5,
                "vehicle_variance": 5.0,
                "speed_variance": 8.0,
                "hour": 17,
                "is_rush_hour": 1,
                "day_of_week": 4,
                "is_weekend": 0
            }
        }


class PredictionResponse(BaseModel):
    """Single prediction response"""
    enter_congestion: str = Field(..., description="Enter congestion level")
    exit_congestion: str = Field(..., description="Exit congestion level")
    enter_confidence: float = Field(..., description="Enter prediction confidence")
    exit_confidence: float = Field(..., description="Exit prediction confidence")
    timestamp: str = Field(..., description="Prediction timestamp")
    model_type: str = Field(..., description="Model type used")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    features: List[TrafficFeatures] = Field(..., description="List of feature sets")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    total_samples: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_type: str
    timestamp: str


# Helper functions
def load_models():
    """Load trained models"""
    global MODELS
    
    # Try loading different model types
    model_paths = [
        ('voting_ensemble_enter_model.pkl', 'voting_ensemble_exit_model.pkl', 'Voting Ensemble'),
        ('stacking_ensemble_enter_model.pkl', 'stacking_ensemble_exit_model.pkl', 'Stacking Ensemble'),
        ('tuned_enter_model.pkl', 'tuned_exit_model.pkl', 'Tuned Model')
    ]
    
    for enter_path, exit_path, model_type in model_paths:
        if Path(enter_path).exists() and Path(exit_path).exists():
            MODELS['enter'] = joblib.load(enter_path)
            MODELS['exit'] = joblib.load(exit_path)
            MODELS['model_type'] = model_type
            
            # Load feature names if available
            metadata_path = enter_path.replace('enter', 'metadata')
            if Path(metadata_path).exists():
                metadata = joblib.load(metadata_path)
                MODELS['feature_names'] = metadata.get('feature_names')
            
            return True
    
    return False


def features_to_dataframe(features: TrafficFeatures) -> pd.DataFrame:
    """Convert TrafficFeatures to DataFrame"""
    feature_dict = features.dict()
    
    # Ensure correct column order
    columns = [
        'vehicle_count', 'avg_speed', 'traffic_density',
        'vehicle_variance', 'speed_variance', 'hour',
        'is_rush_hour', 'day_of_week', 'is_weekend'
    ]
    
    return pd.DataFrame([feature_dict])[columns]


def get_congestion_label(prediction: int) -> str:
    """Convert prediction index to label"""
    labels = {
        0: 'free flowing',
        1: 'light delay',
        2: 'moderate delay',
        3: 'heavy delay'
    }
    return labels.get(prediction, 'unknown')


# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    print("🚀 Starting Traffic Prediction API...")
    if load_models():
        print(f"✅ Models loaded: {MODELS['model_type']}")
    else:
        print("⚠️  No models found. Train a model first.")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Traffic Congestion Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if MODELS['enter'] is not None else "no_models",
        model_loaded=MODELS['enter'] is not None,
        model_type=MODELS['model_type'],
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(features: TrafficFeatures):
    """
    Single prediction endpoint
    
    Predicts congestion levels for given traffic features
    """
    if MODELS['enter'] is None or MODELS['exit'] is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Convert to DataFrame
        X = features_to_dataframe(features)
        
        # Predict
        enter_pred = MODELS['enter'].predict(X)[0]
        exit_pred = MODELS['exit'].predict(X)[0]
        
        # Get probabilities if available
        if hasattr(MODELS['enter'], 'predict_proba'):
            enter_proba = MODELS['enter'].predict_proba(X)[0]
            exit_proba = MODELS['exit'].predict_proba(X)[0]
            enter_conf = float(enter_proba[enter_pred])
            exit_conf = float(exit_proba[exit_pred])
        else:
            enter_conf = 1.0
            exit_conf = 1.0
        
        return PredictionResponse(
            enter_congestion=get_congestion_label(enter_pred),
            exit_congestion=get_congestion_label(exit_pred),
            enter_confidence=enter_conf,
            exit_confidence=exit_conf,
            timestamp=datetime.now().isoformat(),
            model_type=MODELS['model_type']
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Batch prediction endpoint
    
    Predicts congestion levels for multiple feature sets
    """
    if MODELS['enter'] is None or MODELS['exit'] is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    start_time = datetime.now()
    
    try:
        predictions = []
        
        for features in request.features:
            # Convert to DataFrame
            X = features_to_dataframe(features)
            
            # Predict
            enter_pred = MODELS['enter'].predict(X)[0]
            exit_pred = MODELS['exit'].predict(X)[0]
            
            # Get probabilities
            if hasattr(MODELS['enter'], 'predict_proba'):
                enter_proba = MODELS['enter'].predict_proba(X)[0]
                exit_proba = MODELS['exit'].predict_proba(X)[0]
                enter_conf = float(enter_proba[enter_pred])
                exit_conf = float(exit_proba[exit_pred])
            else:
                enter_conf = 1.0
                exit_conf = 1.0
            
            predictions.append(PredictionResponse(
                enter_congestion=get_congestion_label(enter_pred),
                exit_congestion=get_congestion_label(exit_pred),
                enter_confidence=enter_conf,
                exit_confidence=exit_conf,
                timestamp=datetime.now().isoformat(),
                model_type=MODELS['model_type']
            ))
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_samples=len(predictions),
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Get model information"""
    if MODELS['enter'] is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "model_type": MODELS['model_type'],
        "feature_names": MODELS.get('feature_names', []),
        "n_features": len(MODELS.get('feature_names', [])) if MODELS.get('feature_names') else 9,
        "loaded": True,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/model/reload")
async def reload_models(background_tasks: BackgroundTasks):
    """Reload models from disk"""
    background_tasks.add_task(load_models)
    return {"message": "Model reload initiated", "timestamp": datetime.now().isoformat()}


# Example usage documentation
@app.get("/examples")
async def get_examples():
    """Get API usage examples"""
    return {
        "single_prediction": {
            "endpoint": "POST /predict",
            "example_request": {
                "vehicle_count": 25.0,
                "avg_speed": 35.0,
                "traffic_density": 0.5,
                "vehicle_variance": 5.0,
                "speed_variance": 8.0,
                "hour": 17,
                "is_rush_hour": 1,
                "day_of_week": 4,
                "is_weekend": 0
            },
            "curl_command": 'curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d \'{"vehicle_count": 25, "avg_speed": 35, ...}\''
        },
        "batch_prediction": {
            "endpoint": "POST /predict/batch",
            "description": "Send multiple feature sets in one request"
        },
        "health_check": {
            "endpoint": "GET /health",
            "description": "Check API and model status"
        }
    }


# Run server
def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Start FastAPI server"""
    print(f"\n{'='*60}")
    print("🚀 STARTING TRAFFIC PREDICTION API")
    print(f"{'='*60}")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Docs: http://{host}:{port}/docs")
    print(f"{'='*60}\n")
    
    uvicorn.run(app, host=host, port=port, reload=reload)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Traffic Prediction API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    start_server(host=args.host, port=args.port, reload=args.reload)
