"""
FastAPI application for IRIS classification prediction.
Loads model from MLflow and serves predictions via REST API.
Enhanced with structured logging and distributed tracing for Google Cloud.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict
import numpy as np
import logging
import json
import time
from datetime import datetime
import os

# Import tracing setup
from src.tracing import setup_tracing, get_tracer

# Configure structured logging for Google Cloud
class StructuredLogger:
    """Custom logger for structured JSON logging compatible with Google Cloud Logging"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # Create console handler with structured formatter
        handler = logging.StreamHandler()
        handler.setFormatter(self._get_formatter())
        self.logger.addHandler(handler)
    
    def _get_formatter(self):
        """Return a formatter that outputs structured JSON logs"""
        class StructuredFormatter(logging.Formatter):
            def format(self, record):
                log_obj = {
                    "severity": record.levelname,
                    "message": record.getMessage(),
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "component": record.name,
                }
                
                # Add trace context if available
                from opentelemetry import trace
                span = trace.get_current_span()
                if span.is_recording():
                    span_context = span.get_span_context()
                    log_obj['logging.googleapis.com/trace'] = f"projects/{os.getenv('GCP_PROJECT_ID', 'unknown')}/traces/{format(span_context.trace_id, '032x')}"
                    log_obj['logging.googleapis.com/spanId'] = format(span_context.span_id, '016x')
                
                # Add extra fields if present
                if hasattr(record, 'request_id'):
                    log_obj['request_id'] = record.request_id
                if hasattr(record, 'latency_ms'):
                    log_obj['latency_ms'] = record.latency_ms
                if hasattr(record, 'prediction'):
                    log_obj['prediction'] = record.prediction
                if hasattr(record, 'status_code'):
                    log_obj['status_code'] = record.status_code
                if hasattr(record, 'endpoint'):
                    log_obj['endpoint'] = record.endpoint
                
                return json.dumps(log_obj)
        
        return StructuredFormatter()
    
    def info(self, message, **kwargs):
        """Log info message with optional structured fields"""
        extra_dict = {k: v for k, v in kwargs.items()}
        self.logger.info(message, extra=extra_dict)
    
    def warning(self, message, **kwargs):
        """Log warning message with optional structured fields"""
        extra_dict = {k: v for k, v in kwargs.items()}
        self.logger.warning(message, extra=extra_dict)
    
    def error(self, message, **kwargs):
        """Log error message with optional structured fields"""
        extra_dict = {k: v for k, v in kwargs.items()}
        self.logger.error(message, extra=extra_dict)
    
    def exception(self, message, **kwargs):
        """Log exception with optional structured fields"""
        extra_dict = {k: v for k, v in kwargs.items()}
        self.logger.exception(message, extra=extra_dict)


# Initialize structured logger
logger = StructuredLogger("iris-api")

# Initialize FastAPI app
app = FastAPI(
    title="IRIS Classification API",
    description="ML-powered API for classifying IRIS flowers using MLflow models",
    version="1.0.0"
)

# Global model variable
model = None
model_info = {}
request_counter = 0


class IrisFeatures(BaseModel):
    """Input features for IRIS prediction"""
    sepal_length: float = Field(..., ge=0, le=10, description="Sepal length in cm")
    sepal_width: float = Field(..., ge=0, le=10, description="Sepal width in cm")
    petal_length: float = Field(..., ge=0, le=10, description="Petal length in cm")
    petal_width: float = Field(..., ge=0, le=10, description="Petal width in cm")
    
    class Config:
        schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    model_version: str
    timestamp: str
    request_id: str


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    model_loaded: bool
    model_info: Dict


def load_default_model():
    """Train and load a simple default model as fallback"""
    global model, model_info
    
    tracer = get_tracer()
    with tracer.start_as_current_span("load_default_model"):
        try:
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.datasets import load_iris
            
            logger.info("Training default model...")
            iris = load_iris()
            model = DecisionTreeClassifier(max_depth=3, random_state=42)
            model.fit(iris.data, iris.target)
            
            model_info = {
                "source": "default_trained",
                "note": "Fallback model trained on startup"
            }
            logger.info("Default model trained and loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load default model: {str(e)}")
            return False


def load_model_from_mlflow():
    """Load model from MLflow registry or use default model"""
    global model, model_info
    
    tracer = get_tracer()
    with tracer.start_as_current_span("load_model_from_mlflow") as span:
        try:
            import mlflow.sklearn
            
            mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
            if mlflow_tracking_uri:
                mlflow.set_tracking_uri(mlflow_tracking_uri)
                logger.info(f"MLflow tracking URI configured", 
                           mlflow_uri=mlflow_tracking_uri)
            
            model_name = os.getenv("MODEL_NAME", "iris-classifier")
            model_stage = os.getenv("MODEL_STAGE", "Production")
            
            span.set_attribute("model.name", model_name)
            span.set_attribute("model.stage", model_stage)
            
            logger.info(f"Attempting to load model from MLflow", 
                       model_name=model_name, 
                       stage=model_stage)
            
            try:
                # Try with alias first (new way)
                model_uri = f"models:/{model_name}@champion"
                logger.info(f"Trying to load with alias", model_uri=model_uri)
                model = mlflow.sklearn.load_model(model_uri)
                model_info = {
                    "source": "mlflow_alias",
                    "model_name": model_name,
                    "alias": "champion"
                }
                span.set_attribute("model.source", "mlflow_alias")
                logger.info(f"Model loaded from MLflow with alias 'champion'")
                return True
            except Exception as alias_error:
                logger.warning(f"Could not load with alias: {str(alias_error)}")
                
                # Fallback to stage (old way)
                try:
                    model_uri = f"models:/{model_name}/{model_stage}"
                    logger.info(f"Trying to load with stage", model_uri=model_uri)
                    model = mlflow.sklearn.load_model(model_uri)
                    model_info = {
                        "source": "mlflow_stage",
                        "model_name": model_name,
                        "stage": model_stage
                    }
                    span.set_attribute("model.source", "mlflow_stage")
                    logger.info(f"Model loaded from MLflow stage '{model_stage}'")
                    return True
                except Exception as stage_error:
                    logger.warning(f"Could not load from MLflow stage: {str(stage_error)}")
                    raise
            
        except Exception as e:
            logger.warning(f"Could not load from MLflow: {str(e)}")
            logger.info("Falling back to default model...")
            return load_default_model()


# Middleware to log request/response
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log all requests with timing"""
    global request_counter
    request_counter += 1
    request_id = f"{int(time.time())}-{request_counter}"
    
    start_time = time.time()
    
    # Log incoming request
    logger.info(
        f"Incoming request",
        request_id=request_id,
        endpoint=str(request.url.path),
        method=request.method
    )
    
    # Process request
    response = await call_next(request)
    
    # Calculate latency
    latency_ms = round((time.time() - start_time) * 1000, 2)
    
    # Log response
    logger.info(
        f"Request completed",
        request_id=request_id,
        endpoint=str(request.url.path),
        status_code=response.status_code,
        latency_ms=latency_ms
    )
    
    # Add headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-ms"] = str(latency_ms)
    
    return response


@app.on_event("startup")
async def startup_event():
    """Load model and setup tracing on application startup"""
    logger.info("Starting IRIS Classification API...")
    
    # Setup tracing FIRST
    setup_tracing(app)
    
    logger.info(f"Environment configuration", 
               port=os.getenv("PORT", "8080"),
               model_name=os.getenv("MODEL_NAME", "iris-classifier"),
               model_stage=os.getenv("MODEL_STAGE", "Production"))
    
    success = load_model_from_mlflow()
    
    if not success:
        logger.error("Failed to load any model!")
    else:
        logger.info("API ready to serve predictions", model_info=model_info)


@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "IRIS Classification API",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "batch_predict": "/batch-predict (POST)",
            "model_info": "/model-info",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint for Kubernetes probes"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_info": model_info
    }


@app.get("/model-info", tags=["Model"])
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_info": model_info,
        "model_type": str(type(model).__name__),
        "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "classes": ["setosa", "versicolor", "virginica"]
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(features: IrisFeatures, request: Request):
    """
    Predict IRIS flower species from input features
    
    Args:
        features: IRIS flower measurements
        
    Returns:
        Prediction with confidence scores
    """
    if model is None:
        logger.error("Prediction failed: Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    tracer = get_tracer()
    with tracer.start_as_current_span("predict") as span:
        start_time = time.time()
        request_id = request.headers.get("X-Request-ID", "unknown")
        
        # Add attributes to span
        span.set_attribute("prediction.request_id", request_id)
        span.set_attribute("prediction.sepal_length", features.sepal_length)
        span.set_attribute("prediction.sepal_width", features.sepal_width)
        
        try:
            # Prepare input data
            with tracer.start_as_current_span("prepare_input"):
                input_data = np.array([[
                    features.sepal_length,
                    features.sepal_width,
                    features.petal_length,
                    features.petal_width
                ]])
            
            # Make prediction
            with tracer.start_as_current_span("model_inference"):
                prediction = model.predict(input_data)[0]
                probabilities = model.predict_proba(input_data)[0]
            
            # Map to class names
            with tracer.start_as_current_span("format_response"):
                class_names = ["setosa", "versicolor", "virginica"]
                predicted_class = class_names[prediction]
                confidence = float(probabilities[prediction])
                
                prob_dict = {
                    class_names[i]: float(probabilities[i])
                    for i in range(len(class_names))
                }
            
            latency_ms = round((time.time() - start_time) * 1000, 2)
            
            # Add prediction result to span
            span.set_attribute("prediction.result", predicted_class)
            span.set_attribute("prediction.confidence", confidence)
            span.set_attribute("prediction.latency_ms", latency_ms)
            
            # Log successful prediction
            logger.info(
                "Prediction successful",
                request_id=request_id,
                prediction=predicted_class,
                confidence=round(confidence, 3),
                latency_ms=latency_ms
            )
            
            return {
                "prediction": predicted_class,
                "confidence": confidence,
                "probabilities": prob_dict,
                "model_version": model_info.get("model_name", "default"),
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": request_id
            }
            
        except Exception as e:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            logger.error(
                f"Prediction error: {str(e)}",
                request_id=request_id
            )
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch-predict", tags=["Prediction"])
async def batch_predict(features_list: List[IrisFeatures], request: Request):
    """
    Predict multiple IRIS samples in batch
    
    Args:
        features_list: List of IRIS flower measurements
        
    Returns:
        List of predictions with confidence scores
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(features_list) > 100:
        raise HTTPException(status_code=400, detail="Batch size limited to 100 samples")
    
    tracer = get_tracer()
    with tracer.start_as_current_span("batch_predict") as span:
        start_time = time.time()
        request_id = request.headers.get("X-Request-ID", "unknown")
        
        span.set_attribute("batch.size", len(features_list))
        span.set_attribute("batch.request_id", request_id)
        
        try:
            results = []
            
            for idx, features in enumerate(features_list):
                with tracer.start_as_current_span(f"predict_item_{idx}"):
                    input_data = np.array([[
                        features.sepal_length,
                        features.sepal_width,
                        features.petal_length,
                        features.petal_width
                    ]])
                    
                    prediction = model.predict(input_data)[0]
                    probabilities = model.predict_proba(input_data)[0]
                    
                    class_names = ["setosa", "versicolor", "virginica"]
                    predicted_class = class_names[prediction]
                    confidence = float(probabilities[prediction])
                    
                    prob_dict = {
                        class_names[i]: float(probabilities[i])
                        for i in range(len(class_names))
                    }
                    
                    results.append({
                        "prediction": predicted_class,
                        "confidence": confidence,
                        "probabilities": prob_dict
                    })
            
            latency_ms = round((time.time() - start_time) * 1000, 2)
            
            span.set_attribute("batch.latency_ms", latency_ms)
            
            logger.info(
                f"Batch prediction completed",
                request_id=request_id,
                batch_size=len(results),
                latency_ms=latency_ms
            )
            
            return {
                "predictions": results,
                "count": len(results),
                "model_version": model_info.get("model_name", "default"),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            logger.error(
                f"Batch prediction error: {str(e)}",
                request_id=request_id
            )
            raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Get application metrics (for Prometheus/monitoring)"""
    return {
        "model_loaded": model is not None,
        "model_source": model_info.get("source", "unknown"),
        "total_requests": request_counter,
        "uptime": "healthy"
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )