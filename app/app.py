import os
import time
import traceback
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Request
from db.engine import get_mongo_collection
from db.models import TextRequest, PredictionResult
from tools.intent_classifier import IntentClassifier, Config
from fastapi.middleware.cors import CORSMiddleware

import logging

# Import OpenTelemetry setup from observability module
from app.observability import init_opentelemetry
from opentelemetry import trace
from opentelemetry.metrics import get_meter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor 


# Initialize OpenTelemetry
init_opentelemetry()

# Get tracer and meter after initialization
tracer = trace.get_tracer(__name__)
meter = get_meter(__name__)

# Get custom metrics after initialization
prediction_count = meter.create_counter(
    "prediction_count",
    description="Number of predictions made"
)
prediction_latency = meter.create_histogram(
    "prediction_latency_seconds",
    description="Latency of model predictions in seconds",
    unit="s"
)

# Load environment variables from .env file
load_dotenv()

# Configure detailed logging (get logger after OTel setup)
logger = logging.getLogger(__name__)

# Read environment mode (defaults to prod for safety)
ENV = os.getenv("ENV", "prod").lower()
logger.info(f"Running in {ENV} mode")

"""
Authentication Logic
"""

async def conditional_auth():
    """Returns user based on environment mode"""
    if ENV == "dev":
        logger.info("Development mode: skipping authentication")
        return "dev_user"
    else:
        # Import and use real authentication in production
        try:
            from app.utils import verify_token
            return await verify_token()
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            raise HTTPException(status_code=401, detail="Authentication failed")

"""
FastAPI Setup
"""

with open('app/README.md', 'r') as f:
    description = f.read()

app = FastAPI(
    # title="Intent Classifier API",
    description=description,
    version="1.0.0",
    docs_url="/docs",        # Swagger UI
    redoc_url="/redoc",      # ReDoc
    root_path="/intents"     # Root path for the API (URL: http://localhost:8000/intents)
)

# Instrument FastAPI metrics
FastAPIInstrumentor().instrument_app(app)

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log incoming request
    logger.info(f"Incoming request: {request.method} {request.url}")
    if request.method == "POST":
        # Note: we can't log the body here easily without consuming it
        logger.info(f"Request headers: {dict(request.headers)}")
    
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} - Time: {process_time:.3f}s")
    
    return response

# Lista de origens permitidas (pode ser seu domínio, localhost, etc.)
origins = [
    "http://localhost",
    "http://localhost:3000",  # React ou outra frontend local
    "https://meusite.com",    # domínio em produção
]

# Adicionando o middleware de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # libera esses domínios
    allow_credentials=True,
    allow_methods=["*"],              # permite todos os métodos: GET, POST, etc
    allow_headers=["*"],              # permite todos os headers (Authorization, Content-Type...)
    # Durante o desenvolvimento: você pode usar allow_origins=["*"] para liberar tudo.
    # Em produção: evite "*" e especifique os domínios confiáveis.
)

import os

# Initialize models with error handling
MODELS = {}
try:
    logger.info("Loading confusion model...")
    # Construct absolute path to the model file
    model_path = os.path.join(os.path.dirname(__file__), "..", "tools", "models", "confusion-clf-v1.keras")
    MODELS["confusion"] = IntentClassifier(load_model=model_path)
    logger.info("Confusion model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load confusion model: {str(e)}")
    logger.error(traceback.format_exc())

# Initialize database connection
try:
    collection = get_mongo_collection(f"{ENV.upper()}_intent_logs")
    logger.info("Database connection established")
except Exception as e:
    logger.error(f"Failed to connect to database: {str(e)}")
    logger.error(traceback.format_exc())

"""
Routes
"""

# GET http://localhost:8000/intents/
@app.get("/", tags=["Root"])
async def read_root():
    logger.info("Root endpoint accessed")
    return {
        "message": f"Intent Classifier API is running in {ENV} mode. Check /redoc for more info."
    }

# POST http://localhost:8000/intents/confusion
@app.post("/confusion", response_model=PredictionResult)
async def predict_confusion(request: TextRequest, owner=Depends(conditional_auth)):
    logger.info(f"Confusion prediction request from user: {owner}")
    logger.info(f"Input text: '{request.text}'")

    with tracer.start_as_current_span("predict_confusion_model") as span:
        span.set_attribute("input.text.length", len(request.text))

        try:
            # Check if model is loaded
            if "confusion" not in MODELS:
                logger.error("Confusion model not loaded")
                raise HTTPException(status_code=500, detail="Model not available")

            # Make prediction
            logger.info("Making prediction...")
            start_time = time.time()
            # The predict method now returns: (top_intent_name, dict_of_all_probs)
            top_intent, all_probs = MODELS["confusion"].predict(request.text)
            end_time = time.time()
            latency = end_time - start_time

            # Record metrics
            prediction_count.add(1, {"intent": top_intent})
            prediction_latency.record(latency, {"intent": top_intent})

            span.set_attribute("prediction.top_intent", top_intent)
            span.set_attribute("prediction.latency_seconds", latency)

            logger.info(f"Top intent: {top_intent}")
            logger.info(f"All probabilities: {all_probs}")

            certainty = all_probs.get(top_intent) # Get certainty of the top_intent

            span.set_attribute("prediction.certainty", certainty)

            logger.info(f"Processed prediction: {top_intent}, certainty: {certainty}")
            # Create log entry
            log = {
                "text": request.text,
                "prediction": top_intent,
                "certainty": float(certainty) if certainty is not None else None,
                "all_probabilities": all_probs, # Log all probabilities
                "owner": owner
            }
            # Save to database
            try:
                collection.insert_one(log)
                logger.info("Prediction logged to database")
            except Exception as e:
                logger.error(f"Failed to log to database: {str(e)}")
                span.set_attribute("db.log.error", True)
                span.set_attribute("db.log.error.message", str(e))
                # Don't fail the request if logging fails

            result = PredictionResult(
                text=request.text,
                prediction=top_intent,
                certainty=float(certainty) if certainty is not None else None,
                all_probabilities=all_probs
            )
            logger.info(f"Returning result: {result}")
            return result
        except HTTPException as e:
            span.set_status(trace.Status(trace.StatusCode.ERROR, description=str(e)))
            raise
        except Exception as e:
            span.set_status(trace.Status(trace.StatusCode.ERROR, description=str(e)))
            logger.error(f"Error in predict_confusion: {str(e)}")
            logger.error(f"Exception type: {type(e)}")
            logger.error(f"Full traceback:")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")