from fastapi import FastAPI, HTTPException
from db.engine import get_mongo_collection
from db.models import TextRequest, PredictionResult
from tools.intent_classifier import IntentClassifier, Config

"""
FastAPI Setup
"""

with open('README.md', 'r') as f:
    description = f.read()

app = FastAPI(
    title="Intent Classifier API",
    description=description,
    version="1.0.0",
    docs_url="/docs",        # Swagger UI
    redoc_url="/redoc",      # ReDoc
    root_path="/intents"     # Root path for the API (URL: http://localhost:8000/intents)
)

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

MODELS = {
    "confusion": IntentClassifier(load_model="models/confusion_classifier/")
}

collection = get_mongo_collection("intent_logs")

"""
Routes
"""
# GET http://localhost:8000/intents/
@app.get("/", tags=["Root"])
async def read_root():
    return {
        "message": "Intent Classifier API is running. Check /redoc for more info."
    }

# POST http://localhost:8000/intents/confusion
@app.post("/confusion", response_model=PredictionResult)
async def predict_confusion(request: TextRequest):
    try:
        pred = MODELS["confusion"].predict(request.text, get_certainty=True)
        prediction, certainty = pred['label'], pred['certainty']

        log = {
            "text": request.text,
            "prediction": prediction,
            "certainty": certainty
        }
        collection.insert_one(log)

        return PredictionResult(text=request.text, prediction=prediction, certainty=certainty)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
