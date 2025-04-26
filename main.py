import logging
import dotenv
import os
import sys
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import time

from lib.manager import Manager
dotenv.load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "debug")
API_KEY = os.getenv("API_KEY")

level = getattr(logging, LOG_LEVEL.upper())
logging.basicConfig(stream=sys.stdout, level=level)
logging.info(f"Log level set to {level}")

if API_KEY:
    logging.info("Using API Key")

# --- Load models ---
MANAGER = Manager()
logging.info(f"Loaded {len(MANAGER.models)} models successfully")

# --- FastAPI app ---
app = FastAPI()

class InputText(BaseModel):
    text: str

# Healthcheck endpoint
@app.get("/predict")
def pong():
    return {"message": "pong"}

# Predict endpoint
@app.post("/predict")
def predict(data: InputText, x_api_key: str = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        logging.warning("Unauthorized access attempt")
        raise HTTPException(
            status_code=401, detail="Invalid or missing API key"
        )

    start_time = time.perf_counter()
    logging.info(f"Received prediction request: {data.text}")

    try:
        results = MANAGER.predict_all(data.text)
        duration = time.perf_counter() - start_time

        response = {
            "predictions": results,
            "duration_ms": round(duration * 1000, 2)
        }
        logging.info(f"Prediction completed: {response}")
        return response
    except Exception as e:
        logging.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction error")
