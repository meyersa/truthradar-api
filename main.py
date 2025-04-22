import logging
import dotenv
import os
import sys
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import time
from lib.loading import load_pickle_from_url, quick_test

dotenv.load_dotenv()
LOG_LEVEL = os.getenv("LOG_LEVEL", "debug")
MODEL_LINK = os.getenv("MODEL_LINK")
VECTORIZER_LINK = os.getenv("VECTORIZER_LINK")
API_KEY = os.getenv("API_KEY")

level = getattr(logging, LOG_LEVEL.upper())

logging.basicConfig(stream=sys.stdout, level=level)
logging.info(f"Log level set to {level}")

if not MODEL_LINK:
    raise SystemExit("Missing Model Link")

logging.info(f"Model Link: {MODEL_LINK}")

if not VECTORIZER_LINK:
    raise SystemExit("Missing Model Link")

logging.info(f"Vectorizer Link: {VECTORIZER_LINK}")

if API_KEY:
    logging.info("Using API Key")

MODEL = load_pickle_from_url(MODEL_LINK, "model")
VECT = load_pickle_from_url(VECTORIZER_LINK, "vect")


quick_test("Plants are blue", model=MODEL, vect=VECT)

app = FastAPI()


class InputText(BaseModel):
    text: str


@app.post("/predict")
def predict(data: InputText, x_api_key: str = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        logging.warning("Unauthorized access attempt")
        raise HTTPException(
            status_code=401, detail="Invalid or missing API key")

    start_time = time.perf_counter()
    logging.info(f"Received prediction request: {data.text}")
    try:
        X = VECT.transform([data.text])
        logging.debug("Text transformed with vectorizer")

        score = MODEL.predict_proba(X)[0][1]
        logging.debug(f"Predicted score: {score}")

        label = MODEL.predict(X)[0]
        logging.debug(f"Predicted label: {label}")

        duration = time.perf_counter() - start_time
        result = {
            "label": int(label),
            "score": round(float(score), 4),
            "duration_ms": round(duration * 1000, 2)
        }
        logging.info(f"Prediction completed: {result}")
        return result
    except Exception as e:
        logging.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction error")
