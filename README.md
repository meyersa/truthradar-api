# TruthRadar API

FastAPI service for serving predictions from trained models used by the TruthRadar frontend.

## Setup

1. Build the Docker container:
    ```bash
    docker build -t truthradar-api .
    ```

2. Run the container:
    ```bash
    docker run -p 8000:8000 \
      --env-file .env \
      truthradar-api
    ```

## Environment Variables

- `MODEL_LINK`: URL to the trained model (e.g., S3 or B2)
- `VECTORIZER_LINK`: URL to the vectorizer
- `API_KEY`: (optional) API key required for requests to `/predict`
- `LOG_LEVEL`: (optional) Set logging level (`debug`, `info`, etc.)

## API

### POST `/predict`

**Headers:**
- `X-API-Key`: Required if `API_KEY` is set in the environment.

**Request Body:**
```json
{
  "text": "Example claim to verify"
}
```

**Response:**
```json
{
  "label": 1,
  "score": 0.8123,
  "duration_ms": 12.45
}
```
