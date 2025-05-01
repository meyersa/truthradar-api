FROM python:3.9-slim

# Tinit for exiting cleanly
# Build essentials for python packages
RUN apt-get update && apt-get install -y tini build-essential && apt-get clean

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py ./
COPY lib/ lib/

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]