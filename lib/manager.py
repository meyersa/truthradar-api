import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from lib.model import Model

models = [
    {
        "name": "BernoulliNB",
        "vectorizer": "https://f000.backblazeb2.com/file/TruthRadar/count_vectorizer.pkl",
        "link": "https://f000.backblazeb2.com/file/TruthRadar/count_Bernoulli_NB.pkl",
    },
    {
        "name": "LogisticRegression",
        "vectorizer": "https://f000.backblazeb2.com/file/TruthRadar/count_vectorizer.pkl",
        "link": "https://f000.backblazeb2.com/file/TruthRadar/count_Logistic_Regression.pkl",
    },
    {
        "name": "RandomForest",
        "vectorizer": "https://f000.backblazeb2.com/file/TruthRadar/count_vectorizer.pkl",
        "link": "https://truthradar.s3.us-west-000.backblazeb2.com/count_Random_Forest.pkl?versionId=4_z55a3e767b2b550e9956f0d18_f21189830fc161666_d20250422_m182715_c000_v0001412_t0019_u01745346435135",
    },
    {
        "name": "XGBoost",
        "vectorizer": "https://f000.backblazeb2.com/file/TruthRadar/count_vectorizer.pkl",
        "link": "https://f000.backblazeb2.com/file/TruthRadar/count_XGBoost.pkl",
    },
]

MAX_ELAPSED_MS = int(os.getenv("MAX_ELAPSED_MS", 20))

class Manager:
    """
    Manages multiple models and handles predictions.
    """

    def __init__(self):
        """
        Initializes models defined in the file.
        """
        self.models = []
        for config in models:
            try:
                logging.info(f"Attempting to load model: {config['name']}")
                model = Model(link=config['link'], name=config['name'], vectorizer_link=config['vectorizer'])
                model.description = config.get('description', "")
                self.models.append(model)
                logging.info(f"Successfully loaded model: {model.name}")
            except Exception as e:
                logging.error(f"Failed to load model {config['name']}: {e}")

        logging.info(f"Total models initialized: {len(self.models)}")

    def predict_all(self, text: str) -> list:
        """
        Run a text input through all loaded models in parallel.

        :param text: Text input to classify.
        :return: List of dicts [{name, score, description}]
        """
        results = []

        def predict_model(model):
            try:
                logging.info(f"Starting prediction with model: {model.name}")
                import time
                start = time.perf_counter()
                score = model._quick_test(text)
                elapsed_ms = (time.perf_counter() - start) * 1000

                logging.info(f"Prediction complete for {model.name} in {elapsed_ms:.2f} ms. Score: {score:.4f}")

                if elapsed_ms > MAX_ELAPSED_MS:
                    logging.warning(f"Model {model.name} exceeded maximum allowed time ({elapsed_ms:.2f} ms > {MAX_ELAPSED_MS} ms). Skipping result.")
                    return None

                return {
                    "name": model.name,
                    "score": float(score),
                }
            except Exception as e:
                logging.error(f"Prediction error for model {model.name}: {e}")
                return None

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(predict_model, model): model for model in self.models}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except TimeoutError:
                    model = futures[future]
                    logging.error(f"Timeout occurred during prediction for model {model.name}")

        logging.info(f"Prediction completed for {len(results)} models out of {len(self.models)} loaded models.")
        return results
