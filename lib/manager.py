import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from lib.model import Model


MAX_ELAPSED_MS = int(os.getenv("MAX_ELAPSED_MS", 50))

models = [
    {
        "name": "CountBernoulliNB",
        "vectorizer": "https://f000.backblazeb2.com/file/TruthRadar/vectorizer_count.pkl",
        "link": "https://f000.backblazeb2.com/file/TruthRadar/count_Bernoulli_NB.pkl",
    },
    {
        "name": "CountLogisticRegression",
        "vectorizer": "https://f000.backblazeb2.com/file/TruthRadar/vectorizer_count.pkl",
        "link": "https://f000.backblazeb2.com/file/TruthRadar/count_Logistic_Regression.pkl",
    },
    # {
    #     "name": "CountRandomForest",
    #     "vectorizer": "https://f000.backblazeb2.com/file/TruthRadar/vectorizer_count.pkl",
    #     "link": "https://truthradar.s3.us-west-000.backblazeb2.com/count_Random_Forest.pkl?versionId=4_z55a3e767b2b550e9956f0d18_f21189830fc161666_d20250422_m182715_c000_v0001412_t0019_u01745346435135",
    # },
    {
        "name": "CountXGBoost",
        "vectorizer": "https://f000.backblazeb2.com/file/TruthRadar/vectorizer_count.pkl",
        "link": "https://f000.backblazeb2.com/file/TruthRadar/count_XGBoost.pkl",
    },
        {
        "name": "TFIDFBernoulliNB",
        "vectorizer": "https://f000.backblazeb2.com/file/TruthRadar/vectorizer_tfidf.pkl",
        "link": "https://f000.backblazeb2.com/file/TruthRadar/tfidf_Bernoulli_NB.pkl",
    },
    {
        "name": "TFIDFLogisticRegression",
        "vectorizer": "https://f000.backblazeb2.com/file/TruthRadar/vectorizer_tfidf.pkl",
        "link": "https://f000.backblazeb2.com/file/TruthRadar/tfidf_Logistic_Regression.pkl",
    },
    # {
    #     "name": "TFIDFRandomForest",
    #     "vectorizer": "https://f000.backblazeb2.com/file/TruthRadar/vectorizer_tfidf.pkl",
    #     "link": "https://truthradar.s3.us-west-000.backblazeb2.com/tfidf_Random_Forest.pkl?versionId=4_z55a3e767b2b550e9956f0d18_f21189830fc161666_d20250422_m182715_c000_v0001412_t0019_u01745346435135",
    # },
    {
        "name": "TFIDFXGBoost",
        "vectorizer": "https://f000.backblazeb2.com/file/TruthRadar/vectorizer_tfidf.pkl",
        "link": "https://f000.backblazeb2.com/file/TruthRadar/tfidf_XGBoost.pkl",
    },
]


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
                model = Model(
                    link=config["link"],
                    name=config["name"],
                    vectorizer_link=config["vectorizer"],
                )
                model.description = config.get("description", "")
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

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(model.predict, text): model for model in self.models}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)

        logging.info(
            f"Prediction completed for {len(results)} models out of {len(self.models)} loaded models."
        )
        return results