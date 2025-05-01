import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from lib.model import Model
import contractions
import nltk
from nltk.corpus import stopwords
import re

nltk.download('punkt_tab')
nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english"))
MAX_ELAPSED_MS = int(os.getenv("MAX_ELAPSED_MS", 1000))

models = [
    {
        "name": "BernoulliNB",
        "vectorizer": "https://f000.backblazeb2.com/file/TruthRadar/vectorizer_tfidf.pkl",
        "link": "https://f000.backblazeb2.com/file/TruthRadar/BernoulliNB.pkl",
    },
    {
        "name": "LogisticRegression",
        "vectorizer": "https://f000.backblazeb2.com/file/TruthRadar/vectorizer_tfidf.pkl",
        "link": "https://f000.backblazeb2.com/file/TruthRadar/LogisticRegression.pkl",
    },
    {
        "name": "RandomForest",
        "vectorizer": "https://f000.backblazeb2.com/file/TruthRadar/vectorizer_tfidf.pkl",
        "link": "https://truthradar.s3.us-west-000.backblazeb2.com/RandomForest.pkl",
    },
    {
        "name": "XGBoost",
        "vectorizer": "https://f000.backblazeb2.com/file/TruthRadar/vectorizer_tfidf.pkl",
        "link": "https://f000.backblazeb2.com/file/TruthRadar/XGBoost.pkl",
    },
    {
        "name": "PassiveAggressive",
        "vectorizer": "https://f000.backblazeb2.com/file/TruthRadar/vectorizer_tfidf.pkl",
        "link": "https://f000.backblazeb2.com/file/TruthRadar/PassiveAggressive.pkl",
    },
    {
        "name": "RidgeClassifier",
        "vectorizer": "https://f000.backblazeb2.com/file/TruthRadar/vectorizer_tfidf.pkl",
        "link": "https://f000.backblazeb2.com/file/TruthRadar/RidgeClassifier.pkl",
    },
    {
        "name": "SGDClassifier",
        "vectorizer": "https://f000.backblazeb2.com/file/TruthRadar/vectorizer_tfidf.pkl",
        "link": "https://f000.backblazeb2.com/file/TruthRadar/SGDClassifier.pkl",
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

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text to match model training

        :param text: Text to process
        :return: Processed text
        """
        # Remove Contractions
        try:
            text = contractions.fix(text)
        except Exception:
            pass

        # Lowercase
        text = text.lower()

        # Remove Punctuation
        text = re.sub(r"[^\w\s]", "", text)

        # Remove Stopwards
        tokens = nltk.word_tokenize(text)
        tokens = [t for t in tokens if t not in STOPWORDS]
        return " ".join(tokens)

    def predict_all(self, text: str) -> list:
        """
        Run a text input through all loaded models in parallel.

        :param text: Text input to classify.
        :return: List of dicts [{name, score, description}]
        """
        results = []

        # Prepare text for prediction
        text = self._preprocess_text(text)
        
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(model.predict, text): model for model in self.models
            }
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)

        logging.info(
            f"Prediction completed for {len(results)} models out of {len(self.models)} loaded models."
        )
        return results
