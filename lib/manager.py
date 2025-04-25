from lib.model import Model
import logging

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
        "link": "https://f000.backblazeb2.com/file/TruthRadar/count_Random_Forest.pkl",
    },
    {
        "name": "XGBoost",
        "vectorizer": "https://f000.backblazeb2.com/file/TruthRadar/count_vectorizer.pkl",
        "link": "https://f000.backblazeb2.com/file/TruthRadar/count_XGBoost.pkl",
    },
]
vectorizer = "https://f000.backblazeb2.com/file/TruthRadar/count_vectorizer.pkl"

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
                model = Model(link=config['link'], name=config['name'])
                self.models.append(model)
            except Exception as e:
                logging.error(f"Failed to load model {config['name']}: {e}")

    def predict_all(self, text: str) -> list:
        """
        Run a text input through all loaded models.

        :param text: Text input to classify.
        :return: List of dicts [{name, score, description}]
        """
        results = []
        for model in self.models:
            try:
                score = model._quick_test(text)
            except Exception as e:
                logging.error(f"Prediction failed for {model.name}: {e}")
                score = None

            results.append({
                "name": model.name,
                "score": score,
                "description": getattr(model, "description", "")
            })

        return results
