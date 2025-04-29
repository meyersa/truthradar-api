import os
import requests
import logging
import pickle
import time


MAX_ELAPSED_MS = os.getenv("MAX_ELAPSED_MS", 1000)


class Model:
    """
    A class to handle downloading, loading, and testing a model from a URL.
    """

    MODEL_PATH = "./models"

    def __init__(self, link: str, name: str, vectorizer_link: str = None) -> None:
        """
        Initialize a Model instance.

        :param link: URL to download the model file.
        :param name: Name to assign to the model file.
        :param vectorizer_link: Optional URL to download the vectorizer separately.
        :raises ValueError: If the name is invalid or model test fails.
        """
        self.name = self._handle_name(name)
        self.file_path = self._download(link, self.name)
        self.model_obj = self._load_pickle(self.file_path)

        self.model = self.model_obj  # Assume the whole pickle is the model
        self.vectorizer = None

        if vectorizer_link:
            vect_path = self._download(vectorizer_link, f"{self.name}_vectorizer")
            self.vectorizer = self._load_pickle(vect_path)
        else:
            raise ValueError(f"No vectorizer provided for {self.name}")

        self._quick_test("test")

    def _handle_name(self, name: str) -> str:
        """
        Verifies name is valid (5-50 chars) and formats it.

        :param name: Model name.
        :return: Cleaned model name.
        :raises ValueError: If the name length is invalid.
        """
        name = str(name).strip()
        name = name[0].upper() + name[1:]

        if not (5 < len(name) < 50):
            raise ValueError("Name must be between 5 and 50 characters.")
        return name

    def _download(self, link: str, name: str) -> str:
        """
        Downloads a file from a URL and saves it locally.

        :param link: URL of the file to download.
        :param name: Filename to save as.
        :return: Full path to the saved file.
        :raises requests.HTTPError: If download fails.
        """
        logging.info(f"Downloading {name}")
        logging.debug(f"Downloading URL {link}")

        path = os.path.join(self.MODEL_PATH, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        res = requests.get(link, stream=True)
        res.raise_for_status()

        with open(path, "wb") as f:
            for chunk in res.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        logging.info(f"Saved {name} to {path}")
        return path

    def _load_pickle(self, path: str) -> any:
        """
        Loads a pickle object from a file path.

        :param path: Path to the saved pickle file.
        :return: The unpickled Python object.
        :raises ValueError: If loading fails.
        """
        if not path:
            raise ValueError("Input path cannot be empty")

        logging.info(f"Loading pickle from {path}")

        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            raise ValueError(f"Could not load Pickle: {e}")

    def _quick_test(self, text: str) -> float:
        """
        Performs a test prediction using the loaded model and vectorizer.

        :param text: Sample input text to test.
        :return: Prediction score from the model.
        :raises ValueError: If prediction fails, the score is invalid, or it takes too long.
        """
        logging.info(f"Running quick test for {self.name}")
        result = self.predict(text)
        if result is None:
            raise ValueError(
                f"Quick test failed: {self.name} could not make a valid prediction."
            )
        logging.info(f"Quick test passed for {self.name}")

    def predict(self, text: str) -> dict:
        """
        Predicts on input text.

        :param text: Input text string.
        :return: {name, score, duration_ms} or None if prediction fails or times out.
        """
        if self.vectorizer is None or self.model is None:
            logging.error(f"Model {self.name} is not properly initialized.")
            return None

        start_time = time.perf_counter()
        try:
            X = self.vectorizer.transform([text])
            score = self.model.predict_proba(X)[0][1]
        except Exception as e:
            logging.error(f"Prediction failed for model {self.name}: {e}")
            return None

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logging.info(
            f"Predicted with {self.name}: score={score:.4f}, time={elapsed_ms:.2f}ms"
        )

        if elapsed_ms > MAX_ELAPSED_MS:
            logging.warning(
                f"Prediction for {self.name} exceeded {MAX_ELAPSED_MS}ms ({elapsed_ms:.2f}ms), skipping."
            )
            return None

        return {
            "name": self.name,
            "score": float(score),
            "duration_ms": round(elapsed_ms, 2),
        }
