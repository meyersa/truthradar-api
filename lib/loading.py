import logging
import requests
import pickle
import os 

MODEL_PATH = "./models"


def download(link: str, name: str) -> str:
    """
    Downloads a file from a URL and saves it locally.

    Args:
        link (str): The URL of the file to download.
        name (str): The filename to save as (within MODEL_PATH).

    Returns:
        str: The full path to the saved file.

    Raises:
        requests.HTTPError: If the request fails.
        ValueError: If link or name is empty.
    """
    logging.info(f"Downloading model {name}")
    logging.debug(f"Downloading URL {link}")

    path = f"{MODEL_PATH}/{name}"
    os.makedirs(os.path.dirname(path), exist_ok=True)

    res = requests.get(link)
    res.raise_for_status()

    with open(path, "wb") as f:
        f.write(res.content)

    logging.info(f"Saved {name} to {path}")
    return path


def load_pickle_from_url(link: str, name: str) -> any:
    """
    Downloads and loads a pickle object from a URL.

    Args:
        link (str): The URL to download the pickle file from.
        name (str): The name to save the file as locally.

    Returns:
        Any: The unpickled Python object.

    Raises:
        ValueError: If inputs are missing or loading fails.
    """
    if not name:
        raise ValueError("Input Name cannot be empty")

    logging.info(f"Downloading pickle {name} from URL")

    if not link:
        raise ValueError("Input URL cannot be empty")

    try:
        file = download(link=link, name=name)

        with open(file, "rb") as f:
            return pickle.load(f)

    except Exception as e:
        raise ValueError("Could not load Pickle from URL")


def quick_test(text: str, vect, model) -> float:
    """
    Performs a test prediction using a given model and vectorizer.

    Args:
        text (str): The input text to classify.
        vect (Any): A fitted vectorizer with `.transform()`.
        model (Any): A fitted model with `.predict_proba()`.

    Returns:
        float: The prediction score from the model.

    Raises:
        ValueError: If prediction or transformation fails or if score is invalid.
    """
    logging.info("Attempting quick test on model")

    score = 100

    try:
        X = vect.transform([text])
        score = model.predict_proba(X)[0][1]

    except Exception as e:
        raise ValueError("Could not load Model or Vectorizer")

    logging.debug(f"Sample prediction score: {score}")
    if 0 > score > 1:
        raise ValueError("Model is not passing basic tests")

    return score
