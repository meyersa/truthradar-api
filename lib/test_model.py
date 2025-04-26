import unittest
from model import Model
import sys
import logging

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

class TestModelInitialization(unittest.TestCase):
    """
    Tests loading a real model from remote URL.
    """

    def setUp(self):
        self.model_url = "https://f000.backblazeb2.com/file/TruthRadar/count_Bernoulli_NB.pkl"
        self.vector_url = "https://f000.backblazeb2.com/file/TruthRadar/count_vectorizer.pkl"
        self.model_name = "BernoulliNBTest"

    def test_model_loads_successfully(self):
        """
        Test that the Model class correctly downloads and loads the model.
        """
        model = Model(link=self.model_url, name=self.model_name, vectorizer_link=self.vector_url)

        # Check that model and vectorizer exist
        self.assertIsNotNone(model.model, "Model object should not be None")
        self.assertIsNotNone(model.vectorizer, "Vectorizer should not be None")

        # Quick prediction test (should not raise an exception)
        model._quick_test("The sky is blue.")

        # Full predict() test
        prediction = model.predict("The sky is blue.")
        self.assertIsInstance(prediction, dict, "Prediction should be a dictionary.")
        self.assertIn("name", prediction)
        self.assertIn("score", prediction)
        self.assertIn("duration_ms", prediction)
        self.assertIsInstance(prediction["name"], str)
        self.assertIsInstance(prediction["score"], float)
        self.assertIsInstance(prediction["duration_ms"], (int, float))

if __name__ == "__main__":
    unittest.main()
