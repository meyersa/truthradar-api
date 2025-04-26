import unittest
from manager import Manager, models  # your actual manager + models list
import sys 
import logging

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

class TestManager(unittest.TestCase):
    """
    Unit tests for the Manager class.
    """

    def setUp(self):
        """
        Initializes a Manager instance before each test.
        """
        self.manager = Manager()

    def test_models_loaded_correctly(self):
        """
        Tests that all models defined are loaded into the Manager.
        """
        self.assertEqual(
            len(self.manager.models), 
            len(models), 
            "The number of loaded models should match the defined models list."
        )

    def test_predict_all(self):
        """
        Tests that predict_all returns valid results for a sample input.
        """
        sample_text = "The sky is blue."
        predictions = self.manager.predict_all(sample_text)

        self.assertIsInstance(predictions, list, "Predictions output should be a list.")
        for prediction in predictions:
            self.assertIn("name", prediction, "Each prediction must include a 'name'.")
            self.assertIn("score", prediction, "Each prediction must include a 'score'.")
            self.assertIsInstance(prediction["name"], str, "Model name must be a string.")
            self.assertTrue(prediction["score"] is None or isinstance(prediction["score"], float), "Score must be a float or None.")

if __name__ == "__main__":
    unittest.main()
