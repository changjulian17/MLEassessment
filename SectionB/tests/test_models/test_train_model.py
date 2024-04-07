import unittest
import os
import pandas as pd
import pickle
from SectionB.src.models import train_model
from SectionB.src.models.train_model import eval_test_csv


class TestModelTrainingAndEvaluation(unittest.TestCase):
    """
       Unit tests for model training and evaluation.

       Methods:
           test_train_model(): Test case for training the created model.
           test_eval_test_csv(): Test case for evaluated data by the trained model on test data.

   """
    def test_train_model(self) -> None:
        # Check if the model file is not empty
        self.assertGreater(os.path.getsize('../../src/models/xgb_model.pickle'), 0)


    def test_eval_test_csv(self) -> None:
        eval_test_csv()

        # Check if the prediction file is created
        self.assertTrue(os.path.exists('../../data/results/prediction.csv'))

        # Check if the prediction file is not empty
        self.assertGreater(os.path.getsize('../../data/results/prediction.csv'), 0)

        # Load the predictions
        df_Y_eval = pd.read_csv('../../data/results/prediction.csv', index_col='trackID')

        # Perform some basic checks on the loaded predictions
        self.assertTrue('genre' in df_Y_eval.columns)


if __name__ == '__main__':
    unittest.main()
