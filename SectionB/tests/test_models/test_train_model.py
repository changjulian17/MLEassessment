import unittest
import os
import pandas as pd
import pickle
from sklearn.utils import Bunch
from sklearn.metrics import accuracy_score
from SectionB.src.models import train_model
from SectionB.src.models.train_model import eval_test_csv


class TestModelTrainingAndEvaluation(unittest.TestCase):

    def test_train_model(self):
        # Call the train_model function
        train_model

        # Check if the model file is not empty
        self.assertGreater(os.path.getsize('../../src/models/xgb_model.pickle'), 0)

        # Load the saved model
        with open('test_xgb_model.pickle', 'rb') as f:
            loaded_model = pickle.load(f)

        # Perform some basic checks on the loaded model
        self.assertTrue(hasattr(loaded_model, 'predict'))
        self.assertTrue(hasattr(loaded_model, 'fit'))


    def test_eval_test_csv(self):
        eval_test_csv

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
