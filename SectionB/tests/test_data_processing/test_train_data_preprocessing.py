import sys
import numpy as np
from SectionB.src.data_processing.train_data_preprocessing import preprocess_and_save_data
import unittest
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

class TestDataPreprocessing(unittest.TestCase):
    def test_preprocess_and_save_data(self):
        """
            Unit tests for data preprocessing.

            Methods:
                test_preprocess_and_save_data(): Test case for preprocessing and saving data.

        """

        # Call the function
        preprocess_and_save_data()

        # Check if the files are created
        self.assertTrue(os.path.exists('../../data/preprocessed_data/X.pickle'))
        self.assertTrue(os.path.exists('../../data/preprocessed_data/Y.pickle'))
        self.assertTrue(os.path.exists('../../data/preprocessed_data/X_eval.pickle'))
        self.assertTrue(os.path.exists('../../data/preprocessed_data/label_encoder.pickle'))

        # Check if the pickle files are not empty
        self.assertGreater(os.path.getsize('../../data/preprocessed_data/X.pickle'), 0)
        self.assertGreater(os.path.getsize('../../data/preprocessed_data/Y.pickle'), 0)
        self.assertGreater(os.path.getsize('../../data/preprocessed_data/X_eval.pickle'), 0)
        self.assertGreater(os.path.getsize('../../data/preprocessed_data/label_encoder.pickle'), 0)

        # Load data from pickle files
        with open('../../data/preprocessed_data/X.pickle', 'rb') as f:
            X = pickle.load(f)
        with open('../../data/preprocessed_data/Y.pickle', 'rb') as f:
            Y = pickle.load(f)
        with open('../../data/preprocessed_data/X_eval.pickle', 'rb') as f:
            X_eval = pickle.load(f)
        with open('../../data/preprocessed_data/label_encoder.pickle', 'rb') as f:
            label_encoder = pickle.load(f)

        # Check if data is loaded properly
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(Y, np.ndarray)
        self.assertIsInstance(X_eval, pd.DataFrame)
        self.assertIsInstance(label_encoder, LabelEncoder)

        # Clean up
        os.remove('../../data/preprocessed_data/X.pickle')
        os.remove('../../data/preprocessed_data/Y.pickle')
        os.remove('../../data/preprocessed_data/X_eval.pickle')
        os.remove('../../data/preprocessed_data/label_encoder.pickle')

if __name__ == '__main__':
    unittest.main()
