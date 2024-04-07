import pickle
import pandas as pd


class XGBModelWrapper:
    def __init__(self):
        self.label_encoder_path = '../../data/preprocessed_data/label_encoder.pickle'
        self.model_path = '../../src/models/xgb_model.pickle'
        self.label_encoder = None
        self.model = None

    def load_model(self):
        # Load saved encoder
        with open(self.label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        f.close()
        # Load saved model
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        f.close()

    def predict(self, input_data):
        if self.label_encoder is None or self.model is None:
            raise ValueError("Model has not been loaded. Call load_model() first.")
        # Perform prediction using the loaded model
        Y_pred = self.model.predict(input_data)
        # Decode labels to strings
        Y_pred = self.label_encoder.inverse_transform(Y_pred)
        # Prepare predictions as a data frame
        df_Y_eval = pd.DataFrame(Y_pred, columns=['genre'])
        df_Y_eval.index = input_data.index

        return df_Y_eval

