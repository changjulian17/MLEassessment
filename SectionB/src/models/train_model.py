import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import xgboost as xgb
from sklearn.metrics import accuracy_score


def train_model():
    """
    takes data from pickles and trains model and saves it
    """
    # Load pre-processed data
    with open('../../data/preprocessed_data/X.pickle', 'rb') as f:
        X = pickle.load(f)
    f.close()
    with open('../../data/preprocessed_data/Y.pickle', 'rb') as f:
        Y = pickle.load(f)
    f.close()
    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                        Y,
                                                        test_size=0.2,
                                                        random_state=42)

    # Define column transformer for preprocessing
    categorical_features = ['time_signature', 'key', 'mode']
    numeric_features = [col for col in X.columns if col not in ['trackID'] + categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])

    # Define the XGBoost model
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier())
    ])

    # Train the model
    model.fit(X_train, Y_train)

    # Predictions on the test set
    Y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(Y_test, Y_pred)
    print("Accuracy:", accuracy)

    # Count predictions
    predictions_count = {label: (Y_pred == label).sum() for label in set(Y_test)}
    print("Predictions count:", predictions_count)

    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, Y_train, cv=10)

    # Print the cross-validation scores
    print("Cross-validation scores:", cv_scores)
    print("Mean CV score:", cv_scores.mean())

    # Saving model to pickle file
    with open('xgb_model.pickle', 'wb') as f:
        pickle.dump(model, f)
    f.close()


def eval_test_csv():
    """
    predicting the test csv
    """
    # Load saved encoder
    with open('../../data/preprocessed_data/label_encoder.pickle', 'rb') as f:
        label_encoder = pickle.load(f)
    f.close()
    # Load saved model
    with open('xgb_model.pickle', 'rb') as f:
        model = pickle.load(f)
    f.close()
    # Load eval csv
    with open('../../data/preprocessed_data/X_eval.pickle', 'rb') as f:
        X_eval = pickle.load(f)
    f.close()

    # Use saved model to predict genres
    Y_eval = model.predict(X_eval)
    # Decode
    Y_eval = label_encoder.inverse_transform(Y_eval)
    df_Y_eval = pd.DataFrame(Y_eval, columns=['genre'])

    # Save DataFrame to CSV file
    df_Y_eval.to_csv('../data/results/prediction.csv', index_label='trackID')


# train_model()

# eval_test_csv()  #todo remove if called outside


