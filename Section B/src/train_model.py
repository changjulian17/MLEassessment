import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
import xgboost as xgb
from sklearn.metrics import accuracy_score


# Load pre-processed data
with open('../data/preprocessed_data/X.pickle', 'rb') as f:
    X = pickle.load(f)
with open('../data/preprocessed_data/Y.pickle', 'rb') as f:
    Y = pickle.load(f)
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
with open('../model/xgb_model.pickle', 'wb') as f:
    pickle.dump(model, f)
