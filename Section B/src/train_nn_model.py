import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

"""
getting data for training
"""
df_feat = pd.read_csv('../data/features.csv')
df_y = pd.read_csv('../data/labels.csv')

"""
Training with xgboost
"""

# Drop nulls
df_feat = df_feat.dropna()
df_y = df_y.loc[df_feat.index]

# Enriching with two more attributes
# Count of tags
df_feat['tags_count'] = df_feat['tags'].apply(lambda x: len(x.split(',')))
# Longest tag
df_feat['longest_tag_length'] = df_feat['tags'].apply(lambda x: max(len(tag) for tag in x.split(',')))

# Drop title and tags col
X = df_feat.drop(columns=['title', 'tags']).set_index('trackID')
Y = df_y.set_index('trackID')

# Initialize LabelEncoder
label_encoder = LabelEncoder()
# Fit and transform the target variable
Y = label_encoder.fit_transform(Y.genre)

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

# Define the neural network model
neural_network = MLPClassifier(alpha=5e-05,
                               hidden_layer_sizes=(20,),
                               activation='relu',
                               solver='adam',
                               random_state=1,
                               max_iter=2_000_000,
                               batch_size=1000,
                               )


# Define the XGBoost model
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', neural_network)
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
