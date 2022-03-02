# Script to train machine learning model.
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

# Add the necessary imports for the starter code.

# Add code to load in the data.
data = pd.read_csv('processed_census.csv')
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder_test, lb_test = process_data(
    test, categorical_features=cat_features, encoder=encoder, lb=lb, label="salary", training=False
)

# Train and save a model.
clf = train_model(X_train, y_train)
preds = inference(clf, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)

filename = "classifier.sav"
pickle.dump(clf, open(filename, 'wb'))
