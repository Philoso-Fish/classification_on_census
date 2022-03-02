# Script to train machine learning model.
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (compute_model_metrics, inference, test_on_slices,
                      train_model)

# Add the necessary imports for the starter code.

# Add code to load in the data.
data = pd.read_csv("processed_census.csv", index_col=0)
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
    test,
    categorical_features=cat_features,
    encoder=encoder,
    lb=lb,
    label="salary",
    training=False,
)

# Train and save a model.
clf = train_model(X_train, y_train)
preds = inference(clf, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print("Precision: ", precision)
print("Recall: ", recall)
print("Fbeta: ", fbeta)

# save model
filename = "classifier.sav"
pickle.dump(clf, open(filename, "wb"))
# save encoder
filename = "encoder.sav"
pickle.dump(encoder, open(filename, "wb"))

list_slice = test_on_slices(clf, data, "education", "salary", encoder, lb, cat_features)

# print model metrics on slices of variable education
with open("slie_output.txt", "w") as f:
    f.write("Model metrics on variable 'Education'\n")
    for tpl in list_slice:
        f.write(f"Value: {tpl[0]}\n")
        f.write(f"Precision: {tpl[1]}\n")
        f.write(f"Recall: {tpl[2]}\n")
        f.write(f"Fbeta: {tpl[3]}\n")
        f.write("###############################\n")
