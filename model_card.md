# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Sascha Bielawski created the model. It is a Decision Tree classifier using the default hyperparameters in scikit-learn 0.23.2.

## Intended Use
This model should be used to predict the salary greater or lower than 50,000$
based off a handful of attributes.

## Training Data
Training data consists of 80% of the whole dataset. A One-Hot-Encoder
is used to encode the categorical features. The label data 'salary'
is binarized.

## Evaluation Data
Evaluation data consists of 20% of the whole dataset. All encodings
are the same as for the training data.

## Metrics
The model was evaluated using the following scores:  
- Precision:  0.6185632549268912  
- Recall:  0.6289592760180995  
- Fbeta:  0.6237179487179487  

## Ethical Considerations
The dataset is unbalanced as it can be seen for the feature education in the slice_output.txt.

## Caveats and Recommendations
It is recommended to train a model on a more balanced dataset.
