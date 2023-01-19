# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
* Developer: Ivo
* Model date: 03-jan-2023
* Model version: 1.0.0
* Model type: Logistic regression

## Intended Use
* Estimate if income is over or under 50k.

## Training Data
* Datasets: Census Income Data Set from UCI Machine Learning Repository.
* Motivation: Build a binary classification model to predict if a person's income is over 50k.
* Preprocessing:
- 80% of total samples
- Removed extra space in column names
- All categorical column are encoded using OneHotEncoder from scikit-learn
- Label column ('salary') is encoded using LabelBinarizer from scikit-learn

## Evaluation Data
* Datasets: Census Income Data Set from UCI Machine Learning Repository
* Motivation: Build a binary classification model to predict if a person's income is over 50k given several features.
* Preprocessing: (can be checked in -)
- 20% of total samples
- Removed extra space in column names
- All categorical column are encoded using OneHotEncoder from scikit-learn
- Label column ('salary') is encoded using LabelBinarizer from scikit-learn

## Metrics
* Model performance measures:
- Precision: 0.7206
- Recall: 0.2487
- F1: 0.3698

## Ethical Considerations
* There are no sensitive information
* The data is not used to inform decisions about matters central to human life or floursihing - e.g., health or safety.

## Caveats and Recommendations
* When running data slicing model validation by native-country dimension you can see that model is very poor; it performs F1 zero for many of those dimension; probably cause there is a small proportion of samples for country as Haiti, for example.

* For sure, model can be improved by running at least an HPO (Hyperparamenter Optimization) or even calling Autosklearn as a first try.
