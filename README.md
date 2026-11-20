# Diabetes Prediction using Support Vector Machine (SVM)

## Overview
This project uses medical measurements from the PIMA Indians Diabetes dataset to predict the target label `Outcome` (0 or 1). It is a supervised binary-classification project.

## Problem Statement
Given eight input features for a record, predict the dataset's diabetes outcome label:
- `0` — non-diabetic
- `1` — diabetic

This is an educational machine-learning exercise, not a medical diagnostic system.

## Dataset
- File: `diabetes.csv`
- Records: 768
- Columns: 9
- Input features: `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`
- Target: `Outcome`

## Technologies and Libraries
- Python
- NumPy
- Pandas
- scikit-learn
- `StandardScaler`
- Support Vector Machine (`svm.SVC`)
- `accuracy_score`

## Workflow
1. Import the required Python libraries.
2. Load `diabetes.csv` into a Pandas DataFrame.
3. Inspect the dataset with `head()`, `shape()` and `describe()`.
4. Count target labels using `Outcome.value_counts()` and compare group averages using `groupby('Outcome').mean()`.
5. Separate the eight input columns (`X`) from the target (`Y`).
6. Standardize the input features using `StandardScaler`.
7. Split the data using `train_test_split(test_size=0.2, stratify=Y, random_state=2)`.
8. Create an SVM classifier with a linear kernel.
9. Train the classifier and evaluate training and testing accuracy.
10. Convert a new eight-value input to a NumPy array, reshape it, standardize it and predict its outcome.

## Model
The notebook uses **Support Vector Machine (SVM)** with `kernel='linear'`.

`StandardScaler` transforms features to a common scale. The notebook fits the scaler on the full feature dataset before splitting. For a stricter evaluation that avoids preprocessing leakage, split first, fit the scaler only on `X_train`, and use that fitted scaler to transform `X_test` and new inputs.

## Results
| Evaluation | Accuracy |
|---|---:|
| Training accuracy | 78.66% |
| Testing accuracy | 77.27% |

Accuracy is the proportion of predictions that match the actual labels for the evaluated split. Accuracy alone is not sufficient to assess a medical prediction model; recall, precision, F1-score and a confusion matrix should also be examined.

## Example Prediction
The notebook uses this example input, in the feature order listed above:
```python
(5, 166, 72, 19, 175, 25.8, 0.587, 51)
```
The notebook's displayed prediction is `1`, and it prints `The person is diabetic`.

This is only the model's prediction for the example record, not a clinical diagnosis.

## How to Run
1. Place `Diabetes_Prediction.ipynb` and `diabetes.csv` in an accessible working location.
2. Install the dependencies:
   ```bash
   pip install numpy pandas scikit-learn jupyter
   ```
3. Update the CSV path in the notebook if necessary.
4. Open the notebook in Jupyter Notebook or JupyterLab.
5. Run the cells from top to bottom.

## Notes
- The model expects eight numerical features in the same order as the training dataset.
- The notebook uses the dataset's `Outcome` values as supplied.
- Some zero values in medical measurement columns may warrant data-quality investigation before using this workflow for a more serious application.
