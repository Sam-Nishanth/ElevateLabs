# Breast Cancer Classification using Logistic Regression

## Objective

To build a Logistic Regression model that classifies breast tumors as benign or malignant using the Breast Cancer dataset and evaluates its performance using classification metrics.

## Tools Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Dataset

The dataset contains various features computed from breast cancer cell nuclei. The target variable is **diagnosis**:
- **M** – Malignant
- **B** – Benign

## Project Workflow

1. Load and preprocess the dataset.
2. Handle missing values and remove unnecessary columns.
3. Encode the target variable.
4. Split the dataset into training and testing sets.
5. Standardize the features.
6. Train a Logistic Regression model.
7. Predict class labels and probabilities.
8. Evaluate the model using:
   - Confusion Matrix
   - Precision
   - Recall
   - ROC-AUC Score
9. Tune the classification threshold.
10. Plot the ROC Curve and Confusion Matrix.

## Evaluation Metrics

- Confusion Matrix
- Precision
- Recall
- ROC-AUC Score

## Results

The Logistic Regression model successfully classified breast tumors as benign or malignant. Model performance was evaluated using multiple classification metrics, and threshold tuning demonstrated the trade-off between precision and recall.

## Files

- `Logistic_Regression.py`
- `data.csv`
- `README.md`
- `requirements.txt`
- `Screenshots/`

## Requirements

Install the required libraries using:

```bash
pip install -r requirements.txt
```

## Conclusion

This project demonstrates the implementation of Logistic Regression for binary classification, feature standardization, model evaluation, ROC analysis, and threshold tuning using the Breast Cancer Wisconsin dataset.