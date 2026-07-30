# Decision Tree and Random Forest Classification

## Objective

To implement Decision Tree and Random Forest classifiers for breast cancer diagnosis, compare their performance, analyze feature importance, and evaluate the models using cross-validation.

## Tools Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Graphviz

## Dataset

Breast Cancer Wisconsin Dataset

## Workflow

1. Load and preprocess the dataset.
2. Train a Decision Tree classifier.
3. Visualize the Decision Tree.
4. Control overfitting using maximum tree depth.
5. Train a Random Forest classifier.
6. Compare model accuracy.
7. Analyze feature importance.
8. Perform 5-fold cross-validation.

## Evaluation

- Decision Tree Accuracy
- Random Forest Accuracy
- Feature Importance
- Cross Validation Accuracy

## Files

- `Decision_Tree_Random_Forest.py`
- `data.csv`
- `README.md`
- `requirements.txt`
- `Screenshots/`

## Conclusion

Decision Trees are simple and interpretable but can overfit the training data. Limiting the tree depth helps improve generalization. Random Forest combines multiple Decision Trees to produce better accuracy and robustness. Feature importance identifies the most influential attributes in predicting breast cancer diagnosis.