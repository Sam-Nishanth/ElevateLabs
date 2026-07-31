\# K-Nearest Neighbors (KNN) Classification



\## Objective



To implement the K-Nearest Neighbors (KNN) algorithm for breast cancer classification, normalize the dataset, compare different values of K, evaluate model performance, and visualize the decision boundaries.



\## Tools Used



\* Python

\* Pandas

\* NumPy

\* Scikit-learn

\* Matplotlib



\## Dataset



Breast Cancer Wisconsin Dataset



\## Workflow



1\. Load and preprocess the dataset.

2\. Remove unnecessary columns and encode the target variable.

3\. Normalize the feature values using StandardScaler.

4\. Split the dataset into training and testing sets.

5\. Train KNN models with different values of K (3, 5, 7, and 9).

6\. Compare model accuracy for different K values.

7\. Evaluate the model using Accuracy and Confusion Matrix.

8\. Reduce the dataset to two dimensions using PCA.

9\. Visualize the KNN decision boundary.



\## Evaluation Metrics



\* Accuracy

\* Confusion Matrix



\## Files



\* `KNN\_Classification.py`

\* `data.csv`

\* `README.md`

\* `requirements.txt`

\* `Screenshots/`



\## Conclusion



The K-Nearest Neighbors algorithm successfully classified breast cancer samples after feature normalization. Testing multiple values of K showed how the number of neighbors influences classification performance. The confusion matrix and accuracy were used to evaluate the model, while PCA enabled visualization of the decision boundaries in a two-dimensional feature space.



