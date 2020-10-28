# AML Task 1

We used a two-step approach for this task:

Step 1) Feature Selection

In the first step, relevant features were selected by first imputing the missing values ("median" strategy), then selecting the best 20% of features according to the "mutual information" metric, and finally selecting 80 features with model-based greedy forward selection using a SVR model with rbf kernel and C=100.

Step 2) Model training

In the second step, the "reduced" data set with 80 features was first prepared by imputation (again "median" strategy), after scaled using sklearn's "RobustScaler" (which is more robust than the standard scaler when the data contains outliers), then outliers were detected and removed from the data set using the "Local Outlier Factor" algorithm with a contamination of 7%. Finally, a support vector regression model was trained, while hyperparameters were optimized using a randomized search. The optimal set of hyperparamters used in the end was C=62 and epsilon=0.0025.
