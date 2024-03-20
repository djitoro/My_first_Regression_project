# Cancellation forecast
___
## Main idaes: 
### 1. Reduce data dimensionality:
Thanks to the construction of graphs, we determine which data is not valuable to us. For example "year"
  ![rep](https://github.com/djitoro/RTU_LAB_basic_project/blob/master/pic/years.png)
 
### 2. Replacing data type string
Since the linear regression built into Sklern can only work with float data. We consider the ratio of positive responses at this value divided by the number of negative responses. Thanks to this, we obtain a ratio that corresponds to the weight of a given variable for the model

(since log regression involves multiplying coefficients, then simple replacement by 1, 2, 3... is not suitable)
  ![con](https://github.com/djitoro/RTU_LAB_basic_project/blob/master/pic/convert%20to%20float.png)
  

Due to the fact that the amount of data was small, it was much faster to display all the data on the screen and write it into code, however, in more complex cases it is better to make calculations dynamic

### 3. Logistic regression
We need to give a binary answer; the data as a whole can be called independent, which means that the ideal option for us is logistic regression

   ```python
# instantiate the model
log_regression = LogisticRegression(max_iter=1500)
# fit the model using the training data
log_regression.fit(X_train, y_train)
# use model to make predictions on test data
y_pred = log_regression.predict(X_test)
# system response matrix:
# true guesses; true not guessed
# false guesses; false not guessed
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
   ```

### 4. Data Expansion
Since the data was unevenly distributed, there was an attempt to expand and normalize the sample using the SMOTE library, however, on extended data sets the model showed significantly worse results, so it was decided to abandon this idea, however, to further increase the accuracy, it will be possible to try to implement this method

___
## Metrics: 
Accuracy on test data: 0.8093637992831542

![Metrics](https://github.com/djitoro/RTU_LAB_basic_project/blob/master/pic/Metrics.png)

___
## Possible improvements: 
### 1. During the improvement of the model, some solvers were reviewed, but it is possible that a much more advanced version will be available
### 2. Data normalization and equal distribution, in this area it is also possible to improve the model
### 3. You can also try to select a regularization algorithm for better learning quality

___
:wink:
