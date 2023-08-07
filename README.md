# Explainable_Credit_Prediction
Feature Importance Analysis and Rule Extraction with Explainable Classification

Background
This research focuses on the issue of customers' default payments in Taiwan and aims to compare the predictive accuracy of the probability of default using six different data mining methods. From a risk management perspective, the accuracy of predicting the estimated probability of default holds greater value than the binary classification outcome of credible or non-credible clients. This repository contains Python code to perform feature importance analysis and if-then rule extraction using a Decision Tree classifier. Feature importance analysis is an essential step in understanding the contribution of different features within a dataset to classification decisions. The provided code enables loading data from an Excel file, splitting it into training and testing subsets, training a Decision Tree classifier, evaluating its accuracy, computing feature importances, visualizing the results using a bar plot, and extracting important if-then rules for each of the classes.

Dataset
This research employed a binary variable, default payment (Yes = 1, No = 0), as the response variable. This study reviewed the literature and used the following 23 variables as explanatory variables:
X1: Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit.
X2: Gender (1 = male; 2 = female).
X3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).
X4: Marital status (1 = married; 2 = single; 3 = others).
X5: Age (year).
X6 - X11: History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows: X6 = the repayment status in September, 2005; X7 = the repayment status in August, 2005; . . .;X11 = the repayment status in April, 2005. The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.
X12-X17: Amount of bill statement (NT dollar). X12 = amount of bill statement in September, 2005; X13 = amount of bill statement in August, 2005; . . .; X17 = amount of bill statement in April, 2005. 
X18-X23: Amount of previous payment (NT dollar). X18 = amount paid in September, 2005; X19 = amount paid in August, 2005; . . .;X23 = amount paid in April, 2005.

Methodology
Loading the Dataset: The code uses the pandas library to read the dataset from an Excel file and extract the relevant features and target.

Train-Test Split: The dataset is split into training and testing subsets using the train_test_split function from scikit-learn. This allows for independent training and evaluation of the Decision Tree classifier.

Decision Tree Classifier: A Decision Tree classifier is created using the DecisionTreeClassifier class from scikit-learn. The classifier is trained on the training data.

Classifier Evaluation: The accuracy of the trained classifier is computed using the testing data to assess its performance.

Feature Importance Analysis: The trained Decision Tree classifier is used to calculate feature importances. These importances reflect the influence of each feature in the classification process.

Visualization: Feature importances are plotted using matplotlib to create a bar plot. The plot provides a clear visual representation of the importance rankings of different features.

Output
Upon running the code, you will observe the following:

The accuracy of the trained Decision Tree classifier on the testing data.
A bar plot illustrating the relative importance of different features in the dataset according to the Decision Tree classifier.
The list of if-then rules for classification 

