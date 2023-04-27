'''Model Telco data'''

########## IMPORTS ##########
import numpy as np
import pandas as pd
import itertools

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import wrangle as w
import explore as e

### VARIABLES ###
telco_df = w.wrangle_telco_data()
train, validate, test = w.split_data(telco_df,'churn')

f1 = ['senior_citizen',
        'tenure',
        'contract_type_Two year',
        'internet_service_type_Fiber optic',
        'payment_type_Credit card (automatic)']
f2 = ['senior_citizen',
        'partner',
        'dependents',
        'paperless_billing',
        'monthly_charges',
        'total_charges',
        'tenure',
        'online_security_No internet service',
        'online_security_Yes',
        'online_backup_No internet service',
        'online_backup_Yes',
        'device_protection_No internet service',
        'device_protection_Yes',
        'tech_support_No internet service',
        'tech_support_Yes',
        'streaming_tv_No internet service',
        'streaming_tv_Yes',
        'streaming_movies_No internet service',
        'streaming_movies_Yes',
        'contract_type_One year',
        'contract_type_Two year',
        'internet_service_type_Fiber optic',
        'internet_service_type_None',
        'payment_type_Credit card (automatic)',
        'payment_type_Electronic check',
        'payment_type_Mailed check']
f3 = ['senior_citizen',
        'paperless_billing',
        'monthly_charges',
        'total_charges',
        'tenure',
        'contract_type_One year',
        'contract_type_Two year',
        'internet_service_type_Fiber optic',
        'internet_service_type_None',
        'payment_type_Credit card (automatic)']
f4 = ['senior_citizen',
        'total_charges',
        'tenure',
        'contract_type_One year',
        'internet_service_type_Fiber optic',
        'payment_type_Credit card (automatic)']

### X and y ###

def xy(train,validate,test):
    """
    The function takes in three datasets and returns the X and y variables for each dataset after
    dropping certain columns.
    
    :param train: a pandas DataFrame containing the training data
    :param validate: The `validate` parameter is a pandas DataFrame containing the validation dataset.
    It is used in the `xy` function to extract the features and target variables for the validation set
    :param test: The test parameter is a pandas DataFrame containing the test dataset
    :return: The function `xy` returns six variables: `X_train`, `X_validate`, `X_test`, `y_train`,
    `y_validate`, and `y_test`. These variables are used for training, validation, and testing of a
    machine learning model. `X_train`, `X_validate`, and `X_test` are the feature matrices for the
    training, validation, and testing sets respectively, while
    """
    tel_obj = train.select_dtypes(include='object').columns.to_list()
    for i in ['churn','phone_service','female','multiple_lines_No phone service','multiple_lines_Yes']:
        tel_obj.append(i)
    X_train = train.drop(columns=tel_obj)
    X_validate = validate.drop(columns=tel_obj)
    X_test = test.drop(columns=tel_obj)
    y_train = train.churn
    y_validate = validate.churn
    y_test = test.churn
    return X_train,X_validate,X_test,y_train,y_validate,y_test

X_train,X_validate,X_test,y_train,y_validate,y_test = xy(train,validate,test)

### PLOT ###
def score_plot(data, diff=0.1, score='avg_score', score_v=.7):
    """
    The function takes in data and plots the train, validation, and average scores for a classification
    model.
    
    :param data: The input data for the function, which should be a pandas DataFrame containing the
    model performance metrics
    :param diff: The maximum difference allowed between the train and validation scores for a data point
    to be included in the plot
    :param score: The parameter 'score' is a string that represents the name of the column in the input
    data that contains the score to be plotted. The default value is 'avg_score', defaults to avg_score
    (optional)
    :param score_v: The minimum value for the score column to be included in the plot
    :return: a pandas DataFrame containing the sorted and filtered data, and also displays a plot of the
    model performance.
    """
    df = data[(data.diff_score<diff)&(data[score]>score_v)].sort_values([score, 'diff_score'], ascending=[False, True]).reset_index()
    df = df.drop(columns='index')
    # plot
    plt.figure(figsize=(8, 5))
    plt.plot(df.index, df.train_score, label='train', marker='o', color='blue')
    plt.plot(df.index, df.val_score, label='validation', marker='o', color='orange')
    plt.fill_between(df.index, df.train_score, df.val_score, alpha=0.2, color='gray')
    plt.plot(df.index, df.avg_score, label='avg_score', marker='o', color='black')
    plt.xlabel('index', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Classification Model Performance', fontsize=18)
    plt.legend(fontsize=12)
    plt.show()
    return df

### Metrics ###
def cmd(y_train,y_pred,model):
    """
    This function generates and displays a confusion matrix for a given set of predicted and actual
    labels using a trained model.
    
    :param y_train: The true labels of the training data
    :param y_pred: The predicted values of the target variable generated by a machine learning model
    :param model: The machine learning model that was used to make predictions on the training data
    """
    labels = model.classes_
    ConfusionMatrixDisplay(confusion_matrix(y_train,y_pred,labels=labels),display_labels=labels).plot()
    plt.show()

def class_report(y_train,y_train_pred):
    """
    This function returns a pandas DataFrame containing the classification report of the training data
    based on the predicted values.
    
    :param y_train: y_train is a numpy array or pandas series containing the true labels of the training
    set. It is typically used in machine learning models to train the algorithm to predict the target
    variable
    :param y_train_pred: y_train_pred is a numpy array or a pandas series containing the predicted
    target values for the training set. It is the output of a machine learning model that has been
    trained on the training set
    :return: A pandas DataFrame containing the classification report for the input training data and
    predicted training data, with output in dictionary format.
    """
    return pd.DataFrame(classification_report(y_train,y_train_pred,output_dict=True))

### Modeling ### 
def dt(X_train,y_train,X_validate,y_validate,features,depth):
    """
    The function `dt` trains a decision tree classifier on given training data and evaluates its
    accuracy on both training and validation data.
    
    :param X_train: X_train is the training set of input features used to train the decision tree model
    :param y_train: The target variable for the training set
    :param X_validate: X_validate is a pandas DataFrame containing the validation set features. It is
    used as input to evaluate the performance of the decision tree model on the validation set
    :param y_validate: The y_validate parameter is the target variable for the validation dataset. It is
    the set of true values that the model will be compared against to evaluate its performance on the
    validation data
    :param features: The features parameter is a list of column names that are used as input variables
    for the decision tree model. These are the variables that the model will use to make predictions
    :param depth: The maximum depth of the decision tree. It controls how many levels the tree can have.
    A higher depth can lead to overfitting, while a lower depth can lead to underfitting
    """
    tree = DecisionTreeClassifier(max_depth=depth,random_state=42)
    tree = tree.fit(X_train[features],y_train)
    # y_pred = tree.predict(X_train[features])
    # accuracies
    y_train_acc = tree.score(X_train[features],y_train)
    y_validate_acc = tree.score(X_validate[features],y_validate)
    print('Decision Tree','\n')
    print(f'Accuracy on train: {round(y_train_acc*100,2)}','\n')
    print(f'Accuracy on validate: {round(y_validate_acc*100,2)}')
    # cmd(y_train,y_pred,tree)

def rf(X_train,y_train,X_validate,y_validate,features,leaf,depth):
    """
    The function trains a random forest classifier on given training data and evaluates its accuracy on
    both training and validation data.
    
    :param X_train: X_train is the training set of input features used to train the random forest model
    :param y_train: The target variable for the training set
    :param X_validate: X_validate is a validation set of input features used to evaluate the performance
    of a machine learning model trained on X_train and y_train. In this case, it is being used as an
    input to the rf function, which fits a random forest classifier on X_train and y_train using the
    specified hyperparameters
    :param y_validate: `y_validate` is the target variable for the validation set. It is the set of true
    labels for the validation set that will be used to evaluate the performance of the random forest
    classifier
    :param features: The list of features used to train the random forest model
    :param leaf: The minimum number of samples required to be at a leaf node in the decision tree. This
    parameter controls the complexity of the tree and can help prevent overfitting
    :param depth: The maximum depth of the decision tree in the random forest. It controls the maximum
    number of levels in each decision tree. A higher value of depth can lead to overfitting, while a
    lower value can lead to underfitting
    """
    rf = RandomForestClassifier(min_samples_leaf=leaf, max_depth=depth,random_state=42)
    rf.fit(X_train[features], y_train)
    # y_pred = rf.predict(X_train[features])
    # accuracies
    y_train_acc = rf.score(X_train[features], y_train)
    y_validate_acc = rf.score(X_validate[features], y_validate)
    print('Random Forest','\n')
    print(f'Accuracy on train: {round(y_train_acc*100,2)}','\n')
    print(f'Accuracy on validate: {round(y_validate_acc*100,2)}')
    # cmd(y_train,y_pred,rf)

def knn(X_train,y_train,X_validate,y_validate,features,n,weight,algorithm):
    """
    The function "knn" trains a K-Nearest Neighbors classifier on a given dataset and returns the
    accuracy of the model on both the training and validation sets.
    
    :param X_train: The training set features
    :param y_train: The target variable for the training set
    :param X_validate: X_validate is a validation set of input features used to evaluate the performance
    of the KNN model. It is a subset of the overall dataset that is not used during training, but rather
    used to test the model's ability to generalize to new data
    :param y_validate: `y_validate` is the target variable for the validation set. It is a 1-dimensional
    array or pandas series containing the true values of the target variable for the validation set. The
    purpose of using `y_validate` is to evaluate the performance of the KNN model on unseen data and to
    tune
    :param features: The list of features used for training and predicting the model
    :param n: The number of nearest neighbors to consider for classification
    :param weight: The weight parameter in the KNeighborsClassifier function determines how the
    algorithm weights the neighbors when making predictions. It can take on two values: 'uniform'
    (default) where all neighbors are weighted equally, or 'distance' where closer neighbors are given
    more weight than farther neighbors
    :param algorithm: The algorithm parameter in the KNeighborsClassifier function specifies the
    algorithm used to compute the nearest neighbors. It can be set to 'auto', 'ball_tree', 'kd_tree', or
    'brute'. The default value is 'auto', which selects the most appropriate algorithm based on the
    input data
    """
    k = KNeighborsClassifier(n_neighbors=n,weights=weight,algorithm=algorithm)
    k.fit(X_train[features], y_train)
    # y_pred = k.predict(X_train[features])
    # accuracies
    y_train_acc = k.score(X_train[features], y_train)
    y_validate_acc = k.score(X_validate[features], y_validate)
    print('K Nearest Neighbors','\n')
    print(f'Accuracy on train: {round(y_train_acc*100,2)}','\n')
    print(f'Accuracy on validate: {round(y_validate_acc*100,2)}')
    # cmd(y_train,y_pred,k)

def lr(X_train,y_train,X_validate,y_validate,features,c):
    """
    The function performs logistic regression on training and validation data and prints the accuracy
    scores.
    
    :param X_train: X_train is the training set of independent variables (features) used to train the
    logistic regression model
    :param y_train: The target variable for the training set
    :param X_validate: The validation set features
    :param y_validate: The parameter y_validate is the target variable for the validation set. It is the
    set of true values that the model will be compared against to evaluate its performance on the
    validation set
    :param features: The features parameter is a list of column names that are used as input variables
    for the logistic regression model. These variables are used to predict the target variable
    :param c: The regularization parameter for logistic regression. It controls the trade-off between
    fitting the training data well and avoiding overfitting. A smaller value of c will increase the
    regularization strength and a larger value of c will decrease it
    """
    l = LogisticRegression(C=c,random_state=42,max_iter=250)
    l.fit(X_train[features],y_train)
    # y_pred = l.predict(X_train[features])
    # accuracies
    y_train_acc = l.score(X_train[features],y_train)
    y_validate_acc = l.score(X_validate[features],y_validate)
    print('Logistic Regression','\n')
    print(f'Accuracy on train: {round(y_train_acc*100,2)}','\n')
    print(f'Accuracy on validate: {round(y_validate_acc*100,2)}')
    # cmd(y_train,y_pred,l)


### VALIDATE ###
def val_dt():
    """
    The function `val_dt()` returns the f1 score of a decision tree model trained on `X_train` and
    `y_train`, and validated on `X_validate` and `y_validate`, with a maximum depth of 8.
    :return: The function `val_dt()` is returning the output of the function `dt()` with the arguments
    `X_train`, `y_train`, `X_validate`, `y_validate`, `f1`, and `8`.
    """
    return dt(X_train,y_train,X_validate,y_validate,f1,8)

def val_rf():
    """
    The function `val_rf()` returns the result of running a random forest model with specified
    parameters on training and validation data.
    :return: The function `val_rf()` is returning the output of the function `rf()` with the following
    arguments: `X_train`, `y_train`, `X_validate`, `y_validate`, `f2`, `3`, and `7`. Without knowing the
    implementation of the `rf()` function, it is not possible to determine what exactly is being
    returned.
    """
    return rf(X_train,y_train,X_validate,y_validate,f2,3,7)

def val_knn():
    """
    The function `val_knn()` returns the result of applying the k-nearest neighbors algorithm with
    specified parameters to a validation set.
    :return: The function `val_knn()` is returning the output of the `knn()` function with the following
    parameters:
    """
    return knn(X_train,y_train,X_validate,y_validate,f3,15,'uniform','auto')

def val_lr():
    """
    The function `val_lr()` returns the result of running a linear regression model with 10-fold
    cross-validation on training and validation data.
    :return: The function `val_lr()` is returning the output of the function `lr()` with the arguments
    `X_train`, `y_train`, `X_validate`, `y_validate`, `f4`, and `10`.
    """
    return lr(X_train,y_train,X_validate,y_validate,f4,10)

### TEST ###
def test_model():
    """
    The function trains a logistic regression model with specified parameters and prints the accuracy
    score on the test data.
    """
    l=LogisticRegression(C=10,random_state=42,max_iter=500)
    l.fit(X_train[f4], y_train)
    t = l.score(X_test[f4], y_test)
    print('Logistic Regression','\n')
    print(f'Accuracy on test: {round(t*100,2)}','\n')



