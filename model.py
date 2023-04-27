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
    labels = model.classes_
    ConfusionMatrixDisplay(confusion_matrix(y_train,y_pred,labels=labels),display_labels=labels).plot()
    plt.show()

def class_report(y_train,y_train_pred):
    return pd.DataFrame(classification_report(y_train,y_train_pred,output_dict=True))

### Modeling ### 
def dt(X_train,y_train,X_validate,y_validate,features,depth):
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
    return dt(X_train,y_train,X_validate,y_validate,f1,8)

def val_rf():
    return rf(X_train,y_train,X_validate,y_validate,f2,3,7)

def val_knn():
    return knn(X_train,y_train,X_validate,y_validate,f3,15,'uniform','auto')

def val_lr():
    return lr(X_train,y_train,X_validate,y_validate,f4,10)

### TEST ###
def test_model():
    l=LogisticRegression(C=10,random_state=42,max_iter=500)
    l.fit(X_train[f4], y_train)
    t = l.score(X_test[f4], y_test)
    print('Logistic Regression','\n')
    print(f'Accuracy on test: {round(t*100,2)}','\n')



