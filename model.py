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
    plt.title('Logistic Regression Classifier Performance', fontsize=18)
    plt.legend(fontsize=12)
    plt.show()
    return df

### Metrics ###
def cmd(ytr,y_pred,model):
    labels = model.classes_
    ConfusionMatrixDisplay(confusion_matrix(ytr,y_pred,labels=labels),display_labels=labels).plot()
    plt.show()

def class_report(ytr,ytr_pred):
    return pd.DataFrame(classification_report(ytr,ytr_pred,output_dict=True))

### Modeling ### 
def dt(Xtr,ytr,Xv,yv,features,depth):
    tree = DecisionTreeClassifier(max_depth=depth,random_state=42)
    tree = tree.fit(Xtr[features],ytr)
    # accuracies
    ytr_acc = tree.score(Xtr[features],ytr)
    yv_acc = tree.score(Xv[features],yv)
    print('Decision Tree','\n')
    print(f'Accuracy on train: {round(ytr_acc*100,2)}','\n')
    print(f'Accuracy on validate: {round(yv_acc*100,2)}')

def rf(Xtr,ytr,Xv,yv,features,leaf,depth):
    rf = RandomForestClassifier(min_samples_leaf=leaf, max_depth=depth,random_state=42)
    rf.fit(Xtr[features], ytr)
    ytr_acc = rf.score(Xtr[features], ytr)
    yv_acc = rf.score(Xv[features], yv)
    print('Random Forest','\n')
    print(f'Accuracy on train: {round(ytr_acc*100,2)}','\n')
    print(f'Accuracy on validate: {round(yv_acc*100,2)}')

def knn(Xtr,ytr,Xv,yv,features,n,weight,algorithm):
    k = KNeighborsClassifier(n_neighbors=n,weights=weight,algorithm=algorithm)
    k.fit(Xtr[features], ytr)
    ytr_acc = k.score(Xtr[features], ytr)
    yv_acc = k.score(Xv[features], yv)
    print('Random Forest','\n')
    print(f'Accuracy on train: {round(ytr_acc*100,2)}','\n')
    print(f'Accuracy on validate: {round(yv_acc*100,2)}')

def lr(Xtr,ytr,Xv,yv,features,c):
    l = LogisticRegression(C=c,random_state=42,max_iter=250)
    l.fit(Xtr[features],ytr)
    ytr_acc = l.score(Xtr[features],ytr)
    yv_acc = l.score(Xv[features],yv)
    print('Random Forest','\n')
    print(f'Accuracy on train: {round(ytr_acc*100,2)}','\n')
    print(f'Accuracy on validate: {round(yv_acc*100,2)}')





