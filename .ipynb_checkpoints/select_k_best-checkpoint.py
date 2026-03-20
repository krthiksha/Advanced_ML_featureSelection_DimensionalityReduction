import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# feature selection
def selectkbest(indep_x,dep_y,n):
    test=SelectKBest(score_func=chi2,k=n)
    fit1=test.fit(indep_x,dep_y)    # fit - learns which features are important features 
    seleted_features=fit1.transform(indep_x)     #transform - select those features 
    # seleted_features = test.fit_transform(indep_x,dep_y)        # fit & transform in one step 
    return seleted_features

# tran test split and standardization 
def split_scaler(indep_x,dep_y):
    x_train,x_test,y_train,y_test = train_test_split(indep_x,dep_y,test_size=0.20,random_state=0)
    # standardization
    sc=StandardScaler()
    x_train=sc.fit_transform(x_train)
    x_test=sc.transform(x_test)
    return  x_train,x_test,y_train,y_test

# model prediction & evaluation 
def cm_pred_eval(classifier,x_test):
    # prediction
    y_pred=classifier.predict(x_test)
    # confusion matrix
    from sklearn.metrics import confusion_matrix
    cm=confusion_matrix(y_test,y_pred)
    # classification report 
    from sklearn.metrics import classification_report
    classi_report=classification_report(y_test,y_pred)
    # accuracy 
    from sklearn.metrics import accuracy_score
    accuracy=accuracy_score(y_test,y_pred)
    return classifier,accuracy,cm,classi_report,x_test,y_test

### --------------- classification algortihms -----------------------------
# logistic regression 
def logistic(x_train,y_train,x_test):
    # model creation
    from sklearn.linear_model import LogisticRegression
    classifier=LogisticRegression(random_state=0)
    classifier.fit(x_train,y_train)
    # model prediction & evaluation
    classifier,accuracy,cm,classi_report,x_test,y_test = cm_pred_eval(classifier,x_test)
    return classifier,accuracy,cm,classi_report,x_test,y_test

# KNN 
def knn(x_train,y_train,x_test):
    # model creation
    from sklearn.neighbors import KNeighborsClassifier
    classifier=KNeighborsClassifier(n_neighbors=5,p=2, metric='minkowski')
    classifier.fit(x_train,y_train)
    # model prediction & evaluation
    classifier,accuracy,cm,classi_report,x_test,y_test = cm_pred_eval(classifier,x_test)
    return classifier,accuracy,cm,classi_report,x_test,y_test

# SVM - linear 
def svm_linear(x_train,y_train,x_test):
    # model creation
    from sklearn.svm import SVC
    classifier=SVC(kernel='linear',random_state=0)
    classifier.fit(x_train,y_train)
    # model prediction & evaluation
    classifier,accuracy,cm,classi_report,x_test,y_test = cm_pred_eval(classifier,x_test)
    return classifier,accuracy,cm,classi_report,x_test,y_test

# SVM - Non linear 
def svm_nonlinear(x_train,y_train,x_test):
    # model creation
    from sklearn.svm import SVC
    classifier=SVC(kernel='rbf',random_state=0)
    classifier.fit(x_train,y_train)
    # model prediction & evaluation
    classifier,accuracy,cm,classi_report,x_test,y_test = cm_pred_eval(classifier,x_test)
    return classifier,accuracy,cm,classi_report,x_test,y_test

# navie bayes
def naive(x_train,y_train,x_test):
    # model creation 
    from sklearn.naive_bayes import GaussianNB
    classifier=GaussianNB()
    classifier.fit(x_train,y_train)
    # model prediction & evaluation
    classifier,accuracy,cm,classi_report,x_test,y_test = cm_pred_eval(classifier,x_test)
    return classifier,accuracy,cm,classi_report,x_test,y_test

# decision tree
def decision_tree(x_train,y_train,x_test):
     # model creation 
    from sklearn.tree import DecisionTreeClassifier
    classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
    classifier.fit(x_train,y_train)
    # model prediction & evaluation
    classifier,accuracy,cm,classi_report,x_test,y_test = cm_pred_eval(classifier,x_test)
    return classifier,accuracy,cm,classi_report,x_test,y_test

# random forest
def random_forest(x_train,y_train,x_test):
     # model creation 
    from sklearn.ensemble import RandomForestClassifier
    classifier=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
    classifier.fit(x_train,y_train)
    # model prediction & evaluation
    classifier,accuracy,cm,classi_report,x_test,y_test = cm_pred_eval(classifier,x_test)
    return classifier,accuracy,cm,classi_report,x_test,y_test


## ------------ final table for comparision 
def selectk_classification(acclog,accsvml,accsvmnl,accknn,accnav,accdes,accrf):
    dataframe=pd.DataFrame(index=['Chi square'],columns=['logistic','KNN','SVMl','SVMnl','NB','decision','random forest'])
    for number,idex in enumerate(dataframe.index):
        dataframe.loc[idex,'logistic']=acclog[number]
        dataframe.loc[idex,'KNN']=accsvml[number]
        dataframe.loc[idex,'SVMl']=accsvmnl[number]
        dataframe.loc[idex,'SVMnl']=accknn[number]
        dataframe.loc[idex,'NB']=accnav[number]
        dataframe.loc[idex,'decision']=accdes[number]
        dataframe.loc[idex,'random forest']=accrf[number]
        return dataframe

data_original = pd.read_csv('preprocessed_CKD.csv')
data_ckd= data_original
data_ckd

# ip op split 
indep_x=data_ckd.drop(['classification_notckd'],axis=1)
dep_y = data_ckd['classification_notckd']

# feature selection 
kBest=selectkbest(indep_x,dep_y,5)

acclog=[]
accsvml=[]
accsvmnl=[]
accknn=[]
accnav=[]
accdes=[]
accrf=[]

x_train,x_test,y_train,y_test=split_scaler(kBest,dep_y)

# algorithms 
classifier,accuracy,cm,classi_report,x_test,y_test=logistic(x_train,y_train,x_test)
acclog.append(accuracy)
classifier,accuracy,cm,classi_report,x_test,y_test=knn(x_train,y_train,x_test)
accknn.append(accuracy)
classifier,accuracy,cm,classi_report,x_test,y_test=svm_linear(x_train,y_train,x_test)
accsvml.append(accuracy)
classifier,accuracy,cm,classi_report,x_test,y_test=svm_nonlinear(x_train,y_train,x_test)
accsvmnl.append(accuracy)
classifier,accuracy,cm,classi_report,x_test,y_test=naive(x_train,y_train,x_test)
accnav.append(accuracy)
classifier,accuracy,cm,classi_report,x_test,y_test=decision_tree(x_train,y_train,x_test)
accdes.append(accuracy)
classifier,accuracy,cm,classi_report,x_test,y_test=random_forest(x_train,y_train,x_test)
accrf.append(accuracy)


result = selectk_classification(acclog,accsvml,accsvmnl,accknn,accnav,accdes,accrf)