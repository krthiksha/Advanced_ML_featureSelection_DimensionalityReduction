import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, RFE
import pickle
from sklearn.model_selection import train_test_split

# ------------------------------- functions  -------------------------------
# feature selection 
def select_k_best(indep_x,dep_y,n):
    test=SelectKBest(score_func=chi2,k=n)
    selected_features = test.fit_transform(indep_x,dep_y)
    return selected_features

# regression model predict & evaluation 
def r2_score_pred(regressor,x_test,y_test):
    y_pred=regressor.predict(x_test)
    from sklearn.metrics import r2_score
    r2_score=r2_score(y_test,y_pred)
    return r2_score 

# tran test split and standardization 
def split_scaler(indep_x,dep_y):
    x_train,x_test,y_train,y_test = train_test_split(indep_x,dep_y,test_size=0.20,random_state=0)
    # standardization
    sc=StandardScaler()
    x_train=sc.fit_transform(x_train)
    x_test=sc.transform(x_test)
    return  x_train,x_test,y_train,y_test
    
# regression algorithms 
# linear
def linear(x_train,y_train,x_test):
    from sklearn.linear_model import LinearRegression
    regressor=LinearRegression()
    regressor.fit(x_train,y_train)
    r2 = r2_score_pred(regressor,x_test,y_test)
    return r2 
    
# SVR -linear
def svrl(x_train,y_train,x_test):
    from sklearn.svm import SVR
    regressor=SVR(kernel='linear')
    regressor.fit(x_train,y_train)
    r2 = r2_score_pred(regressor,x_test,y_test)
    return r2 
    
# SVR-non linear
def svrnl(x_train,y_train,x_test):
    from sklearn.svm import SVR
    regressor=SVR(kernel='rbf')
    regressor.fit(x_train,y_train)
    r2 = r2_score_pred(regressor,x_test,y_test)
    return r2 
    
# decision tree
def decisionReg(x_train,y_train,x_test):
    from sklearn.tree import DecisionTreeRegressor
    regressor=DecisionTreeRegressor(criterion='squared_error',random_state=0)
    regressor.fit(x_train,y_train)
    r2 = r2_score_pred(regressor,x_test,y_test)
    return r2 
    
# random forest 
def randomForestReg(x_train,y_train,x_test):
    from sklearn.ensemble import RandomForestRegressor
    regressor=RandomForestRegressor(n_estimators=10,random_state=0)
    regressor.fit(x_train,y_train)
    r2 = r2_score_pred(regressor,x_test,y_test)
    return r2 


# select features table 
def select_k_regression(acclinear,accsvrl,accsvrnl,accdes,accrf):
    dataframe=pd.DataFrame(index=['chi square'],columns=['linear','svrl','svrnl','decisionReg','randomForestReg'])
    for number,idex in enumerate(dataframe.index):
        dataframe.loc[idex,'linear']=acclinear[number]
        dataframe.loc[idex,'svrl']=accsvrl[number]
        dataframe.loc[idex,'svrnl']=accsvrnl[number]
        dataframe.loc[idex,'decisionReg']=accdes[number]
        dataframe.loc[idex,'randomForestReg']=accrf[number]
        return dataframe

# main program 
data_original = pd.read_csv('preprocessed_CKD.csv')
data_ckd= data_original
data_ckd

data_ckd=pd.get_dummies(data_ckd,drop_first=True,dtype=int)
data_ckd.head()

# ip op split 
indep_x=data_ckd.drop(['classification_notckd'],axis=1)
dep_y = data_ckd['classification_notckd']

# feature selection 
no_of_features= 20
kBest=selectkbest(indep_x,dep_y,no_of_features)

acclinear=[]
accsvrl=[]
accsvrnl=[]
accdes=[]
accrf=[]

x_train,x_test,y_train,y_test=split_scaler(kBest,dep_y)

# algorithms 
r2=linear(x_train,y_train,x_test)
acclinear.append(r2)
r2=svrl(x_train,y_train,x_test)
accsvrl.append(r2)
r2=svrnl(x_train,y_train,x_test)
accsvrnl.append(r2)
r2=decisionReg(x_train,y_train,x_test)
accdes.append(r2)
r2=randomForestReg(x_train,y_train,x_test)
accrf.append(r2)



result_regression = select_k_regression(acclinear,accsvrl,accsvrnl,accdes,accrf)
print(result_regression)