# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 17:43:53 2021

@author: primi
"""


import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.model_selection import cross_val_score

df = pd.read_csv('glassdoor_data_prepossed.csv')

# choose relevant columns 
df.columns

df_model = df[['avg_salary','Rating','Size','Type of ownership','Industry','Sector','Revenue','num_comp','Hourly_salary','Employer_provided_salary','job_state','same_state','age','python_yn','spark','aws','excel','job_simp','seniority','desc_len']]
df_dum = pd.get_dummies(df_model)
df_dum

#import libraries for train- test split 
from sklearn.model_selection import train_test_split
# get the values for axis ----- categories on x and avg salary on y
X = df_dum.drop('avg_salary', axis =1)
Y=df_dum.avg_salary.values

X_train, X_test, Y_train, Y_test=  train_test_split(X,Y,test_size=0.2, random_state= 42)



# multiple linear regression
from sklearn import linear_model

ml_regr = linear_model.LinearRegression()
ml_regr.fit(X_train, Y_train)
ml_pred = ml_regr.predict(X_test)
#multiple linear regression evaluation
cross_val_score_ml =np.mean(cross_val_score(ml_regr,X_train,Y_train, scoring = 'neg_mean_absolute_error', cv= 5))
print('Mean error for Ml',cross_val_score_ml)

'''from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X_train)
poly_regr= linear_model.LinearRegression()
poly_regr.fit(X_poly, Y_train)

cross_val_score_poly =np.mean(cross_val_score(poly_regr,X_train,Y_train, scoring = 'neg_mean_absolute_error', cv= 3))
print('Mean error for Ml',cross_val_score_poly)
'''
from sklearn.linear_model import Lasso

alpha = []
error = []
for i in range(1,100):
    alpha.append(i/100)
    Lasso_regr = Lasso(alpha=(i/100))
    error.append(np.mean(cross_val_score(Lasso_regr,X_train,Y_train, scoring = 'neg_mean_absolute_error', cv= 5)))
  
df_err = pd.DataFrame(zip(alpha,error), columns = ['alpha','error'])
print(df_err[df_err.error == max(df_err.error)])

Lasso_regr = Lasso(alpha=0.02)
Lasso_regr.fit(X_train,Y_train)
print('mean error for lasso: ',np.mean(cross_val_score(Lasso_regr,X_train,Y_train, scoring = 'neg_mean_absolute_error', cv= 5)))



from sklearn.linear_model import Ridge
alpha = []
error = []
for i in range(1,100):
    alpha.append(i/100)
    Ridge_regr = Ridge(alpha=(i/100))
    error.append(np.mean(cross_val_score(Ridge_regr,X_train,Y_train, scoring = 'neg_mean_absolute_error', cv= 7)))
    
df_err = pd.DataFrame(zip(alpha,error), columns = ['alpha','error'])
print(df_err[df_err.error == max(df_err.error)])
Ridge_regr = Ridge(alpha=.99)
Ridge_regr.fit(X_train,Y_train)
print('mean error for Ridge: ',np.mean(cross_val_score(Ridge_regr,X_train,Y_train, scoring = 'neg_mean_absolute_error', cv= 5)))


# Support vector regression
from sklearn.svm import SVR
svr_regr=SVR(kernel='poly')
svr_regr.fit(X_train,Y_train)

print('mean error for SVR: ',np.mean(cross_val_score(svr_regr,X_train,Y_train, scoring = 'neg_mean_absolute_error', cv= 5)))
# decision tree regresson
from sklearn.tree import DecisionTreeRegressor
ds_regr = DecisionTreeRegressor(random_state = 0)
ds_regr.fit(X_train, Y_train)
print('mean error for Decision tree: ',np.mean(cross_val_score(ds_regr,X_train,Y_train, scoring = 'neg_mean_absolute_error', cv= 5)))

# random forest regresson
from sklearn.ensemble import RandomForestRegressor
rf_regr = RandomForestRegressor()
print('mean error for Random Forest Regressor: ',np.mean(cross_val_score(rf_regr,X_train,Y_train, scoring = 'neg_mean_absolute_error', cv= 5)))
# tuning random forest regressor model
#from sklearn.model_selection import GridSearchCV
#parameters = {'n_estimators':range(10,300,10), 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')}
#gs = GridSearchCV(rf_regr,parameters,scoring='neg_mean_absolute_error',cv=2)
#gs.fit(X_train, Y_train)
#print(gs.best_estimator_)

#print('mean error for gs: ',np.mean(cross_val_score(gs,X_train,Y_train, scoring = 'neg_mean_absolute_error', cv= 3)))

