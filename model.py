# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 14:20:03 2019

@author: Rishabh
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

df = pd.read_csv('data.csv')

# DATA CLEANING

df.drop(['Unnamed: 0', 'match_event_id', 'team_name', 'date_of_game',
          'match_id','team_id' ], axis=1, inplace=True)

tempdf = df['is_goal']

df = df.fillna(df.mode().iloc[0])

df['is_goal'] = tempdf

df.drop(['game_season' ], axis=1, inplace=True)

df2 = df[df['is_goal'].isnull()]

df1 = pd.concat([df, df2, df2]).drop_duplicates(keep=False)

# HANDLING CATEGORICAL DATA

le1 = preprocessing.LabelEncoder()
df1['area_of_shot'] = le1.fit_transform(df1['area_of_shot'])

df2['area_of_shot'] = le1.transform(df2['area_of_shot'])

le2 = preprocessing.LabelEncoder()
df1['shot_basics'] = le2.fit_transform(df1['shot_basics'])

df2['shot_basics'] = le2.transform(df2['shot_basics'])

le3 = preprocessing.LabelEncoder()
df1['range_of_shot'] = le3.fit_transform(df1['range_of_shot'])

df2['range_of_shot'] = le3.transform(df2['range_of_shot'])

le4 = preprocessing.LabelEncoder()
df1['home/away'] = le4.fit_transform(df1['home/away'])

df2['home/away'] = le4.transform(df2['home/away'])

le5 = preprocessing.LabelEncoder()
df1['lat/lng'] = le5.fit_transform(df1['lat/lng'])

df2['lat/lng'] = le5.transform(df2['lat/lng'])


df1['type_of_shot'].unique()

le6 = preprocessing.LabelEncoder()
df1['type_of_shot'] = le6.fit_transform(df1['type_of_shot'])

df2['type_of_shot'] = le6.transform(df2['type_of_shot'])


df1['type_of_combined_shot'].unique()

le7 = preprocessing.LabelEncoder()
df1['type_of_combined_shot'] = le7.fit_transform(df1['type_of_combined_shot'])

df2['type_of_combined_shot'] = le7.transform(df2['type_of_combined_shot'])

df2.drop(['is_goal' ], axis=1, inplace=True)

df1.drop(['shot_id_number' ], axis=1, inplace=True)


###################### continuous to intervals ######################

df1['location_x'] = pd.cut(df1['location_x'], bins=10, labels=False)
df2['location_x'] = pd.cut(df2['location_x'], bins=10, labels=False)


df1['location_y'] = pd.cut(df1['location_y'], bins=8, labels=False)
df2['location_y'] = pd.cut(df2['location_y'], bins=8, labels=False)


df1['remaining_sec'] = pd.cut(df1['remaining_sec'], bins=12, labels=False)
df2['remaining_sec'] = pd.cut(df2['remaining_sec'], bins=12, labels=False)


df1['distance_of_shot'] = pd.cut(df1['distance_of_shot'], bins=8, labels=False)
df2['distance_of_shot'] = pd.cut(df2['distance_of_shot'], bins=8, labels=False)


df1['remaining_min.1'] = pd.cut(df1['remaining_min.1'], bins=13, labels=False)
df2['remaining_min.1'] = pd.cut(df2['remaining_min.1'], bins=13, labels=False)


df1['power_of_shot.1'] = pd.cut(df1['power_of_shot.1'], bins=12, labels=False)
df2['power_of_shot.1'] = pd.cut(df2['power_of_shot.1'], bins=12, labels=False)


df1['knockout_match.1'] = pd.cut(df1['knockout_match.1'], bins=15, labels=False)
df2['knockout_match.1'] = pd.cut(df2['knockout_match.1'], bins=15, labels=False)


df1['remaining_sec.1'] = pd.cut(df1['remaining_sec.1'], bins=15, labels=False)
df2['remaining_sec.1'] = pd.cut(df2['remaining_sec.1'], bins=15, labels=False)


df1['distance_of_shot.1'] = pd.cut(df1['distance_of_shot.1'], bins=12, labels=False)
df2['distance_of_shot.1'] = pd.cut(df2['distance_of_shot.1'], bins=12, labels=False)


###################### EDA ########################################
import time
colnames = list(df1.columns.values)

###### histogram #######

''' Categorical data is label encoded and 
continuous data is transform into intervals with interval size of 10 units '''

## Plotting histogram for each column
for i in colnames:
    plt.figure(figsize=(16,6))
    ax = sns.countplot(x=i, data=df1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="right", fontsize=8)
    plt.title("Histogram for " + str(i), fontsize=14, fontweight='bold')

    if '/' in str(i):
        imagename = str(i)[:2] + '.jpg' 
        plt.savefig(imagename)
        plt.show()
    else:
        imagename = str(i) + '.jpg' 
        plt.savefig(imagename)
        plt.show()
    

####### BOXPLOTS #######


fig = sns.boxplot(y=df1['location_x'], data=df1)
fig.set_ylabel('location_x')
fig.set_title('Box Plot for location_x')
fig.figure.savefig("location_x_boxplot.jpg")
df1['location_x'].describe()



fig = sns.boxplot(y=df1['location_y'], data=df1)
fig.set_ylabel('location_y')
fig.set_title('Box Plot for location_y')
fig.figure.savefig("location_y_boxplot.jpg")
df1['location_y'].describe()



fig = sns.boxplot(y=df1['remaining_min'], data=df1)
fig.set_ylabel('remaining_min')
fig.set_title('Box Plot for remaining_min')
fig.figure.savefig("remaining_min_boxplot.jpg")
df1['remaining_min'].describe()



fig = sns.boxplot(y=df1['remaining_sec'], data=df1)
fig.set_ylabel('remaining_sec')
fig.set_title('Box Plot for remaining_sec')
fig.figure.savefig("remaining_sec_boxplot.jpg")
df1['remaining_sec'].describe()



fig = sns.boxplot(y=df1['distance_of_shot'], data=df1)
fig.set_title('Box Plot for distance_of_shot')
fig.set_ylabel('distance_of_shot')
fig.figure.savefig("distance_of_shot_boxplot.jpg")
df1['distance_of_shot'].describe()



fig = sns.boxplot(y=df1['remaining_min.1'], data=df1)
fig.set_ylabel('remaining_min.1')
fig.set_title('Box Plot for remaining_min.1')
fig.figure.savefig("remaining_min.1_boxplot.jpg")
df1['remaining_min.1'].describe()


fig = sns.boxplot(y=df1['power_of_shot.1'], data=df1)
fig.set_ylabel('power_of_shot.1')
fig.set_title('Box Plot for power_of_shot.1')
fig.figure.savefig("power_of_shot.1_boxplot.jpg")
df1['power_of_shot.1'].describe()


fig = sns.boxplot(y=df1['knockout_match.1'], data=df1)
fig.set_ylabel('knockout_match.1')
fig.set_title('Box Plot for knockout_match.1')
fig.figure.savefig("knockout_match.1_boxplot.jpg")
df1['knockout_match.1'].describe()



fig = sns.boxplot(y=df1['remaining_sec.1'], data=df1)
fig.set_ylabel('remaining_sec.1')
fig.set_title('Box Plot for remaining_sec.1')
fig.figure.savefig("remaining_sec.1_boxplot.jpg")
df1['remaining_sec.1'].describe()


fig = sns.boxplot(y=df1['distance_of_shot.1'], data=df1)
fig.set_ylabel('distance_of_shot.1')
fig.set_title('Box Plot for distance_of_shot.1')
fig.figure.savefig("distance_of_shot.1_boxplot.jpg")
df1['distance_of_shot.1'].describe()


######## HEATMAP #######

fig = df.corrwith(df.is_goal).plot.bar(figsize=(20,20), title='Correlation Plot', fontsize=15,
            rot=45, grid=True)

fig.figure.savefig("correlation.jpg")





# MODEL

trainset_y = df1['is_goal']


df1.drop(['is_goal' ], axis=1, inplace=True)

X = df1.iloc[:, :].values

y = trainset_y.values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


################# XGBOOST ##############################

import xgboost as xgb

xgb1 = xgb.XGBClassifier(booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.8, gamma=1, learning_rate=0.05,
       max_delta_step=1, max_depth=7, min_child_weight=5, missing=None,
       n_estimators=100, n_jobs=1, nthread=-1, objective='binary:logistic',
       random_state=0, reg_alpha=0.1, reg_lambda=0, scale_pos_weight=1,
       seed=None, silent=True, subsample=1)

xgb1.fit(X_train,y_train)

y_pred = xgb1.predict(X_test)


##################### K-NN to the Training set ##########
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 200, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


##################### Fitting Random Forest ############
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

# Evaluting
def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements

#Creating the confusion matrix with y_val and y_pred
cm = confusion_matrix(y_test, y_pred)

print("Accuracy : ", accuracy(cm))


######## CONCLUSION ##########


''' Out of XGBoost, KNN, Random Forest, XGBoost performs best. So we will
take XGBoost for making prediction '''





# TESTSET


s_id_no = df2['shot_id_number'].values

df2.drop(['shot_id_number' ], axis=1, inplace=True)

X2 = df2.iloc[:, :].values

y_pred_testset = xgb1.predict(X2)

y_predtestset_final = y_pred_testset.tolist()

s_id_no_final = s_id_no.tolist()

#creating CSV file
df3 = pd.DataFrame(data={"shot_id_number":s_id_no_final , "is_goal": y_predtestset_final})

df3 = df3.drop_duplicates(subset='shot_id_number', keep="last")
df3.to_csv("submission1.csv", sep=',',index=False)