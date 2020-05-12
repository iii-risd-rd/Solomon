# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import datetime
from itertools import permutations
random_seed = 64
#readfromsqlite3---------------------------------------------------------------
import sqlite3
db_path = r'F:\SOLOMON\SOLOMON_v2.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute('''SELECT * FROM SOLOMON''')
rows = cursor.fetchall()
names = list(map(lambda x: x[0], cursor.description))
conn.commit()
conn.close()
data = pd.DataFrame(rows, columns = names)
data = data.drop(columns = ['num','Date'])
time_list = []
ts = datetime.datetime.now().replace(microsecond = 0)
for i in range(len(data)):
    add_time = datetime.timedelta(seconds = i)
    time_list.extend([ts + add_time])
data['Time'] = time_list
#remove useless col
select_col = ['BacterialConcentration','Formaldehyde','Area','Time']
data_v1 = data[select_col]
#group by area
groupby_data = dict(list(data_v1.groupby(['Area'])))
#seperate each area to feature data
all_area = data_v1['Area'].unique().tolist()
group_data_dict = {}
for i in all_area:
    tmp_data = groupby_data[i]
    groupby_tmp_data = list(tmp_data.groupby(['Time']))
    time_dif = list(map(lambda x: x[0],groupby_tmp_data))
    time_dif = pd.DataFrame(time_dif, columns = ['time'])
    time_dif['time_dif'] = time_dif['time']- time_dif['time'].shift(1)
    #find partition 
    seperate_point = [0]
    seperate_point.extend(time_dif[time_dif['time_dif']>datetime.timedelta(seconds = 1)].index)
    seperate_point.extend([len(time_dif)])
    tmp_data_list = []
    for s,e in zip(seperate_point[:-1], seperate_point[1:]):
        e = e-1
        start_time = time_dif.loc[s,'time']
        end_time = time_dif.loc[e,'time']
        tmp_data_list.append(data_v1[(data_v1['Time']>=start_time)&(data_v1['Time']<= end_time)])
    group_data_dict[i] = tmp_data_list

def get_average(data):
    data = data.reset_index(drop = True)
    bacter_average = data.BacterialConcentration.mean()
    formal_avearge = data.Formaldehyde.mean()
    area = data.loc[0:,'Area'].value
    time = data.loc[int(len(data)/2),'Time']
    return pd.DataFrame([[bacter_average,formal_avearge,area,time]], columns = ['Bacterial_average', 'Formaldehyde_average', 'Area', 'Time'])

average_data_dict = {}
for key, value in group_data_dict.items():
    value = [i for i in value if len(i) > 5]
    value = list(map(lambda x: get_average(x), value))
    average_data_dict[key] = value

def compute_all_different(data_list):
    output_data_list = []
    for i, j in zip(data_list[:-1], data_list[1:]):
        time_dif = (j.loc[0,'Time']-i.loc[0,'Time']).total_seconds()
        bacterial_dif = j.loc[0,'Bacterial_average']-i.loc[0,'Bacterial_average']
        formal_dif = j.loc[0,'Formaldehyde_average']-i.loc[0,'Formaldehyde_average']
        output_data_list.append([time_dif,bacterial_dif,formal_dif])
    return pd.DataFrame(output_data_list, columns = ['time_dif', 'bacteral', 'Formaldehyde'])
    
ready_data_dict = {}
for key, value in group_data_dict.items():
    ready_data_dict['key'] = compute_all_different(value)







#unsupervised------------------------------------------------------------------
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors = 5, algorithm = 'ball_tree').fit(x)



#select a area from datalist & splite to train test
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split()

#supervised--------------------------------------------------------------------
#pipeline lib
from sklearn.pipelint import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import GridSearchCV
#preprocess lib
from sklearn.preprocessing import StandardScaler
#features lib
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
#models lib
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
#criteria lib
from sklearn.metrics import mean_squared_error
#define methods
std_sclr = StandardScaler()
poly  = PolynomialFeatures()
pca = PCA()
rf = RandomForestClassifier(random_state = random_seed,n_estimators=100)
from_model = SelectFromModel(estimator = rf)
lr = LinearRegression()
dt = DecisionTreeClassifier()
xgb = XGBClassifier()

#bulid feature pipeline
feature_methods = [('poly', poly),
                   ('pca', pca)]
creat_features = FeatureUnion(feature_methods)
#bulid data pipeline
pipeline = Pipeline([('sclr',std_sclr),
                     ('create_fea',creat_features),
                     ('from_model', from_model),
                     ('clf', lr)])
#training
trained_model = pipeline.fit(x_train, y_train)
#testing
y_test_pred = trained_model.predict(x_test)
y_pred = mean_squared_error(y_test, y_test_pred)
