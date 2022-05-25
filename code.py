import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyampd.ampd import find_peaks
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from scipy.signal import savgol_filter

PPG_datas = []

raw_training_data = pd.read_csv("smooth45k.csv",header=None)
PPG_data = raw_training_data[1]
PPG_datas.append(PPG_data)
PPG_datas = np.array(PPG_datas)

nppg = savgol_filter(PPG_datas, 51, 3)
ppg=[]
ppg.append(nppg[0][25:45160])
ppg= np.array(ppg)

def generate_whole_based_vector(X):
    vector = np.zeros(4 * SAMPLE_FREQ)
    vector[:len(X)] = X
    return vector

SAMPLE_FREQ = 250
#generate dataset
whole_based_vectors = []

for j in range(len(ppg)):
    sec_15 = 15*SAMPLE_FREQ
    PPG_data = ppg[j]
    PPG_peaks = find_peaks(PPG_data, scale=SAMPLE_FREQ)
    for i in range(2, PPG_peaks.shape[0]):
        X = PPG_data[PPG_peaks[i-1]:PPG_peaks[i]]
        if(len(X) < SAMPLE_FREQ):
            whole_based_vector = generate_whole_based_vector(X)

            whole_based_vectors.append(whole_based_vector)

whole_based_vectors = np.array(whole_based_vectors)
pca = PCA(n_components=43)
pca_whole_based_vectors = pca.fit_transform(whole_based_vectors)

# from project import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyampd.ampd import find_peaks
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import RandomizedSearchCV


PPG_Datas = []
ABP_datas = []

Raw_training_data = pd.read_csv( "rec_100.csv", header=None)
Raw_training_data = np.array(Raw_training_data)
#print(raw_training_data.shape)
PPG_Data = Raw_training_data[0].reshape(-1)
ABP_data = Raw_training_data[1].reshape(-1)

PPG_Datas.append(PPG_Data)
ABP_datas.append(ABP_data)

PPG_Datas = np.array(PPG_Datas)
ABP_datas = np.array(ABP_datas)

sAMPLE_FREQ=125

#generate dataset
Whole_based_vectors = []
SBP_data = []
DBP_data = []
MAP_data = []

for j in range(len(PPG_Datas)):
    sec_15 = 15*sAMPLE_FREQ
    PPG_Data = PPG_Datas[j]
    ABP_data = ABP_datas[j]
    PPG_peaks = find_peaks(PPG_Data, scale=sAMPLE_FREQ)
    for i in range(2, PPG_peaks.shape[0]):
        X = PPG_Data[PPG_peaks[i-1]:PPG_peaks[i]]
        if(len(X) < sAMPLE_FREQ):
            Whole_based_vector = generate_whole_based_vector(X)

            SBP = np.max(ABP_data[PPG_peaks[i-1]:PPG_peaks[i-1]+sec_15])
            DBP = np.min(ABP_data[PPG_peaks[i-1]:PPG_peaks[i-1]+sec_15])
            MAP = SBP/3 + 2*DBP/3

            Whole_based_vectors.append(Whole_based_vector)
            SBP_data.append(SBP)
            DBP_data.append(DBP)
            MAP_data.append(MAP)

Whole_based_vectors = np.array(Whole_based_vectors)
SBP_data = np.array(SBP_data)
DBP_data = np.array(DBP_data)
MAP_data = np.array(MAP_data)
pca = PCA(n_components=43)
Pca_whole_based_vectors = pca.fit_transform(Whole_based_vectors)

x_train1, x_test, SBP_train, SBP_test, DBP_train, DBP_test, MAP_train, MAP_test = train_test_split(
                                            Pca_whole_based_vectors, SBP_data, DBP_data, MAP_data, test_size=0.1, random_state=42)


#SUPPORT VECTOR REGRESSION

param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
clf = GridSearchCV(SVR(),param_grid,refit=True,verbose=2)
clf1 = GridSearchCV(SVR(),param_grid,refit=True,verbose=2)
clf.fit(x_train1, SBP_train)
clf1.fit(x_train1, DBP_train)


svr_SBP = SVR()
print('------SBP------')
print('start fitting...')
svr_SBP.fit(x_train1, SBP_train)
print('finish fitting...')
svr_SBP_predict = svr_SBP.predict(pca_whole_based_vectors)
print('finish predicting...')

svr_DBP = SVR()
print('------DBP------')
print('start fitting...')
svr_DBP.fit(x_train1, DBP_train)
print('finish fitting...')
svr_DBP_predict = svr_DBP.predict(pca_whole_based_vectors)
print('finish predicting...')

svr_MAP = SVR()
print('------MAP------')
print('start fitting...')
svr_MAP.fit(x_train1, MAP_train)
print('finish fitting...')
svr_MAP_predict = svr_MAP.predict(pca_whole_based_vectors)
print('finish predicting...')

print('-----Support Vector Regression-----')
print("---------------RESULT---------------")
print("                 SBP             DBP")
# print("Actual Value:   ", '%.3f       %.3f' %(np.mean(SBP_test),np.mean(DBP_test)))
print("Predicted Value:",'%.3f       %.3f' %(np.mean(svr_SBP_predict),np.mean(svr_DBP_predict)))

#HYPER TUNING

SBP_predict = clf.predict(pca_whole_based_vectors)
DBP_predict = clf1.predict(pca_whole_based_vectors)
print("---------------RESULT[BOOSTING]---------------")
print("Predicted Value:",'%.3f       %.3f' %(np.mean(SBP_predict),np.mean(DBP_predict)))

import requests

finalSBPSVR = '%.3f' %(np.mean(SBP_predict))
finalDBPSVR = '%.3f' %( np.mean(DBP_predict))


# response = requests.post('https://619356b8d3ae6d0017da84d0.mockapi.io/balu', data = {'SBP_SVR':finalSBPSVR, 'DBP_SVR':finalDBPSVR})

# ------------------------------------------------------------------------------------------------------------

#DECISION TREE REGRESSOR

# clf = GridSearchCV(DecisionTreeRegressor(), parameters)
# clf1 = GridSearchCV(DecisionTreeRegressor(), parameters)
# clf.fit(x_train1, SBP_train)
# clf1.fit(x_train1, DBP_train)

dtr_SBP = DecisionTreeRegressor(max_depth=10)
print('------SBP------')
print('start fitting...')
dtr_SBP.fit(x_train1, SBP_train)
print('finish fitting...')
dtr_SBP_predict = dtr_SBP.predict(pca_whole_based_vectors)
print('finish predicting...')

dtr_DBP = DecisionTreeRegressor(max_depth=10)
print('------DBP------')
print('start fitting...')
dtr_DBP.fit(x_train1, DBP_train)
print('finish fitting...')
dtr_DBP_predict = dtr_DBP.predict(pca_whole_based_vectors)
print('finish predicting...')

dtr_MAP = DecisionTreeRegressor(max_depth=10)
print('------MAP------')
print('start fitting...')
dtr_MAP.fit(x_train1, MAP_train)
print('finish fitting...')
dtr_MAP_predict = dtr_MAP.predict(pca_whole_based_vectors)
print('finish predicting...')

# print("---------------RESULT---------------")
# print("                 SBP             DBP")
# print("Actual Value:   ", '%.3f       %.3f' %(np.mean(SBP_test),np.mean(DBP_test)))
# print("Predicted Value:",'%.3f       %.3f' %(np.mean(dtr_SBP_predict),np.mean(dtr_DBP_predict)))

print('-----Decision Tree Regression-----')
print("---------------RESULT---------------")
print("                 SBP             DBP")
print("Predicted Value:",'%.3f       %.3f' %(np.mean(dtr_SBP_predict),np.mean(dtr_DBP_predict)))


#ADABOOST

abr_SBP = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=100, learning_rate=0.1, loss='exponential')
print('------SBP------')
print('start fitting...')
abr_SBP.fit(x_train1, SBP_train)
print('finish fitting...')
abr_SBP_predict = abr_SBP.predict(pca_whole_based_vectors)
print('finish predicting...')

abr_DBP = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=100, learning_rate=0.1, loss='exponential')
print('------DBP------')
print('start fitting...')
abr_DBP.fit(x_train1, DBP_train)
print('finish fitting...')
abr_DBP_predict = abr_DBP.predict(pca_whole_based_vectors)
print('finish predicting...')

abr_MAP = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=100, learning_rate=0.1, loss='exponential')
print('------MAP------')
print('start fitting...')
abr_MAP.fit(x_train1, MAP_train)
print('finish fitting...')
abr_MAP_predict = abr_MAP.predict(pca_whole_based_vectors)
print('finish predicting...')

# print("---------------RESULT---------------")
# print("                 SBP             DBP")
# print("Actual Value:   ", '%.3f       %.3f' %(np.mean(SBP_test),np.mean(DBP_test)))
# print("Predicted Value:",'%.3f       %.3f' %(np.mean(abr_SBP_predict),np.mean(abr_DBP_predict)))


print('--------------ADABOOST-------------')
print("---------------RESULT---------------")
print("                 SBP             DBP")
print("Predicted Value:",'%.3f       %.3f' %(np.mean(abr_SBP_predict),np.mean(abr_DBP_predict)))

# ---
# print(np.mean(abr_DBP_predict))
# # 
# print(np.mean(abr_SBP_predict))

# 

import requests
# import json

# json_data = None
# with open('data.json') as json_file:
#     json_data = json.load(json_file)
# # auth=('token', 'example')

finalSBP = '%.3f' %(np.mean(abr_SBP_predict))
finalDBP = '%.3f' %( np.mean(abr_DBP_predict))


response = requests.post('https://619356b8d3ae6d0017da84d0.mockapi.io/balu/1', data = {'id': '1', 'SBP':finalSBP, 'DBP':finalDBP})


print(response)