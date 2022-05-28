# -*- coding: utf-8 -*-
"""
Created on Tue May 12 21:34:43 2020

@author: Dr. Taimoor
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from tabulate import tabulate
from tqdm import tqdm
from sklearn.cluster import KMeans,  DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from ipywidgets import interactive
from collections import defaultdict
import folium
from prec_recall_test_LOF import prec_recall
from lof_tuner import LOF_AutoTuner
import re

def Lof3(nome_fich, df,rootdir2):
    f=open("LOF_Values.txt","a+")
    g=open("LOF_Predict.txt","a+")

    # nome_fich='07-07-2017-ap-d29dabb1706d-pml'
    # corpus = pd.read_csv(nome_fich+'.csv')
    # corpus.drop(['Date', 'hour','SP-POI','Week_day','user','transp'], axis=1, inplace=True)
    # pd.options.display.max_columns = None
    # pd.options.display.width=None


    # df=corpus

    df = df[df['latitude']!=0]
    df = df[df['longitude']!=0]
    data = df.iloc[:, [6, 7]]


    ncount=data['latitude'].count()
    outdata = pd.DataFrame(columns='latitude longitude'.split())
    indata = pd.DataFrame(columns='latitude longitude'.split())



    from sklearn.neighbors import LocalOutlierFactor

    '''
    Início do plotting para certas condições de Lat e Long
    '''

    # print("min")
    # print(min_nn)
    # print("max")
    # print(max_nn)

    tuner = LOF_AutoTuner(data = data, k_max = 50, c_max = 0.1)
    K,C = tuner.run()


    lof_fin = LocalOutlierFactor(n_neighbors = K, contamination=C)

    result = lof_fin.fit_predict(data)

    print("K e C = ",K, C)
    nlinhas=0
    nlinhasout=0
    nlinhasin=0
    # out = pd.DataFrame()
    for k in range(0,ncount-1):
        if result[k] == -1:
            nlinhasout+=1
            outdata.loc[data.index[k]]=data.iloc[k]
        else:
            nlinhasin+=1
            indata.loc[data.index[k]]=data.iloc[k]


    ind=indata.values
    out = outdata.values
    otl = len(out)
    print(otl)
    prec_recall(nome_fich, data, result,K,ncount,otl)

    text = nome_fich + "," + str(ncount) + "," + str(len(out)) + "," +str(K)+"\n"
    f.write(text)
    f.close()

    outdata.to_csv("./Resultados LOF/"+nome_fich+'_out_lof_ptm.csv', index=False)
    indata.to_csv("./Resultados LOF/"+nome_fich+'_in_lof_ptm.csv', index=False)



    nome_fich2 = "./Resultados LOF/"+nome_fich+'_out_lof'
    df = pd.read_csv(nome_fich2+'_ptm.csv')
    nome_fich1 = "./Resultados LOF/"+nome_fich+'_in_lof'
    df1 = pd.read_csv(nome_fich1+'_ptm.csv')
    df1.head()


    X = np.array(df[['longitude',  'latitude']],  dtype='float64')
    X1 = np.array(df1[['longitude',  'latitude']],  dtype='float64')
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(X[:, 0],  X[:, 1],  alpha=1,  s=50, c='red', label='out')
    ax1.scatter(X1[:, 0],  X1[:, 1],  alpha=0.2,  s=50, c='blue', label ='in')
    plt.legend(loc='upper left');
    plt.title("Long-Lat")
    plt.savefig("./Map_images LOF/"+nome_fich+"-Map-Long-Lat")

    # plt.show()
    plt.close()


    m = folium.Map(location=[df1.latitude.mean(),  df1.longitude.mean()],  zoom_start=14, tiles='Stamen Toner')

    # if escolha == '1':
    #     loc = [row.Lat, row.Long]
    # else:
    #     loc = [row.latitude, row.longitude]
    for _,  row in df.iterrows():

        loc = [row.latitude, row.longitude]
        folium.CircleMarker(
            radius=10,
            location = loc,

            # popup=re.sub(r'[^a-zA-Z ]+',  '',  row.str(Alt_m)),
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity = 0.7
        ).add_to(m)
    for _,  row in df1.iterrows():

        loc = [row.latitude, row.longitude]
        folium.CircleMarker(
            radius=5,
            location = loc,
            # popup=re.sub(r'[^a-zA-Z ]+',  '',  row.str(Alt_m)),
            #color='blue',
            fill=True,
            # fill_colour='blue'
            color = '#3186cc',
            fill_Color = '#3186cc',
            fill_opacity = 0.7
        ).add_to(m)


    m.save("./Map_images LOF/"+nome_fich+'_ptm.html')
    input("ttyttt")