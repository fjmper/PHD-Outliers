# -*- coding: utf-8 -*-
"""
Created on Tue May 12 21:34:43 2020

@author: Dr. Taimoor adaptado ao problema por Francisco Pereira
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
from lof_tuner import LOF_AutoTuner

# from matplotlib.backends.backend_pdf import PdfPages
import re
# import datetime

def prec_recall(nome_fich, rootdir2, X, y, viz,ncount,Outl):
    g=open("LOF_Predict.txt","a+")
    from sklearn.model_selection import train_test_split
    #
    # Create training and test split
    #
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1, stratify=y)

    from sklearn.preprocessing import StandardScaler
    #from sklearn.svm import SVC
    from sklearn.neighbors import LocalOutlierFactor as LOF

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    import matplotlib.pyplot as plt
    #
    # Standardize the data set
    #
    sc = StandardScaler()
    sc.fit(X_train)

    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    lof = LOF(n_neighbors = viz, novelty=True)#, contamination=0.1)
    lof.fit(X_train, y_train)


    y_pred = lof.predict(X_test)

    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions '+str(viz), fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix:'+nome_fich , fontsize=12)
    plt.savefig("./Predict_images LOF/"+nome_fich+"confusion Matriz")
    # plt.show()


    Prec = precision_score(y_test, y_pred)
    Rec = recall_score(y_test, y_pred)
    Acur = accuracy_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred)
    print('Precision: %.3f' % Prec)
    print('Recall: %.3f' % Rec)
    print('Accuracy: %.3f' % Acur )
    print('F1 Score: %.3f' % F1 )
    text = nome_fich + "," + str(Prec) + "," + str(Rec) + "," +str(Acur)+"," +str(F1)+"," +str(ncount)+"," +str(Outl)+"," +str(viz)+"\n"
    g.write(text)
    g.close()





def Lof3(nome_fich, df,rootdir2):
    f=open("LOF_Values.txt","a+")
    g=open("LOF_Predict.txt","a+")

    df = df[df['Lat'].notna()]
    df = df[df['Long'].notna()]
    df = df[df['Alt_m'].notna()]
    df = df[df['Diftime'].notna()]
    df = df[df['dist_Km'].notna()]
    df = df[df['speed_m'].notna()]

    corpus = df

    ncount=corpus['Lat'].count()
    if ncount > 50:
        data = corpus.iloc[1:, [0, 1, 2, 6, 8, 10]]

        # tuner=LOF_AutoTuner(data)

        tuner = LOF_AutoTuner(data = data, k_max = 50, c_max = 0.1)
        K,C = tuner.run()

        outdata = pd.DataFrame(columns='Lat Long Alt_m Diftime dist_Km speed_m'.split())
        indata = pd.DataFrame(columns='Lat Long Alt_m Diftime dist_Km speed_m'.split())

        from sklearn.neighbors import LocalOutlierFactor

        '''
        Início do plotting para certas condições de Lat e Long
        '''

        lof_fin = LocalOutlierFactor(n_neighbors = K, contamination=C) #, contamination=0.01)


        result = lof_fin.fit_predict(data)

        mineps = np.count_nonzero(result==-1)



        if ((K < ncount) and (K>2)):
            prec_recall(nome_fich, rootdir2, data, result,K,ncount,mineps)
        else:
            text = nome_fich + "," + '0' + "," + '0' + "," +'0'+"," +'0'+"," +'0'+"," +'0'+"," +'0'+"\n"
            g.write(text)
            g.close()
            print("min mair que o número de registos")


        nlinhasout=0
        nlinhasin=0
        for k in range(0,ncount-1):
            if result[k] == -1:
                nlinhasout+=1
                outdata.loc[data.index[k]]=data.iloc[k]
            else:
                nlinhasin+=1
                indata.loc[data.index[k]]=data.iloc[k]


        ind=indata.values
        out = outdata.values

        if len(out) !=0:
            text = nome_fich + "," + str(ncount) + "," + str(len(out)) + "," +str(K)+"\n"
            f.write(text)
            f.close()
            outdata.to_csv("./Resultados LOF/"+nome_fich+'_out_lof_geo.csv', index=False)
            indata.to_csv("./Resultados LOF/"+nome_fich+'_in_lof_geo.csv', index=False)


            nome_fich2 = "./Resultados LOF/"+nome_fich+'_out_lof'
            df = pd.read_csv(nome_fich2+'_geo.csv')
            nome_fich1 = "./Resultados LOF/"+nome_fich+'_in_lof'
            df1 = pd.read_csv(nome_fich1+'_geo.csv')
            df1.head()

            X = np.array(df[['Long',  'Lat']],  dtype='float64')
            X1 = np.array(df1[['Long',  'Lat']],  dtype='float64')

            fig = plt.figure(figsize=(8,6))
            ax1 = fig.add_subplot(111)
            ax1.scatter(X[:, 0],  X[:, 1],  alpha=1,  s=50, c='red', label='out')
            ax1.scatter(X1[:, 0],  X1[:, 1],  alpha=0.2,  s=50, c='blue', label ='in')
            plt.legend(loc='upper left');
            plt.title(nome_fich+"-Map-Long-Lat")
            plt.savefig("./Map_images LOF/"+nome_fich+"-Map-Long-Lat")
            # plt.show()
            plt.close()


            m = folium.Map(location=[df.Lat.mean(),  df.Long.mean()],  zoom_start=9, tiles='Stamen Toner')

            for _,  row in df.iterrows():
                loc = [row.Lat, row.Long]

                folium.CircleMarker(
                    radius=10,
                    location = loc,
                    color='red',
                    fill=True,
                    fill_color='red',
                    fill_opacity = 0.7
                ).add_to(m)
            for _,  row in df1.iterrows():

                loc = [row.Lat, row.Long]

                folium.CircleMarker(
                    radius=5,
                    location = loc,
                    fill=True,
                    # fill_colour='blue'
                    color = '#3186cc',
                    fill_Color = '#3186cc',
                    fill_opacity = 0.7
                ).add_to(m)

            m.save("./Resultados LOF/"+nome_fich+'_geo.html')
        else:
            text = nome_fich + "," + '0' + "," + '0' + "," +'0'+"\n"
            f.write(text)
            f.close()
    else:

        return