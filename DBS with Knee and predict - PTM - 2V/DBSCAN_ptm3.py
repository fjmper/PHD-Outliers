# -*- coding: utf-8 -*-
"""
Created on Tue May 12 21:40:26 2020

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
from sklearn.neighbors import NearestNeighbors
from ipywidgets import interactive
from collections import defaultdict
from kneed import KneeLocator, DataGenerator as dg
import folium
import re

def prec_recall(nome_fich, rootdir2, X, y, viz,eps,outl,ncount):
    g=open("DBS_Predict.txt","a+")
    print("entrou prec")
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1, stratify=y)

    from sklearn.preprocessing import StandardScaler

    from sklearn.cluster import DBSCAN

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    import matplotlib.pyplot as plt

    sc = StandardScaler()
    sc.fit(X_train)

    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)


    dbs = DBSCAN(eps = eps, min_samples = viz)
    dbs.fit(X_train, y_train)

    y_pred = dbs.fit_predict(X_test)

    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions '+str(viz), fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix:'+nome_fich , fontsize=12)
    plt.savefig("./Predict_images DBS/"+nome_fich+"confusion Matriz")
    plt.show()


    Prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    Rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    Acur = accuracy_score(y_test, y_pred)
    F1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print('Precision: %.3f' % Prec)
    print('Recall: %.3f' % Rec)
    print('Accuracy: %.3f' % Acur )
    print('F1 Score: %.3f' % F1 )
    text = nome_fich + "," + str(Prec) + "," + str(Rec) + "," +str(Acur)+"," +str(F1)+","+str(ncount)+"," +str(outl)+","+str(eps)+","+str(viz)+"\n"
    g.write(text)


def calc_eps_knee(nome_fich, df,rootdir2,ncount):
    h=open("Knee_Values.txt","a+")

    data=df

    neighbors = NearestNeighbors(n_neighbors=7)
    neighbors_fit = neighbors.fit(data)
    distances, indices = neighbors_fit.kneighbors(data)

    distances = np.sort(distances[:,3])

    x=range(1,ncount+1)
    y= distances

    plt.style.use("ggplot")
    plt.figure(figsize=(8, 6))
    kl = KneeLocator(x, y, curve="convex", S=1)

    Knee=round(kl.knee_y,2)

    percent = (kl.knee/ncount)
    if percent >= 0.11 and percent<0.95:
        percent-=0.1
    elif percent > 0.95:
        percent = 0.95


    st = np.quantile(distances, percent)

    start = np.argmax(distances >= st)
    K = "Knee = "+ str(Knee)
    plt.style.use("ggplot")
    plt.figure(figsize=(8, 6))
    plt.title("Fich: " + nome_fich)
    plt.xlim(start,ncount)
    if (kl.knee_y>90):
        z= kl.knee_y +10
        plt.ylim(0,z)
    else:
        plt.ylim(0,100)
    plt.plot(x, distances)
    plt.vlines(kl.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    plt.hlines(kl.knee_y, plt.xlim()[0], plt.xlim()[1], linestyles='dashed')
    plt.xlabel(K,  fontsize=18)

    plt.savefig("./Map_images Knee/"+nome_fich)

    plt.show()
    plt.close()

    text = nome_fich +" , "+ str(ncount)+ " , "+ str(kl.knee)+" , " + str(Knee) + "\n"
    h.write(text)
    h.close()
    return(Knee)


def Dbs3(nome_fich, df,rootdir2):
    f=open("DBS_Values.txt","a+")
    m=open("DBS_clusters.txt","a+")
    plt.rcParams.update({'figure.max_open_warning': 0})




    df = df[df['latitude']!=0]
    df = df[df['longitude']!=0]
    data = df.iloc[:, [6, 7]]


    ncount = data['latitude'].count()





    outdata = pd.DataFrame(columns='latitude longitude'.split())
    indata = pd.DataFrame(columns='latitude longitude'.split())

    from sklearn.cluster import DBSCAN

    mindbeps=0
    mindbmSamples =4

    epsilon =0.005

    dbscan = DBSCAN(eps = epsilon, min_samples = 4)

    result = dbscan.fit_predict(data)

    unique, counts = np.unique(result, return_counts=True)
    noise=0



    for j in range(1, len(result)):
        if result[j]==-1:
            noise+=1

    mineps = np.count_nonzero(result==-1)

    text = nome_fich+"," +str(ncount)+"," + str(mineps) + "," + str(epsilon)+"," +str(len(counts))+ "\n"
    m.write(text)
    m.close()


    if mineps > 0:

        text = nome_fich + "," + str(ncount) + "," + str(mineps) + "," +str(mindbeps)+ "," +str(mindbmSamples)+"\n"
        f.write(text)


        nlinhasout=0
        nlinhasin=0


        for k in range(0, ncount-1):
            if result[k] == -1:
                nlinhasout+=1
                outdata.loc[data.index[k]]=data.iloc[k]
            else:
                nlinhasin+=1
                indata.loc[data.index[k]]=data.iloc[k]



        ind = indata.values
        out = outdata.values

        outdata.to_csv("./Resultados DBS/"+nome_fich+'out_dbs_ptm.csv', index=False)
        indata.to_csv("./Resultados DBS/"+nome_fich+'in_dbs_ptm.csv', index=False)



        nome_fich2= "./Resultados DBS/"+nome_fich+'out_dbs'
        df = pd.read_csv(nome_fich2+'_ptm.csv')
        nome_fich1= "./Resultados DBS/"+nome_fich+'in_dbs'
        df1 = pd.read_csv(nome_fich1+'_ptm.csv')
        df1.head()




        X = np.array(df[['longitude',  'latitude']],  dtype='float64')
        X1 = np.array(df1[['longitude',  'latitude']],  dtype='float64')



        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.scatter(X[:, 0],  X[:, 1],  alpha=1,  s=50, c='red', label='out')
        ax1.scatter(X1[:, 0],  X1[:, 1],  alpha=0.2,  s=50, c='blue', label ='in')
        plt.legend(loc='upper left')
        plt.title(nome_fich+"-Map-Long-Lat")
        plt.savefig("./Map_images DBS/"+nome_fich+"-Map-Long-Lat")
        plt.show()
        plt.close()




        m = folium.Map(location=[df1.latitude.mean(),  df1.longitude.mean()],  zoom_start=14, tiles='Stamen Toner')


        for _,  row in df.iterrows():

            loc = [row.latitude, row.longitude]

            folium.CircleMarker(
                radius=10, location = loc,
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
                fill=True,
                color = '#3186cc',
                fill_Color = '#3186cc',
                fill_opacity = 0.7
            ).add_to(m)

        m.save("./Map_images DBS/"+nome_fich+'_ptm.html')

        if ((mineps < ncount) and (mineps>2)):

            prec_recall(nome_fich, rootdir2, data, result, mindbmSamples,epsilon,mineps,ncount)
        else:
            print("min maior que o número de registos")



    else:

        return