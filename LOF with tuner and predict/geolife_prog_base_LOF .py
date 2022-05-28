import numpy as np
import pandas as pd
import csv
import os
from Lof_geo3 import Lof3


# Faz importação do ficheiro com os programas a correr

def abrefich(ender1,rootdir2,ender,file):

    print("ender1= ",ender1)
    try:
        os.mkdir(ender)
        df = pd.read_csv(ender1, delimiter = ',', index_col=None)
    except OSError:
        df = pd.read_csv(ender1, delimiter=',', index_col=None)

    # print(ender1[-23:],"\n")
    f=open(rootdir2+"LOF_Values.txt","a+")
    nome_fich= ender1[-22:-19]+"-"+ender1[-18:-4]
    print(nome_fich)
    # f.write(nome_fich)
    # corre programa para correr ficheiros
    Lof3(nome_fich, df,rootdir2)
    # input("Veja o nome, Próximo nome")
    # df.to_excel(ender+file,index=False)

count=0
countdir=0
rootdir='D:/Dropbox/Doutoramento 2021/Result_data/'
rootdir2 = os.getcwd()
f=open("LOF_Values.txt","a+")
text = "Nome, Num Registos, Outliers, Neigbours \n"
f.write(text)
f.close()
g=open("LOF_Predict.txt","a+")
text1 = "Nome, Precision, Recall, Accuracy, F-score, Num linhas, Outliers, Neigbours \n"
g.write(text1)
g.close()

print(len(os.listdir(rootdir)))
print(os.listdir(rootdir))
for dirname, _, filenames in os.walk(rootdir):
    countloc=0
    for filename in filenames:
        caminho = os.path.join(dirname+"/", filename)
        caminho2 = dirname[:-16]+"/result_xls/"+dirname[-3:]
        file_name="/"+filename[:-3]+'xlsx'
        abrefich(caminho, rootdir2,caminho2,file_name)
        caminho = ""
        countloc+=1
        count+=1

    countdir+=1
    print("countdir: ",countdir,"count: ",count,"countloc: ",countloc)
print("\n\n diretories= ", countdir, "\t files= ", count)
f.close()
g.close()