import numpy as np
import pandas as pd
import csv
import os
from LocalOutlierFactor_ptm  import Lof3

# Faz importação do ficheiro com os programas a correr

def abrefich(ender1,rootdir2):

    print("ender1= ",ender1)
    try:
        # os.mkdir(ender)
        df = pd.read_csv(ender1, delimiter = ',', index_col=None)
    except OSError:
        df = pd.read_csv(ender1, delimiter=',', index_col=None)


    f=open(rootdir2+"DBS_Values.txt","a+")
    nome_fich= ender1[44:74]
    print(nome_fich)

    Lof3(nome_fich, df,rootdir2)


count=0
countdir=0
# computador secretária
rootdir='D:/Dropbox/Doutoramento 2021/PtmDataset/'

rootdir2 = os.getcwd()
# print('\n')
# rootdir='C:/Users/mpere/Dropbox/Doutoramento 2021/result_data'
# rootdir2='C:/Users/mpere/Dropbox/Doutoramento 2021/Estudo Julho 2021/Estudo geral'


f=open("Lof_Values.txt","a+")
text = " Nome, Num Linhas, Outliers, Minvalues \n"
f.write(text)
f.close()


g=open("Lof_Predict.txt","a+")
text1 = "Nome, Precision, Recall, Accuracy, F-score,Num Linhas, Outliers, Minvalues \n"
g.write(text1)
g.close()

# print(len(os.listdir(rootdir)))
# print(os.listdir(rootdir))
for dirname, _, filenames in os.walk(rootdir):
    countloc=0
    for filename in filenames:
        caminho = os.path.join(dirname+"/", filename)

        abrefich(caminho, rootdir2)
        caminho = ""
        countloc+=1
        count+=1

    countdir+=1
    print("countdir: ",countdir,"count: ",count,"countloc: ",countloc)
print("\n\n diretories= ", countdir, "\t files= ", count)
f.close()
g.close()