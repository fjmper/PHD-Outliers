import pandas as pd
import numpy as np
#from sklearn import datasets
#
# Load the breast cancer data set
#
# bc = datasets.load_breast_cancer()
# X = bc.data
# y = bc.target

def prec_recall(nome_fich, X, y, viz,ncount,Outl):
    g=open("LOF_Predict.txt","a+")
    # print(bc)
    # print("\n")
    # print(X)
    # print("\n")
    # print(y)
    # print("\n")
    # print(viz)
    # print("\n")
    # print(nome_fich)
    # print("\n")
    # print(rootdir2)
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
    #print(sc)
    #input("ca dentro 1")
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    #
    # Fit the SVC model
    #
    # svc = SVC(kernel='linear', C=10.0, random_state=1)
    # svc.fit(X_train, y_train)
    #lof = LOF(kernel='linear', C=10.0, random_state=1)
    lof = LOF(n_neighbors = viz, novelty=True)#, contamination=0.1)
    lof.fit(X_train, y_train)

    #input("ca dentro 2")
    #
    # Get the predictions
    #
    y_pred = lof.predict(X_test)
    #
    # Calculate the confusion matrix
    #
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    #
    # Print the confusion matrix using Matplotlib
    #
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
    plt.close()

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