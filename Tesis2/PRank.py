import numpy as np
import pandas as pd
import nltk as nl
from sklearn import *

#Hago el algoritmo PRank descrito en "Pranking With Ranking" de Koby Crammer y Yoran Singer

def preproces(X):
    aver=np.average(X,axis=0)
    std=np.std(X,axis=0)
    for i,feature in enumerate(X[:,0]):
        X[i,:]=(X[i,:]-aver)/std

    return X


def PRank(x,y):
    #number features
    n=len(x[0,:])
    #number of examples
    T=len(x[:,0])
    i=0
    #x=preproces(x)
    w0=np.zeros(n)
    ranks=np.unique(y-2)
    b=np.zeros((len(ranks)-1,1))
    w=w0
    bk=float('inf')
    y=y-2
    while(i<T):

        xt=x[i,:]
        yt = y[i]
        prueba=np.array([False,False,True,True])

        lol=np.argmax(prueba)
        y_gorro=ranks[np.argmax(w.dot(xt)-b<0)]
        if (all(w)==0and all(b)==0)or all((w.dot(xt)-b<0)==False):
            y_gorro=np.max(ranks)


        if y_gorro!=yt:
            print("prediccion: "+str(y_gorro)+" real: "+str(yt))
            y_new=np.ones((len(ranks)-1,1))
            tao=np.zeros((len(ranks)-1,1))
            for j, br in enumerate(b):
                if yt<=ranks[j]:
                    y_new[j]=-1
                if(xt.dot(w)-br)*y_new[j]<=0:
                    tao[j]=y_new[j]
            w=w+np.sum(tao)*xt
            b=b-tao
        i=i+1
    return w,b

def PRank_predict(w,b,x,y):
    comp=w.dot(x)
    r=np.unique(y)
    for i,br in enumerate(b):
        if(comp-br)<0:
            return i+np.min(r)
    return r.max()


def test_PRank(w,b,X,y):
    count=0
    for i, _ in enumerate(X[:,0]):
        if PRank_predict(w,b,X[i,:],y)!=y[i]:
            count+=1
    return count/len(y)


