import pprint  # For proper print of sequences.
# from nltk.book import *
from SVMs import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import ExtraTreeClassifier,DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score,KFold,train_test_split
from sklearn.svm import  SVC

import pandas as pd
import warnings
#import treetaggerwrapper as ttw
from docx import Document
import nltk
import matplotlib

import glob
import os
import tkinter as tk
from tkinter import filedialog
#import tensorflow as tf
from Ordnilaregression import *
#from tabulate import tabulate
#import ggplot
import sys
from multiclass import *
from more_treshholded import *
import numpy as np

# 1) build a TreeTagger wrapper:
# tagger = ttw.TreeTagger(TAGLANG='es',TAGDIR='/home/alfredo/Python/Treetagger')


# Se quitan los warnnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pandas", lineno=570)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="sklearn")

################################################################
# Declaracion de las funciones
################################################################


nombretexto = '/home/alfredo/Python/Ensayo 1 gilberto.docx'


def getText(filename):
    doc = Document(filename)
    fullText = []
    for para in doc.paragraphs:
        if len(para.text) > 60:
            fullText.append(para.text)

    return "\n".join(fullText).replace(".", ". ").replace("(", "( ").replace(")", ") ")


centroid = []


# print(getText(text))
# j=ttw.make_tags(tags)
# tokens=nltk.word_tokenize(getText(text),'spanish')
# po=nltk.pos_tag(tokens)
# fdist1 = nltk.FreqDist(texto)
# lol=[w for w in V if len(w)>10]
# print("Palabras largas: "+str(lol))
def sigmoid(Z):
    z = np.array(Z)
    out = 1 / (1 + np.exp(-z))
    return out


def extraerymmostrar(mode):
    if mode == 'todos':
        path = '/home/alfredo/Ensayos'
        datos = []
        lol = 0

        for filename in glob.glob(os.path.join(path, '*.docx')):

            texto = getText(filename)
            if lol == 0:
                frases = nltk.sent_tokenize(texto, 'spanish')
                parrafos = texto.split("\n")
                palabras = texto.split()
                lol = 1
            else:
                frases = frases + nltk.sent_tokenize(texto, 'spanish')
                parrafos = parrafos + texto.split("\n")
                palabras = palabras + texto.split()

            print("Documento procesado: " + filename + "\n\n")
            print(texto + "\n\n")
            op = texto.split()
            print(op.__len__())
            V = set(op)
            print("Vocabulario: " + str(V))
            print("Tamano Vocabulario: " + str(V.__len__()))

            largoletras = [len(w) for w in op]
            print("Tamano de cada letra: " + str(largoletras))
            dist = nltk.FreqDist(largoletras)
            print(dist)
            tamanoletramasusada = dist.max()
            print("tamano de letra mas comun: " + str(tamanoletramasusada))
            dislargas = dist.keys()
            promedio = np.mean(np.array(list(dislargas)))
            print("Largo pormedio: " + str(promedio))
            lol = [w for w in V if len(w) > promedio]
            print("Palabras largas: " + str(lol) + "\n\n")

        return frases, parrafos, palabras
    else:
        root = tk.Tk()
        filename = filedialog.askopenfilename(initialdir='/home/alfredo/Ensayos')
        root.destroy()

        texto = getText(filename)
        frases = nltk.sent_tokenize(texto, 'spanish')
        parrafos = texto.split("\n")
        palabras = texto.split()

        print("Documento procesado: " + filename + "\n\n")
        print(texto + "\n\n")
        op = texto.split()
        print(op.__len__())
        V = set(op)
        print("Vocabulario: " + str(V))
        print("Tamano Vocabulario: " + str(V.__len__()))

        largoletras = [len(w) for w in op]
        print("Tamano de cada letra: " + str(largoletras))
        dist = nltk.FreqDist(largoletras)
        print(dist)
        tamanoletramasusada = dist.max()
        print("tamano de letra mas comun: " + str(tamanoletramasusada))
        dislargas = dist.keys()
        promedio = np.mean(np.array(list(dislargas)))
        print("Largo pormedio: " + str(promedio))
        lol = [w for w in V if len(w) > promedio]
        print("Palabras largas: " + str(lol) + "\n\n")

        return frases, parrafos, palabras


# funcion que devuelve la matriz frase palabra
def LSA(frases, k):
    # Hago es una lista solo de las frases

    # Aqui empiezo a hacer el Latent Semantic analisys
    vectorizer = CountVectorizer(min_df=1,stop_words='english')
    dtm = vectorizer.fit_transform(frases)
    # print(str(pd.DataFrame(dtm.toarray(),index=datos,columns=vectorizer.get_feature_names()).head(10)))
    # print("Numero de Frases: ",len(frases))
    # print(vectorizer.get_feature_names())
    # print("Supuesta matriz (diceeen)"+str(dtm))

    # Hago la Singular Value Descomposition
    lsa = TruncatedSVD(k)
    dtm_lsa = lsa.fit_transform(dtm)
    # dtm_lsa=Normalizer(copy=False).fit_transform(dtm_lsa)
    # print(lsa.components_[2])
    # Redusco dimensionalidad para poder graficar

    lol = TSNE(n_components=2, random_state=0)
    ahorasi = lol.fit_transform(dtm_lsa)
    xs = [w[0] for w in ahorasi]
    ys = [w[1] for w in ahorasi]
    return xs, ys, dtm_lsa, ahorasi


def plot(tabla, frases):
    lol = []
    for indx, coso in enumerate(frases):
        lol.append(coso[0:10])
    #    h=np.zeros((len(frases),1),dtype='str')
    #    h[:,0]=np.array(lol,dtype='str')
    #    ko=np.hstack((tabla,h))
    #    coso=pd.DataFrame(data=ko,columns=['Primera componente','Segunda componente','nombre'])



    #   chart=ggplot(coso,aes(x='Primera componente',y='Segunda componente',label='nombre'))\
    #    + geom_text(hjust=0.2) \
    #    + geom_point() \
    #    + ggtitle("T-SNE de las frases de ensayo")

    # chart.show()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xs = [o[0] for o in tabla]
    ys = [o[1] for o in tabla]

    plt.scatter(xs, ys)
    np = []
    for indx, polkiuy in enumerate(lol):
        np.append(indx)
    ki = 0
    for xy in zip(tabla[:, 0], tabla[:, 1]):  # <--
        ax.annotate(lol[ki], xy=xy, textcoords='data')  # <--
        ki += 1
    plt.xlabel('Segunda componente principal')
    plt.ylabel('Primera componente principal')
    plt.title('Espacio Semantico, cada punto es una frase')
    plt.show()
    plt.close()


def make_tag(texto):
    text = nltk.sent_tokenize(texto, 'english')
    #print(text)
    #tagger = ttw.TreeTagger(TAGLANG='en', TAGDIR='/home/alfredo/Python/Treetagger')
    return_tags=0

    #tags = tagger.tag_text(frase)
    return_tags=nltk.pos_tag(nltk.word_tokenize(texto))
    #j = ttw.make_tags(tags)

    #print(str(j) + "\n")
    return return_tags

# Metodo que mide coherencia
def measure_coherence(Components):
    # numero de componentes
    n = len(Components[0, :])
    # numero de frases
    m = len(Components[:, 0])
    components = np.array(Components)
    avecosdis = 0
    aveeudis = 0
    distance_matrixEU = np.zeros((m, m), np.float)
    distance_matrixCO = np.zeros((m, m))
    R = np.zeros(m)
    RCO = np.zeros(m)
    for indx, frase in enumerate(components[:, 0]):
        if indx < m - 1:
            v1 = components[indx, :]
            v2 = components[indx + 1, :]

        avecosdis += np.dot(v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2))
        aveeudis += np.linalg.norm(v1 - v2)
        for indx2, frase2 in enumerate(components[:, 0]):
            if indx == indx2:
                distance_matrixEU[indx, indx2] = float('inf')
                distance_matrixCO[indx, indx2] = float('inf')
                continue
            distance_matrixEU[indx, indx2] = np.linalg.norm(components[indx, :] - components[indx2, :])
            distance_matrixCO[indx, indx2] = (components[indx, :] / np.linalg.norm(components[indx, :])).dot(components[indx2, :] / np.linalg.norm(components[indx2, :]))
        R[indx] = np.min(distance_matrixEU[indx, :])
        RCO[indx] = np.min(distance_matrixCO[indx, :])
    else:
        for indx2, frase2 in enumerate(components[:, 0]):
            if indx == indx2:
                distance_matrixEU[indx, indx2] = float('inf')
                distance_matrixCO[indx, indx2] = float('inf')
                continue
            distance_matrixEU[indx, indx2] = np.linalg.norm(components[indx, :] - components[indx2, :])
            distance_matrixCO[indx, indx2] = (components[indx, :] / np.linalg.norm(components[indx, :])).dot(
                components[indx2, :] / np.linalg.norm(components[indx2, :]))
        R[indx] = np.min(distance_matrixEU[indx, :])
        RCO[indx] = np.min(distance_matrixCO[indx, :])

    min_distanceEU = np.min(distance_matrixEU)
    min_distanceCO=np.min(distance_matrixCO)
    np.fill_diagonal(distance_matrixEU, 0)
    np.fill_diagonal(distance_matrixCO, 0)
    max_distanceEU = np.max(distance_matrixEU)
    max_distanceCO = np.max(distance_matrixCO)

    Det_distance = np.linalg.det(distance_matrixEU)
    np
    Clarck_and_Evans = np.sqrt(m) * (2) * (1 / m) * np.sum(R)
    Clarck_and_EvansCO = np.sqrt(m) * (2) * (1 / m) * np.sum(RCO)

    averangemindistanceEU = np.average(R)
    averangemindistanceCO=np.average(RCO)
    avecosdis = avecosdis / (m - 1)
    aveeudis = aveeudis / (m - 1)
    detCO=np.linalg.det(distance_matrixCO)
    return distance_matrixCO, distance_matrixEU, np.array([Det_distance,detCO ,aveeudis, avecosdis, max_distanceEU, min_distanceEU,max_distanceCO,min_distanceCO,averangemindistanceCO,averangemindistanceEU, Clarck_and_Evans],Clarck_and_EvansCO)


# MIdo spatial data analisys
def spatial_measure(components, distance_matrix_EU,distance_matrix_CO=None):
    n = len(components[0, :])
    m = len(components[:, 0])
    # construyo el centroide
    global centroid
    centroid = np.average(components, axis=0)
    discentroid = np.apply_along_axis(distance_to_centroid, axis=1, arr=components)
    # DISTANCIA PROMEDIO AL CENTORIDE
    avercentroid = np.average(discentroid)
    #####################################
    diferencia_centorid = np.apply_along_axis(diff_to_centroid, axis=1, arr=components)
    diff = np.sum(diferencia_centorid, axis=0)
    # STANDAR DISTANCE
    ################
    S_D = np.sqrt(np.sum(diff) / m)
    ######################
    downR_d = np.argmax(diferencia_centorid)
    # DISTANCIA RELATIVA
    R_D = S_D / diferencia_centorid.reshape((diferencia_centorid.size, 1))[downR_d]
    ###################################

    # SPATIAL AUTOCORRELATION
    # MORAN'S I



    # Getti's G
    aver_distace_betwen_points = np.average(distance_matrix_EU, axis=1)
    new_matrix = np.zeros((m, m))
    # for indx, elem in enumerate(distance_matrix_EU[0,:]):
    #   for indx2,elem2 in enumerate(distance_matrix_EU[0,:]):
    #      new_matrix[indx,indx2]=1 if distance_matrix_EU[indx,indx2]<=aver_distace_betwen_points[indx]else 0
    joderarriba = np.zeros((1, n))
    joderabajo = np.zeros((1, n))
    for i in range(m):
        for j in range(i + 1, m):
            if distance_matrix_EU[i, j] <= aver_distace_betwen_points[i]:
                joderarriba = joderarriba + np.multiply(components[i, :], components[j, :])
            joderabajo = joderabajo + np.multiply(components[i, :], components[j, :])
    new = joderarriba / joderabajo
    G = np.sum(new)

    return np.array([G, avercentroid, S_D, R_D,])


def in_range(distance, d):
    if distance <= d:
        return 1


def diff_to_centroid(point):
    return (point - centroid) ** 2


def distance_to_centroid(point):
    dis = np.linalg.norm(point - centroid)
    return dis


# Metodo que retorna el numero de componentes pruincipales que se debe usar.
def componentesprincipales(matriz):
    U, S, P = np.linalg.svd(matriz, full_matrices=0)
    if len(matriz[0, :]) == 2:
        return 1

    for indx, colum in enumerate(U[0, :]):
        # print("Iteracion: "+str(indx+1))
        if indx == 0:
            continue
        U_reduce = np.array(U[:, 0:indx])
        Z = U_reduce.transpose().dot(matriz)
        X_approx = U_reduce.dot(Z)
        lol = np.linalg.norm((matriz - X_approx), axis=0)
        up = (1 / len(lol)) * np.sum(lol)
        down = (1 / len(lol)) * np.sum(np.linalg.norm(matriz, axis=0))
        number = up / down
        if number < 0.1:
            return indx
    return 2


def myLSA(frases, palabras):
    h = sorted(set(palabras))
    vocabulario = [d for d in h if len(d) >= 4]
    lsa = np.zeros([len(vocabulario), len(frases)])
    for i, palabra in enumerate(vocabulario):
        for j, frase in enumerate(frases):

            if palabra in frase:
                lsa[i, j] = lsa[i, j] + 1
    return lsa


def cost_function(X):
    y = list(X[:, len(X[0, :]) - 1])
    theta = np.zeros((len(np.unique(y)), 1))
    w = np.random.rand(len(X[0, :]) - 1, 1)

    loss = 0
    for indx, element in enumerate(X[:, 0]):
        if indx == 0:
            continue
        z1 = theta[y[indx]] - np.dot(w.T, X[indx, :].T)
        z2 = theta[y[indx - 1]] - np.dot(w.T, X[indx, :].T)
        loss += -np.log(sigmoid(z1) - sigmoid(z2))
    return loss


def otros_ensayos():

    filename = '/media/luis/Data/Universidad/Tesis (Natural language)/Training_Materials/training_set_rel3.xlsx'

    archivo = pd.read_excel(filename,header=0)
    Tabla = archivo

    set = np.array(Tabla['essay_set'])
    index = np.where(set > 1)[0][0]
    coso = Tabla['essay'].iloc[:index]
    y = np.array(Tabla['rater1_domain1'].iloc[:index],dtype=int)
    # Saco todos los ensayos y los pongo en un solo string
    Textos = '\n'
    for ensayo in coso:
        Textos = Textos + ensayo

    return Textos, list(coso), y


def extraer(text):
    frases = nltk.sent_tokenize(text)
    parrafos = text.split("\n")
    palabras = sorted([w for w in nltk.word_tokenize(text) if not w.startswith('@')])
    return palabras, frases, parrafos


def crearX(listaensayos, num_features):
    X = np.zeros((len(listaensayos), num_features))

    for indx,ensayo in enumerate(listaensayos):





        ko =extraer_caract(ensayo)
        if ko==[]:
            continue
        X[indx,:] =ko

    X[:,0]=(X[:,0])/np.std(X[:,0])

    return np.column_stack((np.ones((len(X[:,0]),1)),X))


def extraer_caract(ensayo):
    palabras, frases, parrafos = extraer(ensayo)
    matrix = myLSA(frases, palabras)
    if len(frases)==1:
        return []
    k = componentesprincipales(matrix)
    _, _, comp, _ = LSA(frases, k)
    _ = None
    _, matriz, coso = measure_coherence(comp)
    tags=make_tag(ensayo)
    number_of_different_taggs=len(set([w[1] for w in tags]))
    number_of_words=len(nltk.word_tokenize(ensayo))
    longitud_promedio=np.average([len(w)for w in palabras])
    palabra_mas_larga=len(max(set(palabras)))
    spatial=list(spatial_measure(components=comp, distance_matrix_EU=matriz))

    retu=np.hstack((coso,spatial,number_of_words,number_of_different_taggs,palabra_mas_larga,longitud_promedio))


    return retu


#def sofisticacion_lexica(en,):

# Prueba en un set
def probar_Test_set(X,y,logisticclass):

    porcentaje_de_error = 0

    up=0
    distancia_promedio=[]
    predict=logisticclass.predict(X)

    for j, real in enumerate(y):

        if predict[j]!=real:
            up=up+1
            distancia_promedio.append(predict[j]-real)

    errores = np.unique(distancia_promedio)
    acum= np.zeros((1,len(errores)))

    for i,er in enumerate(distancia_promedio):

        for j, cos in enumerate(errores):

            if distancia_promedio[i]==cos:
                acum.T[j]=acum.T[j]+1



    distancia_promedio=np.average(np.array(distancia_promedio))

    porcentaje_de_error = up / len(y)
    return porcentaje_de_error,distancia_promedio,acum,errores



#
# def crearX_prueba(k):
#     filename = '/media/alfredo/Data/Universidad/Tesis (Natural language)/Training_Materials/training_set_rel3.xlsx'
#     archivo = pd.ExcelFile(filename)
#     Tabla = archivo.parse(sheetname='training_set')
#
#     set = np.array(Tabla['essay_set'])
#     index = np.where(set > 1)[0][0]
#     coso = np.array(Tabla['essay'].iloc[index - 800:index])
#     y = np.array(Tabla['domain1_score'].iloc[index - 800:index])
#     respuesta_modelo = np.zeros((len(y), 1))
#     X = crearX(coso, k)
#
#     return X,y
#
#





def Cross_val(X,y, logisticclass):
    kf = KFold(n_splits=7, random_state=13)
    porcentaje=[]
    acums=[]
    errores=[]
    for train_index, test_index in kf.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        logisticclass.fit(X_train,y_train)
        por,_,acum,err=probar_Test_set(X_test,y_test,logisticclass)
        porcentaje.append(por)
        acums.append(acum)
        errores.append(err)
    return porcentaje,acums,errores

def weigthed_Kappa(y_real,y_predicho,min, max):
    N=len(y_real)
    S=range(min,max+1,1)
    O=np.zeros((len(S),len(S)))
    w=np.zeros((len(S),len(S)))
    HA=0
    HB=0
    for  i,rankA in enumerate(S):
        for j, rankB in enumerate(S):
            temp1=np.where(y_real==rankA)[0]
            temp2=np.where(y_predicho==rankB)[0]
            O[i,j]=len((set(temp2)& set(temp1)))
            w[i,j]=(((i+1)-(j+1))**2)/((len(S)-1)**2)

    #HA=np.sum(O,1)
    #HB=np.sum(O,0)
    E=np.zeros((len(S),len(S)))
    for i, numberA in enumerate(O[:,0]):
        for j, numberB in enumerate(O[0,:]):
            E[i,j]=(np.sum(O[i,:])*np.sum(O[:,j]))/np.sum(O)
    up=0
    down=0
    for i, nada in enumerate(S):
        for j , nada2 in enumerate(S):
            up+=w[i,j]*O[i,j]
            down+=w[i,j]*E[i,j]



    loss=up/down

    k= 1-loss
    return k
def change_y(y,clase):
    lol=np.zeros((len(y),1))

    for i, cos in enumerate(y):
        if cos==clase :
            lol[i]=1

    return lol

class clasificador():

    def __init__(self, min, max):
        self.clases = range(min,max+1,1)

        # for elem in self.clases:
        #     if  elem==5 or elem==4 or elem==12 or elem==2:
        #         self.clasificadores.append(SVC(C=2,class_weight={1:'10'},probability=True))
        #         continue
        #     # if elem==3:
        #     #     self.clasificadores.append(SVC(C=2,class_weight={1:'20'},probability=True))
        #     #     continue
        #     self.clasificadores.append(SVC(C=2,probability=True))



    def fit(self,X_train,y_train,C_i):
        pesos_clases={}

        for i,clase in enumerate(self.clases):
            cuantos=len(np.where(y_train==clase)[0])
            if cuantos==0:
                continue
            if cuantos<=50:
                pesos_clases[int(clase)]=30

        self.options=pesos_clases
        self.clasificador=SVC(C=C_i,class_weight=self.options)
        self.clasificador.fit(X_train,y_train)

    def predict(self,X_test):
        #y_out=np.zeros((len(X_test[:,0]),1))

        y_out=self.clasificador.predict(X_test)

        return y_out













##############################################################################################
# Ejecucion de las cosas

#############################################################################################
#a = np.array(np.mat('1 0 1 0 0 0;0 1 0 0 0 0;1 1 0 0 0 0;1 0 0 1 1 0;0 0 0 1 0 1'))
textos, textos_sep, y1 = otros_ensayos()
el_respectivo=OrdinalSM(50)
#X_train,X_test,y_train,y_test=train_test_split(textos_sep,y1,train_size=0.8)





# frases,parrafos,palabras = extraerymmostrar(mode='todos')
# xs,ys,matrix_reduce,matrix=LSA(frases)
X = crearX(textos_sep,num_features=19)
#
# mr_plot=TSNE(n_components=2,random_state=np.random.rand(1))
# ahorasi=mr_plot.fit_transform(X)
# xs = [w[0] for w in ahorasi]
# ys = [w[1] for w in ahorasi]
# colors=y1
#
#
# data=pd.DataFrame(data=np.column_stack((np.array(xs),np.array(ys),y1)),columns=['x','y','clase'])
# for color in np.unique(y1):
#     index=np.where(y1==color)[0]
#     # if color==6:
#     #     s=np.random.rand(1)
#     #     temp=np.ones((len(y1[index]),1))*s
#     #     colors=temp
#     #
#     # else:
#     colors=y1[index]
#     temp1=np.array(xs)[index]
#     temp2=np.array(ys)[index]
#     print(color)
#
#     plt.scatter(list(temp1), list(temp2),c=list(color),labels =list(color))
# plt.legend()
# #plt.scatter(xs, ys,c=colors)
# plt.xlabel('Segunda componente principal')
# plt.ylabel('Primera componente principal')
# plt.title('Todo el conjunto de Datos')
# plt.show()
#
#
X_train,X_test,y_train,y_test=train_test_split(X,y1,train_size=0.8)
el_respectivo.fit(X_train,y_train)
el_respectivo.crearTheta()



#
#
#
# #w,b=PRank(X,y)
# #X_test,y_test=crearX_prueba(11)
# #porcentaje=test_PRank(w,b,X_test,y_test)
# #print("MI propio algoritmo con juegos de azar y mujerzuelas: %f :D"%porcentaje)
# # U,S,P=np.linalg.svd(a,full_matrices=1)
# # U2,S2,P2=np.linalg.svd(a,full_matrices=0)
#
#
# #W, theta = ordinal_logistic_fit(X, y, alpha=0)
#
# # ####################Pruebo la regresion logistica############
# print("Regresion  Ordinal Logistica")
# z=LogisticIT(alpha=0.1, verbose=0)
# #
# #
# z.fit(X_train,y_train)
# y_predicho=z.predict(X_test)
# k=weigthed_Kappa(y_test,y_predicho,1,6)
# #
# #
# print("El indicador kappa: %0.3f"%k)
# porcentaje,_,_,_=probar_Test_set(X_test,y_test,z)
# print("Porcentajes: "+str(porcentaje)+"\n")
# # ### Pruebo extremely ramdomized trees######
# print("Extremely randomized forest")
# arbol=ExtraTreeClassifier(random_state=7)
# arbol.fit(X_train,y_train)
# y_predicho=arbol.predict(X_test)
# k=weigthed_Kappa(y_test,y_predicho,1,6)
# print("Indicador appa: %0.5f"%k)
# porcentaje,_,_,_=probar_Test_set(X_test,y_test,arbol)
# print("Porcentajes: "+str(porcentaje)+"\n")
#
# print("Ahora mi propio clasificador")
#
# C_de_prueba=range(5,30,5)
#
#
# el_que_es=clasificador(1,6)
#
#
#
# indx=np.where(y_train==3)[0]
# if indx==[]:
#     indx=np.where(y_test==3)[0]
#     temp=X_train[-1,:]
#     temp2=X_test[-1,:]
#     y_temp=y_train[-1]
#     y_temp1=y_test[indx]
#     X_train[-1, :]=temp2
#     X_test[-1,:]=temp
#     y_train[-1]=y_temp1
#     y_test[indx]=y_temp
# for c in C_de_prueba:
#     print("Con C : %i"%c)
#     el_que_es.fit(X_train,y_train,c)
#     y_out=el_que_es.predict(X_test)
#     k=weigthed_Kappa(y_test,y_out,1,6)
#     print("El indicador kappa: %0.3f"%k)
#     porcentaje,_,_,_=probar_Test_set(X_test,y_test,el_que_es)
#     print("Porcentajes: "+str(porcentaje)+"\n")
#
#
#
#
#
#
#
#
# ### Pruebo clasificacion por porcesos gaussianos (one vs all)  ######
#
#
#
# #diferentes gaussianos
#
# gauss= GaussianProcessClassifier(n_restarts_optimizer=0)
# gauss.fit(X_train,y_train)
# y_predicho = gauss.predict(X_test)
# k = weigthed_Kappa(y_test,y_predicho,1,6)
# print("#######Proceso Gaussiano con  %i reinicios ##############\n"%1)
# print("Indicador appa: %0.5f" % k)
# porcentaje, _, _, _ = probar_Test_set(X_test, y_test, gauss)
# print("Porcentajes " + str(porcentaje) + "\n")
#
#
#
#
#
#
#
#
#
#
#
#








##filename=filedialogaskopenfilename(initialdir = '/home/alfredo/Ensayos')
# root.destroy()
# testo=getText(filename)
# palabrasp,frasesp,parrafosp=extraer(testo)

# matrixp=myLSA(frasesp,palabrasp)
# kp=componentesprincipales(matrixp)
# print("Componentes usadas:"+str(kp))
# xs,ys,componentsp,mirar=LSA(frasesp,kp)


#porcentaje,aver,acum,errores = probar_Test_set(z)
#print("LogisticIT: "+str(porcentaje))


#porcentajes, acumulados,err=Cross_val(X,y_train,z)


#print("Acumulados: "+str(acumulados)+"\n")
#print("Errores: "+str(err)+"\n")



# RESPUESTA=ordinal_logistic_predict(W,theta,measure_coherence(Components=componentsp))
# print(RESPUESTA)

# euclidiana,cosenoidal,min_distance,max_distance,averminn,Clarck_Evans=measure_coherence(components)
# print("Distancia euclidiana entre frases adyacentes: "+str(euclidiana)+" Distancia promedio cosenoidal entre frases adyacentes: "+str(cosenoidal))
# spatial_measure(components)
# plot(mirar,frases)
