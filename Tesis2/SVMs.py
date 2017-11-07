# Substring kernel

import time as t
import numpy as np
from modshogun import StringCharFeatures,RAWBYTE
from shogun.Kernel import StringSubsequenceKernel
from sklearn import svm
from sklearn.datasets import fetch_20newsgroups
from sklearn.svm import libsvm
import sys
from time import time
from functools import wraps
from scipy import optimize
from modshogun import StringCharFeatures,RAWBYTE
def caching():
    """
    Cache decorator. Arguments to the cached function must be hashable.
    """
    def decorate_func(func):
        cache = dict()
        # separating positional and keyword args
        kwarg_point = object()

        @wraps(func)
        def cache_value(*args, **kwargs):
            key = args
            if kwargs:
                key += (kwarg_point,) + tuple(sorted(kwargs.items()))
            if key in cache:
                result = cache[key]
            else:
                result = func(*args, **kwargs)
                cache[key] = result
            return result

        def cache_clear():
            """
            Clear the cache
            """
            cache.clear()

        # Clear the cache
        cache_value.cache_clear = cache_clear
        return cache_value
    return decorate_func


class StringKernel():
    """
    Implementation of string kernel from article:
    H. Lodhi, C. Saunders, J. Shawe-Taylor, N. Cristianini, and C. Watkins.
    Text classification using string kernels. Journal of Machine Learning Research, 2, 2002 .
    svm.SVC is a basic class from scikit-learn for SVM classification (in multiclass case, it uses one-vs-one approach)
    """
    def __init__(self, subseq_length=3, lambda_decay=0.5):
        """
        Constructor
        :param lambda_decay: lambda parameter for the algorithm
        :type  lambda_decay: float
        :param subseq_length: maximal subsequence length
        :type subseq_length: int
        """
        self.lambda_decay = lambda_decay
        self.subseq_length = subseq_length
       # svm.SVC.__init__(self, kernel='precomputed')


    @caching()
    def _K(self, n,s, t): #modifique aqui, quite el n que tocaba mandar
        """
        K_n(s,t) in the original article; recursive function
        :param n: length of subsequence
        :type n: int
        :param s: document #1
        :type s: str
        :param t: document #2
        :type t: str
        :return: float value for similarity between s and t
        """

        #n=self.subseq_length
        if min(len(s), len(t)) < n:
            return 0
        else:
            part_sum = 0
            for j in range(1, len(t)):
                if t[j] == s[-1]:
                    #not t[:j-1] as in the article but t[:j] because of Python slicing rules!!!
                    part_sum += self._K1(n - 1, s[:-1], t[:j])
            result = self._K(n, s[:-1], t) + self.lambda_decay ** 2 * part_sum
            return result


    @caching()
    def _K1(self, n, s, t):
        """
        K'_n(s,t) in the original article; auxiliary intermediate function; recursive function
        :param n: length of subsequence
        :type n: int
        :param s: document #1
        :type s: str
        :param t: document #2
        :type t: str
        :return: intermediate float value
        """
        if n == 0:
            return 1
        elif min(len(s), len(t)) < n:
            return 0
        else:
            part_sum = 0
            for j in range(1, len(t)):
                if t[j] == s[-1]:
        #not t[:j-1] as in the article but t[:j] because of Python slicing rules!!!
                    part_sum += self._K1(n - 1, s[:-1], t[:j]) * (self.lambda_decay ** (len(t) - (j + 1) + 2))
            result = self.lambda_decay * self._K1(n, s[:-1], t) + part_sum
            return result


    def _gram_matrix_element(self, s, t, sdkvalue1, sdkvalue2):
        """
        Helper function
        :param s: document #1
        :type s: str
        :param t: document #2
        :type t: str
        :param sdkvalue1: K(s,s) from the article
        :type sdkvalue1: float
        :param sdkvalue2: K(t,t) from the article
        :type sdkvalue2: float
        :return: value for the (i, j) element from Gram matrix
        """
        if s == t:
            return 1
        else:
            try:
                return self._K(self.subseq_length, s, t) / \
                       (sdkvalue1 * sdkvalue2) ** 0.5
            except ZeroDivisionError:
                print("Maximal subsequence length is less or equal to documents' minimal length."
                      "You should decrease it")
                sys.exit(2)


    def string_kernel(self, X1, X2):
        """
        String Kernel computation
        :param X1: list of documents (m rows, 1 column); each row is a single document (string)
        :type X1: list
        :param X2: list of documents (m rows, 1 column); each row is a single document (string)
        :type X2: list
        :return: Gram matrix for the given parameters
        """
        len_X1 = len(X1)
        len_X2 = len(X2)
        # numpy array of Gram matrix
        gram_matrix = np.zeros((len_X1, len_X2), dtype=np.float32)
        sim_docs_kernel_value = {}
        #when lists of documents are identical
        if X1 == X2:
        #store K(s,s) values in dictionary to avoid recalculations
            for i in range(len_X1):
                sim_docs_kernel_value[i] = self._K(self.subseq_length, X1[i], X1[i])
        #calculate Gram matrix
            for i in range(len_X1):
                for j in range(i, len_X2):
                    gram_matrix[i, j] = self._gram_matrix_element(X1[i], X2[j], sim_docs_kernel_value[i],
                                                                 sim_docs_kernel_value[j])

        #using symmetry
                    gram_matrix[j, i] = gram_matrix[i, j]
                print("Fila: %i" % i)
        #when lists of documents are not identical but of the same length
        elif len_X1 == len_X2:
            sim_docs_kernel_value[1] = {}
            sim_docs_kernel_value[2] = {}
        #store K(s,s) values in dictionary to avoid recalculations
            for i in range(len_X1):
                sim_docs_kernel_value[1][i] = self._K(self.subseq_length, X1[i], X1[i])
            for i in range(len_X2):
                sim_docs_kernel_value[2][i] = self._K(self.subseq_length, X2[i], X2[i])
        #calculate Gram matrix
            for i in range(len_X1):
                for j in range(i, len_X2):
                    gram_matrix[i, j] = self._gram_matrix_element(X1[i], X2[j], sim_docs_kernel_value[1][i],
                                                                 sim_docs_kernel_value[2][j])
        #using symmetry
                    gram_matrix[j, i] = gram_matrix[i, j]
        #when lists of documents are neither identical nor of the same length
        else:
            sim_docs_kernel_value[1] = {}
            sim_docs_kernel_value[2] = {}
            min_dimens = min(len_X1, len_X2)
        #store K(s,s) values in dictionary to avoid recalculations
            for i in range(len_X1):
                sim_docs_kernel_value[1][i] = self._K(self.subseq_length, X1[i], X1[i])
            for i in range(len_X2):
                sim_docs_kernel_value[2][i] = self._K(self.subseq_length, X2[i], X2[i])
        #calculate Gram matrix for square part of rectangle matrix
            for i in range(min_dimens):
                for j in range(i, min_dimens):
                    gram_matrix[i, j] = self._gram_matrix_element(X1[i], X2[j], sim_docs_kernel_value[1][i],
                                                                 sim_docs_kernel_value[2][j])
                    #using symmetry
                    gram_matrix[j, i] = gram_matrix[i, j]

        #if more rows than columns
            if len_X1 > len_X2:
                for i in range(min_dimens, len_X1):
                    for j in range(len_X2):
                        gram_matrix[i, j] = self._gram_matrix_element(X1[i], X2[j], sim_docs_kernel_value[1][i],
                                                                     sim_docs_kernel_value[2][j])
        #if more columns than rows
            else:
                for i in range(len_X1):
                    for j in range(min_dimens, len_X2):
                        gram_matrix[i, j] = self._gram_matrix_element(X1[i], X2[j], sim_docs_kernel_value[1][i],
                                                                     sim_docs_kernel_value[2][j])
        print(sim_docs_kernel_value)
        return gram_matrix


    #



class OrdinalSM():


    def __init__(self,c,e=1e-3):

        self.C=c
        self.sigma=10
        self.visitados=[]
        self.e=e


    def kernel(self,x1,x2):

        return np.exp(-(np.linalg.norm(x1-x2)**2)/(2*self.sigma**2))



    def rank_diference(self,comb):
        y1, y2=comb[0],comb[1]
        res=1 if y1 >y2 else -1
        return res


    def aux_fun(self,comb):
        res=np.zeros((len(comb),1))
        for i,coso in enumerate(comb):
            res[i]=self.mapeo(coso[0])-self.mapeo(coso[1])



        return res
    def calcular_gradient(self,alpha,combs):

        import itertools
        #calculo solo los elementos de la diagonal superior puesto que la matriz es diagonal
        matriz=[]
        for i in range(len(combs)):
            comb=combs[i]
            temp=[]
            for j in range(i, len(combs)):
                comb2=combs[j]
                temp.append(temp.append(self.kernel(comb[0],comb2[0])-self.kernel(comb[0],comb2[1])-self.kernel(comb[1],comb2[0])+self.kernel(comb[1],comb2[1])))

            matriz.append(temp)
        #realizo la multiplicacion
    def fun(self,a):
        i = np.where(self.alpha == a)[0]
        prueba = self.Y_combs[i] * self.mapeo(self.combs[i])
        if a == 0:
            # si la condicion de abajo se cumple significa que no cumple KKT y que es apto para optimizar


            if prueba - 1 < -self.e:
              return 0
            return 1
        elif 0 < a and a < self.C:
            # si la condicion de abajo se cumple significa que no cumple KKT y que es apto para optimizar

            if abs(prueba - 1) > self.e:
                return 0
            return 1
        elif a == self.C:
            # si la condicion de abajo se cumple significa que no cumple KKT y que es apto para optimizar

            if prueba - 1 > self.e:
                return 0
            return 1


    def probar_KKT(self,alpha):
        indices_malos=[]
        p=list(map(self.fun,alpha))
        indices_malos=np.where(p==0)[0]


        return indices_malos
    def alpha_j(self,i,posibles_indices):

        E_i=self.mapeo(self.combs[i])-self.Y_combs[i]
        max=0
        indice=None

        for j in posibles_indices:
            if j!=i:
                E_j=self.mapeo(self.combs[j])-self.Y_combs[j]
                temp=E_i-E_j
                if abs(temp)>max:
                    max=temp
                    indice=j

            else:
                continue
        return indice

    #escojo el par de $\alpha's$ que voy a omptimizar
    def elegir_alpha(self,alpha,X):
        #Escojo el primer indice
        indices=list(self.todos_los_indices-set(self.visitados))

        alpha_i=None
        alpha_j=None

        potenial_alpha = int(np.random.choice(indices,1))
        start_time=t.clock()
        uno=self.mapeo(X[potenial_alpha])

        end_time=t.clock()
        #print('Tiempo en mapear: %0.5f'%((end_time-start_time)))

        prueba=self.Y_combs[potenial_alpha]*uno
        if alpha[potenial_alpha]==0:
                # si la condicion de abajo se cumple significa que no cumple KKT y que es apto para optimizar
               if prueba-1<-self.e:
                    alpha_i=alpha[potenial_alpha]
                    self.visitados.append(potenial_alpha)
        elif 0<alpha[potenial_alpha] and alpha[potenial_alpha]<self.C:
                # si la condicion de abajo se cumple significa que no cumple KKT y que es apto para optimizar

                if abs(prueba-1)>self.e:
                    alpha_i = alpha[potenial_alpha]
                    self.visitados.append(potenial_alpha)
        elif alpha[potenial_alpha]==self.C:
                # si la condicion de abajo se cumple significa que no cumple KKT y que es apto para optimizar

                if prueba-1>self.e:
                    alpha_i = alpha[potenial_alpha]
                    self.visitados.append(potenial_alpha)
        indices = list(self.todos_los_indices - set(self.visitados))
        #alpha_j=int(np.random.choice(indices,1))
        alpha_j=self.alpha_j(potenial_alpha,indices)

        self.visitados.append(alpha_j)


        return potenial_alpha,alpha_j



    def big_loss(self,alpha,Q):
        uno=np.ones((len(alpha),1)).T.dot(alpha)
        alpha=alpha.reshape((-1,1))
        z=self.Y_combs.reshape((-1,1))
        temp=np.multiply(z,alpha)
        dos=-(1/2)*temp.T.dot(Q).dot(temp)

        return -uno-dos





    def ordinal_kernel(self,comb,comb2):
        #En el articuo Heibrich et al 2000 las entradas de la matriz son (X_i^1-X_i^2)*(X_j^1-X_j^2) (con * como producto punto) esto se traduce en productos punto que al final se reemplazan con el kernel
       return self.kernel(comb[0], comb2[0]) - self.kernel(comb[0], comb2[1]) - self.kernel(comb[1], comb2[0]) + self.kernel(comb[1], comb2[1])

    def loss(self,alpha,a_old1,a_old2,y_1,y_2,x_1,x_2):
        comb=x_1
        comb2=x_2
        s=y_2*y_1
        v_1=y_1*(self.mapeo(comb)-a_old1*y_1*(self.ordinal_kernel(comb,comb)) -a_old2*y_2*(self.ordinal_kernel(comb,comb2))  )
        v_2=y_2*(self.mapeo(comb2)-a_old2*y_2*(self.ordinal_kernel(comb2,comb2)) -a_old1*y_1*(self.ordinal_kernel(comb,comb2)))
        res=alpha[0]*(1-y_1*v_1)+alpha[1]*(1-y_2*v_2)-(1/2)*self.ordinal_kernel(comb,comb)*((alpha[0])**2)-(1/2)*self.ordinal_kernel(comb2,comb2)*((alpha[1])**2)-(s*self.ordinal_kernel(comb,comb2))*alpha[0]*alpha[1]
        return -res

    def obtener_combinaciones(self,textos,y):
        import itertools
        numbers=range(1,7,1)
        x_res=[]
        y_res=[]
        for number in numbers:
            if number==1:
                indices1=(np.reshape(np.where(y==number+1)[0],[-1,1]))
                indices2= (np.reshape(np.where(y==number)[0],[-1,1]))
                temp=list(itertools.product(textos[indices1,:][:,0,:],textos[indices2,:][:,0,:]))
                temp2=list(itertools.product(textos[indices2,:][:,0,:],textos[indices1,:][:,0,:]))
                y_temp1=list(itertools.product(y[indices1],y[indices2]))
                y_temp2=list(itertools.product(y[indices2],y[indices1]))
                x_sub=np.vstack((temp,temp2))
                y_sub=np.vstack((y_temp1,y_temp2))


                x_res = x_sub
                y_res = y_sub


            elif number==6:
                # indices1=np.reshape(np.where(y==number-1)[0],[-1,1])
                # indices2=np.reshape(np.where(y==number)[0],[-1,1])
                # temp = list(itertools.product(textos[indices1, :][:, 0, :], textos[indices2, :][:, 0, :]))
                # temp2 = list(itertools.product(textos[indices2, :][:, 0, :], textos[indices1, :][:, 0, :]))
                # y_temp1 = list(itertools.product(y[indices1], y[indices2]))
                # y_temp2 = list(itertools.product(y[indices2], y[indices1]))
                # x_sub = np.vstack((temp, temp2))
                # y_sub = np.vstack((y_temp1, y_temp2))
                #
                # x_res = np.vstack((x_res, np.reshape(x_sub, [-1, 1])))
                # y_res = np.vstack((y_res, np.reshape(y_sub, [-1, 1])))
                continue

            else:
                # indices1=np.reshape(np.where(y==number-1)[0],[-1,1])
                # indices2=np.reshape(np.where(y==number+1)[0],[-1,1])
                # indices3=np.reshape(np.where(y==number)[0],[-1,1])
                # temp1= list(itertools.product(textos[indices1, :][:, 0, :], textos[indices3, :][:, 0, :]))
                # temp2=list(itertools.product(textos[indices3, :][:, 0, :], textos[indices1, :][:, 0, :]))
                # temp3=list(itertools.product(textos[indices3, :][:, 0, :], textos[indices2, :][:, 0, :]))
                # temp4=list(itertools.product(textos[indices2, :][:, 0, :], textos[indices3, :][:, 0, :]))
                #
                # ytemp1 = list(itertools.product(y[indices1], y[indices3]))
                # ytemp2 = list(itertools.product(y[indices3], y[indices1]))
                # ytemp3 = list(itertools.product(y[indices3], y[indices2]))
                # ytemp4 = list(itertools.product(y[indices2], y[indices3]))
                #
                # x_sub=np.vstack((np.vstack((temp1,temp2)),np.vstack((temp3,temp4))))
                # y_sub=np.vstack((np.vstack((ytemp1,ytemp2)),np.vstack((ytemp3,ytemp4))))
                # x_res = np.vstack((x_res, np.reshape(x_sub, [-1, 1])))
                # y_res = np.vstack((y_res, np.reshape(y_sub, [-1, 1])))
                indices1 = (np.reshape(np.where(y == number + 1)[0], [-1, 1]))
                indices2 = (np.reshape(np.where(y == number)[0], [-1, 1]))
                temp = np.array(list(itertools.product(textos[indices1, :][:, 0, :], textos[indices2, :][:, 0, :])))
                temp2 = np.array(list(itertools.product(textos[indices2, :][:, 0, :], textos[indices1, :][:, 0, :])))
                sections = np.random.randint(0, len(temp), 500)
                temp= temp[sections, :, :]
                temp2=temp2[sections, :, :]
                #temp=temp[]

                y_temp1 = np.array(list(itertools.product(y[indices1], y[indices2])))
                y_temp2 = np.array(list(itertools.product(y[indices2], y[indices1])))
                y_temp1=y_temp1[sections,:]
                y_temp2=y_temp2[sections,:]
                x_sub = np.vstack((temp, temp2))
                y_sub = np.vstack((y_temp1, y_temp2))

                x_res = np.vstack((x_res,x_sub))
                y_res = np.vstack((y_res,y_sub))

        return x_res,y_res





    def fit(self,textos,y):
        import itertools
        import  sys
        np.place(textos,np.isnan(textos),0)
        self.combs,y_usar=self.obtener_combinaciones(textos,y)

        self.Y_combs=np.array(list(map(self.rank_diference,y_usar)))



        sys.setrecursionlimit(10000)
        Q=np.zeros((len(y_usar),len(y_usar)))
        #comparador =SubsequenceStringKernel()
        
        
       # gram_matrix=comparador.string_kernel(textos,textos)
        #o = 1
        #utiles=np.where(self.Y_combs==1)[0]
        #combs_usar=self.combs[utiles]
        #y_usar=y_usar[utiles]
        
        #self.alpha=np.random.random_sample((len(y_usar),1))*self.C
        #self.todos_los_indices = set(range(0, len(self.alpha)))
        #print len(self.alpha)
        # while o!=0:
        #
        #     i,j=self.elegir_alpha(self.alpha,self.combs)
        #     options=(self.alpha[i],self.alpha[j],self.Y_combs[i],self.Y_combs[j],self.combs[i],self.combs[j])
        #
        #     s=self.Y_combs[i]*self.Y_combs[j]
        #     ro=self.alpha[i]+s*self.alpha[j]
        #     #constrains = {'type': 'eq', 'fun': lambda x: x[0]+s*x[1]-ro}
        #     #bounds=[(0,self.C),(0,self.C)]
        #     #x_0=np.random.random_sample((2, 1)) * self.C
        #
        #     # Utilizo el optimizador
        #     #alphas=optimize.minimize(self.loss,x0=x_0,args=options,constraints=constrains,bounds=bounds).x
        #     #Lo hago analiticamente
        #     E_i=self.mapeo(self.combs[i])-self.Y_combs[i]
        #     E_j=self.mapeo(self.combs[j])-self.Y_combs[j]-self.ordinal_kernel(self.combs[i],self.combs[j])
        #     n=2*self.ordinal_kernel(self.combs[i],self.combs[j])-self.ordinal_kernel(self.combs[i],self.combs[i])-self.ordinal_kernel(self.combs[j],self.combs[j])
        #     a_j_new=self.alpha[j]-(self.Y_combs[j]*E_i-E_j)/n
        #     H=max((0,self.alpha[j]-self.alpha[i])) if self.Y_combs[i]!=self.Y_combs[j] else max((0,self.alpha[j]-self.alpha[i]-self.C))
        #     L=min((self.C,self.alpha[j]-self.alpha[i])) if self.Y_combs[i]!=self.Y_combs[j] else min((self.C,self.alpha[j]+self.alpha[i]-self.C))
        #     if a_j_new>=H:
        #         a_j_new_clip=H
        #     elif L< a_j_new and a_j_new<H :
        #         a_j_new_clip=a_j_new
        #     elif a_j_new<=L:
        #         a_j_new_clip=L
        #     a_i_new=self.alpha[i]+s*(self.alpha[j]-a_j_new_clip)
        #
        #     self.alpha[i]=a_i_new
        #     self.alpha[j]=a_j_new_clip
        #
        #     if len(self.visitados)==len(self.alpha):
        #         print "Ya acabe, probare KKT en todos los datos"
        #         start=t.clock()
        #         probar=self.probar_KKT(self.alpha)
        #         end=t.clock()
        #         print "tiempo en probar KKT: %0.5f"%(end-start)
        #         print len(probar)
        #         nada=input("Enter para continuar")
        #         if len(probar)==0:
        #             o=0
        #         elif len(self.visitados)==len(self.alpha):
        #             print  "Si hay datos que no lo cumplen los vulevo a meter en la bolsa de posibles valores"
        #             self.visitados=list(set(self.visitados)-set(probar))
        boundes=[]
        print "%i"%len(self.combs)

        start=t.clock()
        for i, comb in enumerate(self.combs):
             boundes.append((0, self.C))



             for j in range(i+1,len(self.combs)):
                     comb2=self.combs[j]
                 #temp.append(comparador._K(s=comb[0], t=comb2[0]) / ((comparador._K(s=comb[0], t=comb[0]) * comparador._K(s=comb2[0], t=comb2[0])) ** (1/ 2)) - comparador._K(s=comb[0], t=comb2[1]) / (( comparador._K(s=comb[0], t=comb[0]) * comparador._K(s=comb2[1], t=comb2[1]))** (1 / 2)) - comparador._K(s=comb[1], t=comb2[0]) / ( ( comparador._K(s=comb[1], t=comb[1]) * comparador._K(s=comb2[0], t=comb2[0])) ** (1 / 2)) + comparador._K(s=comb2[1], t=comb[1]) / (( comparador._K(s=comb2[1], t=comb2[1]) * comparador._K(s=comb[1], t=comb[1])) ** (1 / 2)))
                     Q[i,j]=self.kernel(comb[0],comb2[0])-self.kernel(comb[0],comb2[1])-self.kernel(comb[1],comb2[0])+self.kernel(comb[1],comb2[1])
                     Q[j,i]=Q[i,j]

        end=t.clock()
        print "Tiemop en crear la matriz: %0.5f"%(end-start)

        constrains = {'type': 'eq', 'fun': lambda x: x.dot(np.array(self.Y_combs))}
        options=(Q)
        x_0=np.random.random_sample((len(y_usar),1))*self.C
        self.alpha=optimize.minimize(self.big_loss, x0=x_0, args=options, constraints=constrains, bounds=boundes,maxiter=100000).x
        support_vectors=np.where(self.alpha!=0)[0]
        self.sup_vectors=self.combs[support_vectors]
        self.indx=support_vectors
        
    def matriz(self,combs):
        resta=lambda x:x[0]-x[1]
        x_new=np.array(list(map(resta,combs)))
        return x_new


    def mapeo(self,x):
        #x: TEXTO
        #DTYPE: string

        #matriz=self.matriz(self.combs)
        res=0
        #valor = comparador._K(x, x)
        ind=0
        combinaciones=self.combs
        if len(x)!=20:
            x=x[0]-x[1]



        if len(x)==20:

             for par in combinaciones:

             #res+=self.alpha[ind]*self.Y_combs[ind]*(comparador._K(par[0],x))/((comparador._K(par[0],par[0])*valor)**(1/2))-comparador._K(par[1],x)/((comparador._K(par[1],par[1])*valor)**(1/2))

                 res += self.alpha[ind] * self.Y_combs[ind] *(self.kernel(par[0],x)-self.kernel(par[1],x))
                 ind += 1

        #     temp=np.multiply(self.alpha,self.Y_combs)[:,None]*matriz[None,:]
        # for elem in temp:
        #
        #     res+=self.kernel()
        else:
            for par in combinaciones:
                # res+=self.alpha[ind]*self.Y_combs[ind]*(comparador._K(par[0],x))/((comparador._K(par[0],par[0])*valor)**(1/2))-comparador._K(par[1],x)/((comparador._K(par[1],par[1])*valor)**(1/2))
                res += self.alpha[ind] * self.Y_combs[ind] * (self.ordinal_kernel(par,x))


                ind += 1
        return res

    def crearTheta(self):
        x="hola"
        lol=np.where(np.multiply(np.array(self.Z),np.array(map(self.mapeo,self.combs[:][0]))-np.array(map(self.mapeo,self.combs[:][1]))>=1))



        nada=0

        theta={}
        calificaciones =range(1,6,1)
        for rank in calificaciones:
            temp = []
            for indx in self.indx:
                par=self.Y_combs[indx]

                if par[0]-par[1]==1:
                    temp.append(self.combs[indx])



            theta[rank]=temp
        tresholds={}
        for key, lista in theta.items():
            coso=self.aux_fun(np.array(lista))
            mejor_comb=lista[np.argmin(coso)]
            tresholds[key]=(self.mapeo(mejor_comb[0])+self.mapeo(mejor_comb[1]))/2


        self.umbrales=tresholds





















