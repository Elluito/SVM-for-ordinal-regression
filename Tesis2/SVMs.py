# Substring kernel


import numpy as np

# from shogun.Kernel import StringSubsequenceKernel
from sklearn import svm
from sklearn.datasets import fetch_20newsgroups
from sklearn.svm import libsvm
import sys
from time import time
from functools import wraps
from scipy import optimize

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
    # def fit(self, X, Y):
    #     gram_matr = self.string_kernel(X, X)
    #     self.__X = X
    #     super(svm.SVC, self).fit(gram_matr, Y)
    #

    # def predict(self, X):
    #     svm_type = LIBSVM_IMPL.index(self.impl)
    #     if not self.__X:
    #         print('You should train the model first!!!')
    #         sys.exit(3)
    #     else:
    #         gram_matr_predict_new = self.string_kernel(X, self.__X)
    #         gram_matr_predict_new = np.asarray(gram_matr_predict_new, dtype=np.float64, order='C')
    #         return libsvm.predict(
    #             gram_matr_predict_new, self.support_, self.support_vectors_, self.n_support_,
    #             self.dual_coef_, self._intercept_,
    #             self._label, self.probA_, self.probB_,
    #             svm_type=svm_type,
    #             kernel=self.kernel, C=self.C, nu=self.nu,
    #             probability=self.probability, degree=self.degree,
    #             shrinking=self.shrinking, tol=self.tol, cache_size=self.cache_size,
    #             coef0=self.coef0, gamma=self._gamma, epsilon=self.epsilon)
    #
def loss(alpha,Q,y,z):
    Z=np.diag(np.array(z))
    parte1=np.ones((len(y),1)).T.dot(alpha)
    i=0
    while i<len(Q):
        for j,list in Q:
            alpha[i]
        i+=1
    parte2=-(1/2)*(alpha.T.dot(Z.T).dot(Q).dot(Z).dot(alpha))
    return -parte1-parte2


class OrdinalSM():


    def __init__(self,c):

        self.C=c
        self.sigma=2


    def kernel(self,x1,x2):

        return np.exp(((x1-x2).dot(x1-x2))/(2*self.sigma))
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

    def elegir_alpha(self):
        nada=0


    def loss(self,alpha):





    def fit(self,textos,y,max_iter):
        import itertools
        import  sys
        coso = itertools.product(textos, textos)
        y_usar = list(itertools.product(y, y))
        self.combs=list(coso)
        self.Y_combs=np.array(map(self.rank_diference,y_usar))


        combinaciones=list(self.combs)
        sys.setrecursionlimit(10000)
        Q=[]
        #comparador =SubsequenceStringKernel()
        boundes = []
        z = []
       # gram_matrix=comparador.string_kernel(textos,textos)
        o = 0
        utiles=np.where(self.Y_combs==1)[0]
        combs_usar=self.combs[utiles]
        y_usar=y_usar[utiles]
        self.alpha=np.random_sample((len(y_usar),1))*self.C

        while o< max_iter:

            i,j=self.elegir_alpha()
            self.l














            o+=1



        # for i, comb in enumerate(combinaciones):
        #     boundes.append((0, self.C))
        #     z.append(self.rank_diference(y_usar[o]))
        #     temp = []
        #     for j, comb2 in enumerate(combinaciones):
        #
        #         #temp.append(comparador._K(s=comb[0], t=comb2[0]) / ((comparador._K(s=comb[0], t=comb[0]) * comparador._K(s=comb2[0], t=comb2[0])) ** (1/ 2)) - comparador._K(s=comb[0], t=comb2[1]) / (( comparador._K(s=comb[0], t=comb[0]) * comparador._K(s=comb2[1], t=comb2[1]))** (1 / 2)) - comparador._K(s=comb[1], t=comb2[0]) / ( ( comparador._K(s=comb[1], t=comb[1]) * comparador._K(s=comb2[0], t=comb2[0])) ** (1 / 2)) + comparador._K(s=comb2[1], t=comb[1]) / (( comparador._K(s=comb2[1], t=comb2[1]) * comparador._K(s=comb[1], t=comb[1])) ** (1 / 2)))
        #         temp.append(self.kernel(comb[0],comb2[0])-self.kernel(comb[0],comb2[1])-self.kernel(comb[1],comb2[0])+self.kernel(comb[1],comb2[1]))
        #
        #     Q.append(temp)
        #     o += 1
        # constrains = {'type': 'eq', 'fun': lambda x: x.dot(np.array(z))}
        self.alpha=optimize.minimize(loss,np.zeros((len(z),1)),(Q,y,z),bounds=boundes,constraints=constrains).x

        support_vectors=np.where(self.alpha!=0)[0]
        self.sup_vectors=self.combs[support_vectors]
        self.indx=support_vectors
        self.Z=list(z)

        # for ind in support_vectors:
        #     temp=combinaciones[ind]
        #     res+=self.alpha[ind]*z[ind]*comparador._K(temp[0],temp[1])
        #





    def mapeo(self,x):
        #x: TEXTO
        #DTYPE: string

        indices=self.indx
        comparador= StringKernel()
        res=0
        valor = comparador._K(x, x)
        for ind in indices:
            par=self.combs[ind]

            res+=self.alpha[ind]*self.z[ind]*(comparador._K(par[0],x))/((comparador._K(par[0],par[0])*valor)**(1/2))-comparador._K(par[1],x)/((comparador._K(par[1],par[1])*valor)**(1/2))
            #res += self.alpha[ind] * self.z[ind] *(self.kernel(par[0],x)-self.kernel(par[1],x))

        return res

    def crearTheta(self):
        x="hola"
        lol=np.where(np.multiply(np.array(self.Z),np.array(map(self.mapeo,self.combs[:,0]))-np.array(self.mapeo,self.combs[:,1]))>=1)



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





















