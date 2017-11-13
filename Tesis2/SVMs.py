# Substring kernel

import time as t
import numpy as np
from modshogun import StringCharFeatures,RAWBYTE
from shogun.Kernel import SubsequenceStringKernel
from sklearn import svm
from sklearn.datasets import fetch_20newsgroups
from sklearn.svm import libsvm
import sys
from time import time
from functools import wraps
from scipy import optimize




class OrdinalSM():


    def __init__(self,c,e=10e-3):

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

        for i in range(len(combs)):
            comb=combs[i]
            temp=[]
            for j in range(i, len(combs)):
                comb2=combs[j]
                temp.append(self.ordinal_kernel(comb,comb2))

            matriz.append(temp)
        #realizo la multiplicacion
    def fun(self,i):
        a=self.alpha[i]
        prueba = self.Y_combs[i] * self.mapeo(self.combs[i])
        if a == 0:
            # si la condicion de abajo se cumple significa que no cumple KKT y que es apto para optimizar


            if prueba-1<-self.e:
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


    def probar_KKT(self):
        temp=list(self.todos_los_indices)
        p=np.array(list(map(self.fun,temp)))
        indices_malos=np.where(p==0)[0]


        return indices_malos
    def alpha_j(self,elem):

        E_j=self.mapeo(self.combs[elem])-self.Y_combs[elem]
        return self.E_i-E_j



    #escojo el par de $\alpha's$ que voy a omptimizar
    def elegir_alpha(self,alpha,X):
        #Escojo el primer indice
        indices=list(self.todos_los_indices-set(self.visitados))
        if len(indices)==0:
            return 0, 0


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

        if len(indices)==0:
            return potenial_alpha,int(np.random.choice(np.array(list(self.todos_los_indices)),1))

        alpha_j=int(np.random.choice(indices,1))
        #alpha_j=self.alpha_j(potenial_alpha,indices)

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
                sections = np.random.randint(0, len(temp), 100)
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
        if len(y_usar) % 2 == 1:
            self.combs.remove(self.combs[-1])
            y_usar.remove(y_usar[-1])

        self.Y_combs=np.array(list(map(self.rank_diference,y_usar)))



        sys.setrecursionlimit(10000)
        Q=np.zeros((len(y_usar),len(y_usar)))
        #comparador =SubsequenceStringKernel()
        
        MAX_ITER=1000
        #gram_matrix=comparador.string_kernel(textos,textos)
        o = 0

        self.alpha=np.ones((len(y_usar),1))
        self.todos_los_indices = set(range(0, len(self.alpha)))
        #self.probar_KKT(self.alpha)
        s=t.clock()
        self.mapeo(self.combs[0])
        e=t.clock()
        print "tiempo en hacer 1 mapeo: %0.5f"%(e-s)
        print "Tamano de alpha %i"%len(self.alpha)
        todos=list(self.todos_los_indices)
        epsilon=10e-3



















        # while 1:
        #     for i in todos:
        #
        #         #self.E_i=E_i
        #         j=int(np.random.randint(0,len(self.alpha),1))
        #         while j==i:
        #             j=int(np.random.randint(0,len(self.alpha),1))
        #
        #
        #         #i,j=self.elegir_alpha(self.alpha,self.combs)
        #         #t1=t.clock()
        #         #j=np.argmax(np.array(list(map(self.alpha_j,todos))))
        #         #t2=t.clock()
        #         #print "tiempo en hallar el ideal: %0.9f"%(t2-t1)
        #
        #         #options=(self.alpha[i],self.alpha[j],self.Y_combs[i],self.Y_combs[j],self.combs[i],self.combs[j])
        #
        #         s=self.Y_combs[i]*self.Y_combs[j]
        #         #ro=self.alpha[i]+s*self.alpha[j]
        #         #constrains = {'type': 'eq', 'fun': lambda x: x[0]+s*x[1]-ro}
        #         #bounds=[(0,self.C),(0,self.C)]
        #         #x_0=np.random.random_sample((2, 1)) * self.C
        #
        #         # Utilizo el optimizador
        #         #alphas=optimize.minimize(self.loss,x0=x_0,args=options,constraints=constrains,bounds=bounds).x
        #         #Lo hago analiticamente
        #
        #         E_i = self.mapeo(self.combs[i]) - self.Y_combs[i]
        #
        #         E_j=self.mapeo(self.combs[j])-self.Y_combs[j]
        #
        #         n=(self.ordinal_kernel(self.combs[i],self.combs[i])+self.ordinal_kernel(self.combs[j],self.combs[j])-2*self.ordinal_kernel(self.combs[i],self.combs[j]))
        #         if n==0:
        #             continue
        #         a_j_new=self.alpha[j]-float(self.Y_combs[j]*E_i-E_j)/n
        #
        #         H= min((self.C,self.C+(self.alpha[j]-self.alpha[i]))) if self.Y_combs[i]!=self.Y_combs[j] else min((self.C,self.alpha[j]+self.alpha[i]))
        #         L=  max((0,self.alpha[j]-self.alpha[i])) if self.Y_combs[i]!=self.Y_combs[j] else  max((0,self.alpha[j]+self.alpha[i]-self.C))
        #         if a_j_new>=H:
        #             a_j_new_clip=H
        #         elif L< a_j_new and a_j_new<H :
        #             a_j_new_clip=a_j_new
        #         elif a_j_new<=L:
        #             a_j_new_clip=L
        #         a_i_new=self.alpha[i]+s*(self.alpha[j]-a_j_new_clip)
        #
        #         self.alpha[i]=a_i_new
        #         self.alpha[j]=a_j_new_clip
        #
        #     print "Ya acabe, probare KKT en todos los datos"
        #     start=t.clock()
        #     probar=self.probar_KKT()
        #     end=t.clock()
        #     print "tiempo en probar KKT: %0.5f"%(end-start)
        #     print len(probar)
        #     #nada=input("Enter para continuar")
        #     if len(probar)==0 or o==MAX_ITER:
        #         break
        #     else:
        #         print  "Si hay datos que no lo cumplen los vulevo a meter en la bolsa de posibles valores"
        #         self.visitados=list(set(self.visitados)-set(probar))
        #         #alpha = self.alpha.copy()
        #         o+=1
        #         print "Veces que he visitado los datos: %i"%o
        #
        # print len(self.probar_KKT())
        # print "Termine"
        #
        # np.save('/home/luis/alphas',self.alpha)
        # boundes=[]
        # # print "%i"%len(self.combs)
        # #
        # start=t.clock()
        # for i, comb in enumerate(self.combs):
        #      boundes.append((0, self.C))
        #
        #
        #
        #      for j in range(i+1,len(self.combs)):
        #              comb2=self.combs[j]
        #          #temp.append(comparador._K(s=comb[0], t=comb2[0]) / ((comparador._K(s=comb[0], t=comb[0]) * comparador._K(s=comb2[0], t=comb2[0])) ** (1/ 2)) - comparador._K(s=comb[0], t=comb2[1]) / (( comparador._K(s=comb[0], t=comb[0]) * comparador._K(s=comb2[1], t=comb2[1]))** (1 / 2)) - comparador._K(s=comb[1], t=comb2[0]) / ( ( comparador._K(s=comb[1], t=comb[1]) * comparador._K(s=comb2[0], t=comb2[0])) ** (1 / 2)) + comparador._K(s=comb2[1], t=comb[1]) / (( comparador._K(s=comb2[1], t=comb2[1]) * comparador._K(s=comb[1], t=comb[1])) ** (1 / 2)))
        #              Q[i,j]=self.kernel(comb[0],comb2[0])-self.kernel(comb[0],comb2[1])-self.kernel(comb[1],comb2[0])+self.kernel(comb[1],comb2[1])
        #              Q[j,i]=Q[i,j]

        # constrains = {'type': 'eq', 'fun': lambda x: x.dot(np.array(self.Y_combs))}
        # options=(Q)
        # x_0=np.random.random_sample((len(y_usar),1))*self.C
        # self.alpha=optimize.minimize(self.big_loss, x0=x_0, args=options, constraints=constrains, bounds=boundes,maxiter=100000).x
        # support_vectors=np.where(self.alpha!=0)[0]
        # self.sup_vectors=self.combs[support_vectors]
        # self.indx=support_vectors
        
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






        theta={}
        calificaciones =range(1,6,1)


        self.umbrales=tresholds





















