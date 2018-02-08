# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 15:02:17 2018

@author: weng
"""
import numpy as np
import matplotlib.pyplot as plt

dataMat = []; labelMat = []  
fr = open('C:/Users/weng/Desktop/xg.txt')  
for line in fr.readlines():  
    lineArr = line.strip().split()  
    dataMat.append([float(lineArr[1]), float(lineArr[2])])  
    labelMat.append(int(lineArr[3]))  
    
x=dataMat
y=np.mat(labelMat).T
trainnum1=8
trainnum2=9
w0=np.mat([1 , -1]).T
beta=np.vstack((w0,1))
x_bar=np.hstack((x,np.ones([trainnum1+trainnum2,1])))

for count in range(100):
    betax=np.dot(np.transpose(beta),np.transpose(x_bar))
    p1=np.exp(betax)/(1+np.exp(betax))
    
    dl=-sum(np.multiply(x_bar,y-p1.T)).T
    ddl=0
    for i in range(len(x)):
        mt=np.mat(x_bar[i,:])
        ddl=ddl+np.dot(mt.T,mt)*p1[0,i]*(1-p1[0,i])    
    beta=beta-np.dot(ddl.I,dl)
    z=np.dot(np.transpose(beta),np.transpose(x_bar))
    yy=1/(1+np.exp(-betax))
    yy=yy.reshape(trainnum1+trainnum2,1)
    if np.linalg.norm(y-yy,ord=2)<1e-6: #Stopping Criterion
        break

# plot classfy line
x_test=np.linspace(0,0.8,100)
y_test=(-beta[2]-beta[0]*x_test)/beta[1]
x=np.mat(x)
plt.plot(x[0:trainnum1,0],x[0:trainnum1,1],'*')
plt.plot(x[trainnum1:-1,0],x[trainnum1:-1,1],'+')

plt.plot(x_test,np.asarray(y_test)[0],'--')