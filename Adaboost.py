#!/usr/bin/env python
# coding: utf-8

# In[3]:


#以CART做弱分类器
from tree import DecisionTreeNode,CARTRegressor,CARTClassifier

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import operator
import math

class AdaboostClassfier(CARTClassifier):
    def __init__(self,K=100,min_samples_leaf=1,max_depth=2,epsilon=1e-5):
        self.K=K  #弱分类器个数
        
        self.min_samples_leaf=min_samples_leaf
        self.max_depth=max_depth
        
        self.epsilon=epsilon   #误差率阈值
        
        self.alphas=[]  #弱分类器权重
        self.models=[]  #弱分类器
    
    def fit(self,data):
        data=np.array(data)
        X,y=data[:,:-1],data[:,-1]
        m=len(X)  #样本个数
        
        D=[1/m]*m#初始化样本权重矩阵D
        self.models=[0]*self.K
        
        for k in range(self.K):
            #训练第k个弱分类器
            self.models[k]=CARTClassifier(min_samples_leaf=self.min_samples_leaf,max_depth=self.max_depth)
            self.models[k].fit(data,D)
            
            #计算误差率e
            e=0
            leaf_node_list=self.models[k].root.print_leaf_node()
            curr_res=[]   #记录当前结果，如果分类正确记为1，错误记为-1，以便更新权重
            for label,index_list in leaf_node_list:
                for index in index_list:
                    if data[index][-1]!=label:
                        curr_res.append((index,-1))
                        e+=D[index]
                    else:
                        curr_res.append((index,1))
                        
            #计算弱分类器权重alpha
            alpha=np.log((1-e)/(e+1e-8))/2
            self.alphas.append(alpha)
            
            #更新样本权重
            curr_res=sorted(curr_res,key=operator.itemgetter(0))
            D=[d*np.exp(-res[1]*alpha) for d,res in zip(D,curr_res)]
            z=sum(D)
            D=[x/z for x in D]
            if e<self.epsilon:break
        
    def predict(self,data):
        res=[0]*len(data)
        for k in range(self.K):
            if self.models[k]==0:break
            res=[x+self.alphas[k]*y for x,y in zip(res,self.models[k].predict(data))]
        res=[1 if x>0 else -1 for x in res]
        return res
    
class AdaboostRegressor(CARTRegressor):
    def __init__(self,K=100,min_samples_leaf=1,max_depth=2):
        self.K=K  #弱分类器个数
        self.alphas=[]  #弱分类器权重
        self.models=[]  #弱分类器
        
        self.min_samples_leaf=min_samples_leaf
        self.max_depth=max_depth
        
    def fit(self,data):
        data=np.array(data)
        m=len(data)  #样本个数
        self.models=[0]*self.K
        
        D=[1/m]*m   #初始化样本权重D
        
        for k in range(self.K):
            #训练第k个弱分类器
            self.models[k]=CARTRegressor(min_samples_leaf=self.min_samples_leaf,max_depth=self.max_depth)
            self.models[k].fit(data,D)
            
            e_list=[]   #训练集误差
            leaf_node_list=self.models[k].root.print_leaf_node()
            for value,index_list in leaf_node_list:
                for index in index_list:
                    e_list.append((index,abs(data[index][-1]-value)))
            e_list=sorted(e_list,key=operator.itemgetter(0))
            e_list=[x[1] for x in e_list]
            
            e_max=max(e_list)    #计算训练集上最大误差
            e_list=[x/e_max for x in e_list]    #计算每个样本的相对误差,以线性误差为例
            e=sum([x*y for x,y in zip(D,e_list)])   #计算样本误差率

            alpha=e/(1-e)   #计算弱分类器权重alpha
            self.alphas.append(alpha)
            
            D=[x*alpha**(1-y) for x,y in zip(D,e_list)]
            Z=sum(D)    #规范化因子
            D=[x/Z for x in D]#更新样本权重
            
    def predict(self,data):
        k_res=sorted(enumerate([math.log(1/x) for x in self.alphas]), key=operator.itemgetter(1))[(self.K+1)//2-1][0]
        model=self.models[k_res]
        return model.predict(data)


# In[ ]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train,X_test,y_train,y_test=train_test_split(X[50:,],y[50:],test_size=0.2)
y_train=np.array([-1 if x==1 else 1 for x in y_train])
y_test=np.array([-1 if x==1 else 1 for x in y_test])

print("========================Adaboost训练iris数据集========================")
model=AdaboostClassfier()
model.fit(np.hstack((X_train,y_train.reshape((-1,1)))))
print('预测结果：',model.predict(X_test))

print("========================sklearn实现========================")
from sklearn.ensemble import AdaBoostClassifier as AdaBoostClassifier_sklearn
clf = AdaBoostClassifier_sklearn(random_state=0,n_estimators=100)
clf.fit(X_train,y_train)
print('预测结果：',clf.predict(X_test))


# In[11]:


datasets = [[1,4.5],
               [2,4.75],
               [3,4.91],
               [4,5.34],
               [5,5.80],
               [6,7.05],
               [7,7.9],
               [8,8.23],
               [9,8.7],
               [10,9.0]]
print('================================Adaboost回归结果================================')

model=AdaboostRegressor(max_depth=4)
model.fit(datasets)
print('预测结果：',model.predict([[1.8],[6],[-4]]))

from sklearn.ensemble import AdaBoostRegressor as AdaboostRegressor_sklearn
reg=AdaboostRegressor_sklearn(random_state=0, n_estimators=100)
X,y=np.array(datasets)[:,:-1],np.array(datasets)[:,-1]
reg.fit(X,y)
print('预测结果：',reg.predict([[1.8],[6],[-4]]))

