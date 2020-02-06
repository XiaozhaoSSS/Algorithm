from tree import DecisionTreeNode,CARTRegressor

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder 


class GBDTClassfier(CARTRegressor):
    def __init__(self,learning_rate=1,n_trees=None,min_samples_leaf=None,max_depth=None):
        #super().__init__(min_samples_leaf ,max_depth)
        self.learning_rate = learning_rate
        self.n_trees = n_trees
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.trees=[]
        
    def fit(self,data):
        data=np.array(data)
        label=np.array(datasets)[:,-1]  #标签
        data=data[:,:-1]  #去掉标签
        
        self.K=len(np.unique(label))  #类别个数
        m=len(data)   #样本个数
        self.trees=np.zeros((self.n_trees,self.K)).tolist()
        
        #计算每个类别占比
        label_dict=pd.value_counts(label,normalize=True).to_dict()
        #独热编码
        y=OneHotEncoder(sparse = False).fit_transform(label.reshape(-1,1))
        
        #初始化F
        F=np.zeros((m,self.K))
        for i in range(self.K):
            F[:,i]=[label_dict[i]]*m
        
        for i in range(self.n_trees):
            P=np.exp(F)
            P=P/np.sum(P,axis=1).reshape(-1,1)
            for k in range(self.K):
                yk=y[:,k]-P[:,k]
                next_data=np.hstack((data,yk.reshape(-1,1)))
                
                self.trees[i][k]=CARTRegressor(min_samples_leaf=self.min_samples_leaf,max_depth=self.max_depth,is_gradient=True,K=self.K)
                self.trees[i][k].fit(next_data)

                gamma_index_list=self.trees[i][k].root.print_leaf_node()  #经决策树拟合后各叶子节点最佳负梯度拟合值及包含样本的index
                gammas=[x[0] for x in gamma_index_list]
                index_list=[x[1] for x in gamma_index_list]
                
                #更新F的k列
                for index,gamma in zip(index_list,gammas):
                    #index=index_list[i]
                    #gamma=gammas[i]
                    for j in range(len(index)):
                        F[index[j],k]=F[index[j],k]+self.learning_rate*gamma

    def predict(self,data):
        m=len(data)  #样本个数
        F=np.zeros((m,self.K))
        for i in range(self.n_trees):
            for k in range(self.K):
                F[:,k]=F[:,k]+self.learning_rate*self.trees[i][k].predict(data)
        P=np.exp(F)
        P=P/np.sum(P,axis=1).reshape(-1,1)
        y_pre=np.argmax(P,axis=1)
        return y_pre

if __name__=='__main__':
    datasets = [[6,0],
               [12,0],
               [14,0],
               [18,0],
               [20,0],
               [65,1],
               [31,1],
               [40,1],
               [1,1],
               [2,1],
               [100,2],
               [101,2],
               [65,2],
               [54,2]]

    print('================================GBDT分类结果================================')
    model=GBDTClassfier(n_trees=5,min_samples_leaf=1,max_depth=2)
    model.fit(datasets)
    X_test=[[25],[100],[10],[900],[-10]]
    pred=model.predict(X_test)
    print(pred)
    
    from sklearn.ensemble import GradientBoostingClassifier

    print('================================GBDT分类结果_sklearn实现================================')
    gbdt = GradientBoostingClassifier(loss='deviance', learning_rate=1, min_samples_leaf=1, max_depth=2,
                                      init=None, random_state=None,verbose=0, max_leaf_nodes=None,)

    X,y=np.array(datasets)[:,:-1],np.array(datasets)[:,-1:]
    gbdt.fit(X,y)
    pred = gbdt.predict(X_test)
    print(pred)