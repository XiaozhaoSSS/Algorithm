#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
GEM算法实现高斯混合分布模型求解
初始值的获取与K-means++相同
"""
import numpy as np
#from k_means_plus import k_means_plus
from collections import defaultdict
from sklearn.cluster import KMeans
import math
import copy

get_ipython().run_line_magic('matplotlib', 'inline')

class GEM:
    def __init__(self,K=4,epochs=1000,epsilon=1e-3):
        self.K=K  # 高斯混合模型的个数
        self.N=None  #观测数据的个数
        self.m=None  #数据的维度
        self.epsilon=epsilon
        
        self.mu=None  # 高斯模型的均值  K*m
        self.sigma=None  # 高斯模型的标准差  K*m*m
        self.alpha=None  # 高斯模型的系数   K*1
        self.gamma=None  # 模型对观测数据的响应度   K*N
        
        self.epochs=epochs
    def init_params(self,data):
        '''
        用K-means取参数的初始值
        '''
        #km=k_means_plus()
        #res=km.kpp_centers(data, self.K)
        self.N=np.shape(data)[0]
        self.m=np.shape(data)[1]
        
        KMEANS = KMeans(n_clusters=self.K).fit(data)
        clusters = defaultdict(list)
        for ind, label in enumerate(KMEANS.labels_):
            clusters[label].append(ind)
        mu = []
        alpha = []
        sigma = []
        for inds in clusters.values():
            partial_data = data[inds]
            mu.append(partial_data.mean(axis=0))  # 分模型的均值向量
            alpha.append(len(inds) / self.N)  # 权重
            sigma.append(np.cov(partial_data.T))  # 协方差,m个维度间的协方差
        self.mu = np.array(mu)
        self.alpha = np.array(alpha)
        self.sigma = np.array(sigma)
        
        return
        
    def phi(self,y,mu,sigma):
        '''
        高斯分布密度函数
        '''
        s1 = 1.0 / math.sqrt(np.linalg.det(sigma))
        s2 = np.linalg.inv(sigma)  # m*m
        delta = np.array([y - mu])  # 1*m
        return s1 * math.exp(-1.0 / 2 * delta @ s2 @ delta.T)
    
    def fit(self,data):
        '''
        迭代训练
        '''
        data=np.array(data)
        #初始化参数
        self.init_params(data)
        
        for epoch in range(self.epochs):
            old_alpha = copy.copy(self.alpha)
            
            gamma=[]  #存储所有模型对观测数据的响应度
            
            #E步,计算所有模型对观测数据的响应度
            for k in range(self.K):
                gamma_k=[]   #存储模型k对观测数据的响应度
                alpha_k=self.alpha[k]   #模型k的权重
                mu_k=self.mu[k]      #模型k的均值
                sigma_k=self.sigma[k]   #模型k的标准差
                for j in range(self.N):
                    gamma_k.append(alpha_k*self.phi(data[j],mu_k,sigma_k))
                gamma_s=np.sum(gamma_k)
                gamma_k=[i/gamma_s for i in gamma_k]
                gamma.append(gamma_k)
            gamma=np.array(gamma)
            
            #M步，更新参数
            for k in range(self.K):
                SUM=np.sum(gamma[k])
                self.mu[k]=np.dot(gamma[k],data)/SUM
                self.sigma[k] = sum([curr_gamma * (np.outer(np.transpose([curr_delta]), curr_delta)) 
                                     for curr_gamma, curr_delta in zip(gamma[k], data-self.mu[k])]) / SUM
                self.alpha[k]=SUM/self.N
                
            if np.linalg.norm(self.alpha - old_alpha, 1) < self.epsilon:
                break
        self.gamma=gamma
        return
    
    def predict(self):
        cluster = defaultdict(list)
        for j in range(self.N):
            max_ind = np.argmax(self.gamma[:,j])
            cluster[max_ind].append(j)
        return cluster

if __name__ == '__main__':
    def generate_data(N=500):
        X = np.zeros((N, 2))  # N*2, 初始化X
        mu = np.array([[5, 35], [20, 40], [20, 35], [45, 15]])
        sigma = np.array([[30, 0], [0, 25]])
        for i in range(N): # alpha_list=[0.3, 0.2, 0.3, 0.2]
            prob = np.random.random(1)
            if prob < 0.1:  # 生成0-1之间随机数
                X[i, :] = np.random.multivariate_normal(mu[0], sigma, 1)  # 用第一个高斯模型生成2维数据
            elif 0.1 <= prob < 0.3:
                X[i, :] = np.random.multivariate_normal(mu[1], sigma, 1)  # 用第二个高斯模型生成2维数据
            elif 0.3 <= prob < 0.6:
                X[i, :] = np.random.multivariate_normal(mu[2], sigma, 1)  # 用第三个高斯模型生成2维数据
            else:
                X[i, :] = np.random.multivariate_normal(mu[3], sigma, 1)  # 用第四个高斯模型生成2维数据
        return X

    data = generate_data()
    gem = GEM()
    gem.fit(data)
    # print(gem.alpha, '\n', gem.sigma, '\n', gem.mu)
    cluster = gem.predict()
    
    print(gem.mu,'\n',gem.sigma)
    
    import matplotlib.pyplot as plt
    from itertools import cycle
    colors = cycle('grbk')
    for color, inds in zip(colors, cluster.values()):
        partial_data = data[inds]
        plt.scatter(partial_data[:,0], partial_data[:, 1], edgecolors=color)
    plt.show()


# In[ ]:




