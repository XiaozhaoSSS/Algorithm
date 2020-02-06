#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import math
from abc import abstractmethod
import operator

# 定义节点类
class DecisionTreeNode:
    def __init__(self,  label=None, feature_name=None, feature=None,value=None,split_point=None,data_index=None):
        self.label = label
        self.feature_name = feature_name
        self.feature = feature
        self.tree = {}
        self.value=value
        self.split_point=split_point
        self.data_index=data_index
    '''
    def feature_name(self):
        return self.feature_name_list(self.feature)
    '''
    def display(self,feature_name_list=None):
        '''
        将树打印出来'''
        if feature_name_list:featurename=feature_name_list[self.feature] if self.feature!=None else None
        else:featurename=self.feature
        res={'label':self.label,'feature':featurename,'tree':{}}

        if self.value:res['value']=self.value
        if self.split_point:res['split_point']=self.split_point
        for next_node in self.tree:
            res['tree'][next_node]=self.tree[next_node].display(feature_name_list)
        return res
    def print_leaf_node(self):
        res=[]
        def dfs(node):
            if node.tree=={}:
                res.append((node.label,node.data_index))
                return
            for next_node in node.tree:
                dfs(node.tree[next_node])
            return
        dfs(self)
        return res

class BaseDecisionTree:
    def __init__(self,epsilon=1e-3,min_samples_leaf=1,max_depth=float('inf'),is_gradient=False,K=None):
        self.root=None
        self.epsilon=epsilon  # 信息增益/信息增益比/Gini小于该阈值时，算法停止
        self.min_samples_leaf=min_samples_leaf  #叶子节点拥有的样本最小个数，当节点样本个数小于该阈值时算法停止
        self.max_depth = max_depth  #树的最大深度
        self.is_gradient=is_gradient   #是否用于GBDT
        self.K=K  #用于GBDT分类时的种类
    '''
    @abstractmethod
    def __init__(self,
                 criterion,
                 splitter,
                 max_depth,
                 min_samples_split,
                 min_samples_leaf,
                 min_weight_fraction_leaf,
                 max_features,
                 max_leaf_nodes,
                 random_state,
                 min_impurity_decrease,
                 min_impurity_split,
                 class_weight=None,
                 presort=False):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.class_weight = class_weight
        self.presort = presort
'''
    @staticmethod
    def entropy(data):
        '''
        输入数据data,输出其经验熵'''
        n=len(data)   #数据个数
        label_dict={}
        for i in range(n):
            label_dict[data[i][-1]]=label_dict.get(data[i][-1],0)+1
        k=len(label_dict)  #类别个数
        ent=0
        for n_k in label_dict.values():
            ent+= n_k/n * math.log(n_k/n,2)
        return -ent
    
    @staticmethod
    def conditional_entropy(data,a):
        '''
        输入数据data和用来分类的特征a(即数据的第a列),输出条件熵'''
        n=len(data)   #数据个数
        con_ent=0
        new_data=BaseDecisionTree.data_divide(data,a)
        for curr_data in new_data:
            con_ent+= len(curr_data)/n * BaseDecisionTree.entropy(curr_data)        
        return con_ent
    
    @staticmethod
    def gini(data,a=None,value=None):
        n=len(data)  
        if not a:
            label_dict={} 
            for i in range(n):
                label_dict[data[i][-1]]=label_dict.get(data[i][-1],0)+1
            return 1-sum((x/n)**2 for x in label_dict.values())
        else:
            new_data=BaseDecisionTree.data_divide(data,a,value=value)
            return len(new_data[0])/n*BaseDecisionTree.gini(new_data[0]) + len(new_data[1])/n*BaseDecisionTree.gini(new_data[1])
            
    @staticmethod
    def data_divide(data,a,data_index=None,value=None):
        '''
        根据第a列特征将数据划分
        如果输入特征a的某个value，将数据集按a=value和a≠value划分成两个
        如果没有输入某个value，将数据集按a所有特征划分'''

        if not value:
            new_data={}
            i=0
            for curr_data in data:
                if data_index==None:next_data=curr_data
                else:next_data=(data_index[i],curr_data)
                new_data[curr_data[a]]=new_data.get(curr_data[a],[])
                new_data[curr_data[a]].append(next_data)
                i+=1
            return list(new_data.values())
        else:
            new_data=[[],[]]
            i=0
            for curr_data in data:
                if data_index==None:next_data=curr_data
                else:next_data=(data_index[i],curr_data)
                if curr_data[a]==value:
                    new_data[0].append(next_data)
                else:
                    new_data[1].append(next_data)
                i+=1
            return new_data
    
    @staticmethod
    def most_class(data):
        '''
        返回数据集中实例数最多的类'''
        n=len(data)   #数据个数
        label_dict={}
        for i in range(n):
            label_dict[data[i][-1]]=label_dict.get(data[i][-1],0)+1
        m=0
        for key in label_dict.keys():
            if label_dict[key]>m:
                m=label_dict[key]
                res=key
        return res
    
    def predict(self,data):
        pre=[]
        for curr_data in data:
            curr_node=self.root
            while curr_node.tree:
                curr_node=curr_node.tree[curr_data[curr_node.feature]]
            pre.append(curr_node.label)
        return pre
    
class ID3(BaseDecisionTree):
    '''
    ID3算法
    '''

    def fit(self,data):
        def dfs(new_data,feature_list,depth,data_index):  #递归创建树
            if len(new_data)<self.min_samples_leaf or depth>=self.max_depth:  #当前节点样本个数小于阈值，停止
                new_node=DecisionTreeNode()
                new_node.label=ID3.most_class(new_data)
                new_node.data_index=data_index
                return new_node

            best_feature_index,information_gain=self.chooseBestFeature(new_data)  #选取最优的特征
            best_feature=feature_list[best_feature_index]

            if information_gain<self.epsilon:   #当信息增益小于阈值epsilon，停止
                new_node=DecisionTreeNode()
                new_node.label=ID3.most_class(new_data)
                new_node.data_index=data_index
                return new_node
            
            new_node=DecisionTreeNode()
            new_node.feature=best_feature
            new_node.label=ID3.most_class(new_data)
            new_node.data_index=data_index
            
            next_data_with_index_list=ID3.data_divide(new_data,best_feature,data_index=data_index)  #用最优的特征划分当前数据集
            
            for next_data_with_index in next_data_with_index_list:  #对划分后的每个新数据集递归创建树
                next_data=[x[1] for x in next_data_with_index]
                next_data_index=[x[0] for x in next_data_with_index]
                #print( next_data)
                feature_value=next_data[0][best_feature_index]  #最优特征在当前数据集中的取值
                if len(feature_list)>1:
                    new_node.tree[feature_value]=dfs(
                        [x[:best_feature_index]+x[best_feature_index+1:] for x in next_data],
                        feature_list[:best_feature_index]+feature_list[best_feature_index+1:],
                        depth+1,next_data_index)
                else:new_node.tree[feature_value]=DecisionTreeNode(label=ID3.most_class(next_data))

            return new_node
        data_index=list(range(len(data)))
        self.root=dfs(data,list(range(len(data[0])-1)),1,data_index)
        
        return self.root
    
    def chooseBestFeature(self,data):#选取最优的特征
        
        ent=ID3.entropy(data)  #数据集的经验熵
        n_features=len(data[0])-1  #特征个数
        information_gain_list=[]  #每个特征对数据集的信息增益
        for i in range(n_features):
            information_gain_list.append(ent-ID3.conditional_entropy(data,i))
        
        #获取最大的信息增益对应的特征索引及信息增益值
        min_index, min_number = max(enumerate(information_gain_list), key=operator.itemgetter(1))  
        return min_index, min_number
    
class C45(ID3):
    '''
    C4.5算法
    '''
    def chooseBestFeature(self,data):#选取最优的特征
        ent=ID3.entropy(data)  #数据集的经验熵
        n_features=len(data[0])-1  #特征个数
        n=len(data)  #数据个数
        information_gain_ratio_list=[]  #每个特征对数据集的信息增益比
        for i in range(n_features):
            
            split_data=ID3.data_divide(data,i)  #按当前特征划分数据集
            h=-sum([len(x)/n * math.log(len(x)/n,2) for x in split_data])  #数据集关于当前特征的值的熵
            
            information_gain_ratio_list.append((ent-ID3.conditional_entropy(data,i))/h)
        
        #获取最大的信息增益比对应的特征索引及信息增益比的值
        min_index, min_number = max(enumerate(information_gain_ratio_list), key=operator.itemgetter(1))  
        return min_index, min_number

class CARTClassifier(ID3):
    '''
    CART分类算法
    '''
    def fit(self,data):
        def dfs(new_data,depth,data_index):  #递归创建树
            #当前节点样本个数小于阈值 or 数据集的Gini指数小于阈值 or 树深度大于max_depth时，停止
            if len(new_data)<self.min_samples_leaf or ID3.gini(new_data)<self.epsilon or depth>=self.max_depth:  
                new_node=DecisionTreeNode()
                new_node.label=ID3.most_class(new_data)
                new_node.data_index=data_index
                return new_node
            
            best_feature,best_value,min_gini=self.chooseBestFeature(new_data)  #选取最优的特征

            new_node=DecisionTreeNode()
            new_node.feature=best_feature
            new_node.label=ID3.most_class(new_data)
            new_node.data_index=data_index

            #用最优特征及特征值划分当前数据集
            next_data_with_index_list=ID3.data_divide(new_data,best_feature,data_index=data_index,value=best_value)
            
            next_data_index0,next_data0=[x[0] for x in next_data_with_index_list[0]],[x[1] for x in next_data_with_index_list[0]]
            next_data_index1,next_data1=[x[0] for x in next_data_with_index_list[1]],[x[1] for x in next_data_with_index_list[1]]
            new_node.tree["="+best_value]=dfs(next_data0,depth+1,next_data_index0)
            new_node.tree["≠"+best_value]=dfs(next_data1,depth+1,next_data_index1)
            return new_node
        
        data_index=list(range(len(data)))
        self.root=dfs(data,1,data_index)
        
        return self.root
    
    def chooseBestFeature(self,data):#选取最优的特征及特征值
        
        ent=ID3.entropy(data)  #数据集的经验熵
        n_features=len(data[0])-1  #特征个数
        n=len(data)  #数据个数
        min_gini,best_feature,best_value=float("Inf"),None,None  #每个特征对数据集的信息增益比
        for i in range(n_features):
            values=list(set([x[i] for x in data]))
            for value in values:
                curr_gini=ID3.gini(data,i,value)
                if curr_gini<min_gini:
                    min_gini=curr_gini
                    best_feature,best_value=i,value
        return best_feature,best_value,min_gini
    
    def predict(self,data):
        pre=[]
        for curr_data in data:
            curr_node=self.root
            while curr_node.tree:
                if "="+curr_data[curr_node.feature] in curr_node.tree:curr_node=list(curr_node.tree.values())[0]
                else:curr_node=list(curr_node.tree.values())[1]
            pre.append(curr_node.label)
        return pre
    
class CARTRegressor(BaseDecisionTree):
    '''
    CART回归算法
    '''        
    def fit(self,data,sample_weight=None):
        #sample_weight用于adaboost
        def dfs(new_data,depth,data_index):  #递归创建树
            #当前节点样本个数小于阈值 or 数据集的MSE小于阈值 or 树深度大于max_depth时，停止
            if len(new_data)<self.min_samples_leaf or self.cal_mse(new_data)[1]<self.epsilon or depth>=self.max_depth:  
                new_node=DecisionTreeNode()
                if not self.is_gradient:new_node.label=self.cal_mse(new_data)[0]
                else:new_node.label=self.cal_gamma([x[-1] for x in new_data],self.K)  #如果用于GBDT的话，将y值取出计算最佳负梯度拟合值
                new_node.data_index=data_index
                return new_node
            
            #选取最优的特征
            best_feature,best_split_point,min_mse,next_data_with_index_list=self.chooseBestFeature(new_data,data_index,sample_weight)  

            new_node=DecisionTreeNode()
            new_node.feature=best_feature
            new_node.split_point=best_split_point
            if not self.is_gradient:new_node.label=self.cal_mse(new_data)[0]
            else:new_node.label=self.cal_gamma([x[-1] for x in new_data],self.K) 
            new_node.data_index=data_index

            next_data_index0,next_data0=[x[0] for x in next_data_with_index_list[0]],[x[1] for x in next_data_with_index_list[0]]
            next_data_index1,next_data1=[x[0] for x in next_data_with_index_list[1]],[x[1] for x in next_data_with_index_list[1]]
            new_node.tree["<="+str(best_split_point)]=dfs(next_data0,depth+1,next_data_index0)
            new_node.tree[">"+str(best_split_point)]=dfs(next_data1,depth+1,next_data_index1)
            return new_node
        
        data_index=list(range(len(data)))
        self.root=dfs(data,1,data_index)
        
        return self.root
    
    def chooseBestFeature(self,data,data_index,sample_weight):#选取最优的特征及特征值
        
        ent=ID3.entropy(data)  #数据集的经验熵
        n_features=len(data[0])-1  #特征个数
        n=len(data)  #数据个数
        min_mse,best_feature,best_split_point,best_new_data_with_index_list=float("Inf"),None,None,None  #每个特征对数据集的信息增益比
        for i in range(n_features):   #遍历特征
            values=sorted(set([x[i] for x in data]))
            split_points=[(values[i]+values[i+1])/2 for i in range(len(values)-1)]
            for split_point in split_points:  #对特征i扫描切分点
                new_data_with_index_list=self.data_split(data,i,split_point,data_index)
                new_data_index0,new_data0=[x[0] for x in new_data_with_index_list[0]],[x[1] for x in new_data_with_index_list[0]]
                new_data_index1,new_data1=[x[0] for x in new_data_with_index_list[1]],[x[1] for x in new_data_with_index_list[1]]
                
                c1,mse1=self.cal_mse(new_data0,sample_weight)
                c2,mse2=self.cal_mse(new_data1,sample_weight)
                
                mse=mse1+mse2
                if mse<min_mse:
                    min_mse,best_feature,best_split_point,best_new_data_with_index_list=mse,i,split_point,new_data_with_index_list
                
        return best_feature,best_split_point,min_mse,best_new_data_with_index_list
    
    def data_split(self,data,a,split_point,data_index):
        new_data=[[],[]]
        i=0
        for curr_data in data:
            if curr_data[a]<=split_point:new_data[0].append((data_index[i],curr_data))
            else:new_data[1].append((data_index[i],curr_data))
            i+=1
        return new_data
    
    def cal_mse(self,data,sample_weight):
        c=sum([x[-1] for x in data])/len(data)
        if sample_weight==None:mse=sum([(x[-1]-c)**2 for x in data])
        else :mse=sum([y*(x[-1]-c)**2 for (x,y) in zip(data,sample_weight)])
        return c,mse
    
    def cal_gamma(self,data,K):
        #用于GBDT时，求叶子节点的最佳负梯度拟合值
        temp1=sum(data)
        temp2=sum([abs(x)*(1-abs(x)) for x in data])
        gamma=(K-1)*temp1/(K*temp2)
        return gamma
    
    def predict(self,data):
        pre=[]
        for curr_data in data:
            curr_node=self.root
            while curr_node.tree:
                if curr_data[curr_node.feature]<=curr_node.split_point:curr_node=list(curr_node.tree.values())[0]
                else:curr_node=list(curr_node.tree.values())[1]
            pre.append(curr_node.label)
        return pre


if __name__=='__main__':
    datasets = [['青年', '否', '否', '一般', '否否'],
               ['青年', '否', '否', '好', '否否'],
               ['青年', '是', '否', '好', '是是'],
               ['青年', '是', '是', '一般', '是是'],
               ['青年', '否', '否', '一般', '否否'],
               ['中年', '否', '否', '一般', '否否'],
               ['中年', '否', '否', '好', '否否'],
               ['中年', '是', '是', '好', '是是'],
               ['中年', '否', '是', '非常好', '是是'],
               ['中年', '否', '是', '非常好', '是是'],
               ['老年', '否', '是', '非常好', '是是'],
               ['老年', '否', '是', '好', '是是'],
               ['老年', '是', '否', '好', '是是'],
               ['老年', '是', '否', '非常好', '是是'],
               ['老年', '否', '否', '一般', '否否'],
               ]
    labels = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']
    print('================================C45分类结果================================')
    tree=C45(min_samples_leaf=1,max_depth=5)
    tree.fit(datasets)

    print(tree.root.display(labels))

    tree.predict([['老年', '否', '否', '一般']])
    print(tree.root.print_leaf_node())

    print('================================ID3分类结果================================')
    tree=ID3()
    tree.fit(datasets)

    print(tree.root.display(labels))

    tree.predict([['老年', '否', '否', '一般']])
    print(tree.root.print_leaf_node())

    print('================================CART分类结果================================')
    tree=CARTClassifier()
    tree.fit(datasets)

    print(tree.root.display(labels))

    tree.predict([['老年', '否', '否', '一般']])
    print(tree.root.print_leaf_node())


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
    print('================================CART回归结果================================')
    labels = ['维度一']
    tree=CARTRegressor(max_depth=2)
    tree.fit(datasets)

    print(tree.root.display(labels))

    tree.predict([[1.8]])
    tree.root.print_leaf_node()