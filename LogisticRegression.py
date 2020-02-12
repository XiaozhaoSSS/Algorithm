import numpy as np
class LogisticRegression():
    def __init__(self,learning_rate=1e-3,maxIter=500,epsilon=1e-5):
        
        
        self.learning_rate = learning_rate
        self.maxIter = maxIter
        self.epsilon=epsilon
        
        self.weights = None
        self.k = None   #类别数
        
        
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def softmax(self,x,w,k):
        '''
        输出第k个类别的概率
        '''
        temp=np.exp(np.dot(x,w))
        p=temp[:,k] / np.sum(temp,axis=1)
        return p
        
    def fit(self,X,y):
        self.k=len(set(list(y)))
        X,y=np.array(X),np.array(y).reshape((-1,1))
        self.update_weights(X,y)
        
    def update_weights(self,X,y):
        '''
        梯度下降法更新参数
        '''
        m,n=np.shape(X)   #m-数据个数,n-特征个数
        X=np.hstack((X,np.ones((m,1))))   #因为参数b与w合在一起，所以对输入数据做扩充
        
        if self.k==2:  #二分类
            self.weights=np.ones((n+1,1))   #初始化weights
            for iters in range(self.maxIter):
                error=y-self.sigmoid(np.dot(X,self.weights))
                
                g=np.dot(X.T,error)  #梯度方向                
                g=np.where(np.abs(g)<=self.epsilon,0,g)  #将梯度更新小于阈值的置0
                if any(g)==0:break
                    
                self.weights += (self.learning_rate*g)
                
        else:   #多分类
            self.weights=np.ones((n+1,self.k))   #初始化weights
            for iters in range(self.maxIter):
                for curr_class in range(self.k):
                    p=self.softmax(X,self.weights,curr_class)
                    error=[1 if curr_y==curr_class else 0 for curr_y in y]-p
                    error=error.reshape((-1,1))
                    
                    g=np.sum(X*error,axis=0)  #梯度方向
                    g=np.where(np.abs(g)<=self.epsilon,0,g)  #将梯度更新小于阈值的置0
                    if any(g)==0:break
                    
                    self.weights[:,curr_class] += (self.learning_rate*g)

    def predict_proba(self,data):
        X=np.array(data)
        m=np.shape(X)[0]   #数据个数
        X=np.hstack((X,np.ones((m,1))))
        if self.k==2:
            pred_prob = self.sigmoid(np.dot(X,self.weights))
        else:
            pred_prob = []
            for i in range(self.k):
                pred_prob.append(self.softmax(X,self.weights,i))
            pred_prob=np.array(pred_prob).T
        return pred_prob
    
    def predict(self,data):
        pred_prob = self.predict_proba(data)
        if self.k==2:
            pred=[1 if x>=0.5 else 0 for x in pred_prob]
        else:
            pred=np.argmax(pred_prob,axis=1)
        return pred


if __name__=='__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    X, y = load_iris(return_X_y=True)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

    print("========================LR训练iris数据集========================")
    model=LogisticRegression()
    model.fit(X_train,y_train)
    print('预测结果：',model.predict(X_test))
    print('预测概率：',model.predict_proba(X_test))

    print("========================sklearn实现LR========================")
    from sklearn.linear_model import LogisticRegression as LR_sklearn
    model = LR_sklearn(random_state=0).fit(X_train, y_train)
    print('预测结果：',model.predict(X_test))
    print('预测概率：',model.predict_proba(X_test))

    X_train,X_test,y_train,y_test=train_test_split(X[:99,:],y[:99],test_size=0.2)

    print("========================LR训练iris数据集_二分类========================")
    model=LogisticRegression()
    model.fit(X_train,y_train)
    print('预测结果：',model.predict(X_test))
    print('预测概率：',model.predict_proba(X_test))

    print("========================sklearn实现LR_二分类========================")
    from sklearn.linear_model import LogisticRegression as LR_sklearn
    model = LR_sklearn(random_state=0).fit(X_train, y_train)
    print('预测结果：',model.predict(X_test))
    print('预测概率：',model.predict_proba(X_test))

