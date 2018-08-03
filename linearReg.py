import numpy as np
import pickle
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
class linreg:
    def __init__(self,name,reg=False):
        self.name=name
        self.trained=False
        self.reg=reg
        
    def regularize(self,dataSet):
        if(self.reg==True):
            mean=np.mean(dataSet,axis=0)
            stdev=np.std(dataSet,axis=0)
        else:
            n=np.shape(dataSet)[1]
            mean=np.zeros((1,n))
            stdev=np.ones((1,n))
        dataSet=(dataSet-mean)/stdev
        return mean,stdev,dataSet
        
    def train(self,dataSet,label,theta,alpha=0.1,numIter=20):
        dataSet=np.array(dataSet)
        m=np.shape(dataSet)[0]
        label=np.array(label)
        theta=np.array(theta)
        if not(np.shape(dataSet)[0]==np.shape(label)[0]):
            raise Exception(f"No. of rows in dataSet ({np.shape(dataSet)[0]}) and No. of label ({np.shape(label)[0]}) should be the same")
        if not(np.shape(dataSet)[1]==(np.shape(theta)[0]-1)):
            raise Exception(f"No. of elements in theta ({np.shape(theta)[0]}) should be {np.shape(dataSet)[1]+1}\n")
        [mean,stdev,dataSet]=self.regularize(dataSet)
        print(mean,':',stdev)
        dataSet=np.append(np.ones((m,1)),dataSet,axis=1)
        self.trained=True
        m=np.shape(dataSet)[0]
        for i in range(1,numIter+1):
            cost=dataSet.dot(theta)-label
            delta=dataSet.T.dot(cost)/m
            theta=theta-alpha*delta
            print('cost at iteration ',i,':',np.sum(np.power(cost,2))/(2*m))
        f=open(self.name,'wb')
        pickle.dump(dataSet,f)
        pickle.dump(theta,f)
        pickle.dump(mean,f)
        pickle.dump(stdev,f)
        f.close
        return theta
    def find(self,inputVect):
        inputVect=np.array([inputVect])
        try:
            f=open(self.name,'rb')
        except:
            raise Exception(f"{self.name} dosen't exist")
        dataSet=pickle.load(f)
        theta=pickle.load(f).T
        mean=pickle.load(f)
        stdev=pickle.load(f)
        m=np.shape(dataSet)[0]
        if(self.trained==False):
            raise Exception("Dataset not trained")
        inputVect=np.append(1,(inputVect-mean)/stdev)
        print(mean,':',stdev,':',inputVect)
        return float(np.sum(inputVect*theta))
    def getRegValues():
        if(self.trained==False):
            raise Exception("Dataset not trained")
        dataSet=pickle.load(f)
        theta=pickle.load(f).T
        mean=pickle.load(f)
        stdev=pickle.load(f)
        return [dataSet,theta,mean,stdev]
        
