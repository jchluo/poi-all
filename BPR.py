#coding:utf8
import reader
import math
import random
import time
import numpy as np

def sig(x):
    return 1.0/(1+math.exp(-x))

class BPR(object):
    def __init__(self):
        self.debug=False
        #self.debug=True
        self.user_num=0
        self.item_num=0 
        self.sample_num=0 #sum number of samples 
        self.iter_num=1000000 #iteration num of train
        self.learning_rate=0.05
        self.user_reg=0.01
        self.item_pos_reg=0.01
        self.item_neg_reg=0.01
        self.item_bias_reg=0.01
        self.item_bias_flag=0
        self.factor_num=400
        self.alpha_dist=0.05
        self.decay_rate=0.5
        self.user_list={}
        self.item_list={}
        self.U={}
        self.V={}
        self.user_item={}
        self.user_item_pos={}
        self.user_item_test={}
        self.item_bias={}
        self.loss=0
        self.user_item_dist={}
        self.last_pre=0
        self.user_item_kde={}
        
        self.readData()
        self.initUV()
        

    def initUV(self):
        for user in self.user_list:
            self.U[user]=np.array([random.gauss(0,0.01) for k in range(self.factor_num)])
        
        for item in self.item_list:
            self.item_bias[item]=0
            self.V[item]=np.array([random.gauss(0,0.01) for k in range(self.factor_num)])
             
    
    def readData(self):
        time1=time.time()
        self.user_item_pos,self.user_item_test,self.user_item,self.user_list\
            ,self.item_list,self.sample_num=reader.readMain()
        self.user_num=len(self.user_list)
        self.item_num=len(self.item_list)
        self.user_item_dist=reader.readUserItemDist()
        self.user_item_kde=reader.readUserItemKde()
        time2=time.time()
        print "readData cost time: %.1f s"%(time2-time1)
        print "user num:",self.user_num
        print "item num:",self.item_num
    
    def sample(self):
        user=random.choice(self.user_list)
        item_pos=random.choice(self.user_item_pos[user])
        item_neg=random.choice(self.item_list)
        while item_neg in self.user_item_pos[user]:
            item_neg=random.choice(self.item_list)
        
        return user,item_pos,item_neg
    
    def update(self,user,item_pos,item_neg):
        diff=self.predict(user,item_pos)-self.predict(user,item_neg)
        self.loss+=-math.log(sig(diff))
        w=sig(-diff)
        
        if self.debug:
            #print self.U[user]
            pass
        
        self.U[user]-=self.learning_rate*(self.user_reg*self.U[user]\
                    +w*(self.V[item_neg]-self.V[item_pos]))
        self.V[item_pos]-=self.learning_rate*(self.item_pos_reg*self.V[item_pos]\
                    -w*self.U[user])
        self.V[item_neg]-=self.learning_rate*(self.item_neg_reg*self.V[item_neg]\
                    +w*self.U[user])

        if self.item_bias_flag:
            self.item_bias[item_pos]-=self.learning_rate*(self.item_bias_reg\
                *self.item_bias[item_pos]-w)
            self.item_bias[item_neg]-=self.learning_rate*(self.item_bias_reg\
                *self.item_bias[item_neg]+w)
        
        self.loss+=np.sum(self.user_reg*self.U[user]**2)
        self.loss+=np.sum(self.item_pos_reg*self.V[item_pos]**2)
        self.loss+=np.sum(self.item_neg_reg*self.V[item_neg]**2)
        self.loss+=self.item_bias_reg*self.item_bias[item_pos]**2
        self.loss+=self.item_bias_reg*self.item_bias[item_neg]**2
        
        if self.debug:
            #print self.U[user]
            pass
    
    def predict(self,user,item):
        sum=self.U[user].dot(self.V[item])+self.item_bias[item]\
            +self.alpha_dist*self.user_item_dist[user][item]
        return sum

    def predict_onlyKde(self,user,item):
        return self.user_item_kde[user][item]

    def predict_withoutDist(self,user,item):
        sum=self.U[user].dot(self.V[item])+self.item_bias[item]
        return sum

    def predict_withKde(self,user,item):
        sum=self.U[user].dot(self.V[item])+self.item_bias[item]
        return sum*self.user_item_kde[user][item]
        
    def estimate(self):
        return self.precision()
    
    def auc(self):
        auc=0
        for user in self.user_list:
            item_sco_list=[]
            for item in self.item_list:
                item_sco_list.append([item,self.predict(user,item)])
            item_sco_list.sort(key=lambda x:x[1],reverse=True)
            
            count=0
            n=self.item_num
            for i,(item,s) in enumerate(item_sco_list):
                if item in self.user_item_pos[user]:
                    count+=n-i
            N=len(self.user_item_pos[user])
            M=n-N
            auc+=(count-M*(M-1)*0.5)/(M*N)
        auc=auc/self.user_num    
        return auc
    
    def precision(self):
        pre=0
        count=0
        k=5
        valid_user_count=0
        for user in self.user_list:
            if not self.user_item_test.has_key(user):
                continue
            
            valid_user_count+=1
            item_sco_list=[]
            for item in self.item_list:
                item_sco_list.append([item,self.predict_onlyKde(user,item)])
            item_sco_list.sort(key=lambda x:x[1],reverse=True)
            if self.debug:
                if user%200==0:
                    print ""
                    print "user:%d"%user
            for i,(item,s) in enumerate(item_sco_list[:k]):
                if self.debug:
                    if user%200==0:
                        print item,
                if item in self.user_item_test[user]:
                    count+=1

        if self.debug:
            print "count: %d"%(count)
        pre=count*1.0/(5*valid_user_count)    
        return pre
    
    def train(self):
        time1=time.time()
        for i in range(self.iter_num):
            for j in range(100*self.user_num):
                user,item_pos,item_neg=self.sample()
                self.update(user,item_pos,item_neg)
            if j%1==0:
                pre=self.estimate()
                time2=time.time()
                cost=time2-time1
                time1=time2
                print "iter %d, pre: %.4f,loss: %.2f, cost time: %.1fs"%(i,pre,self.loss,cost) 
                if pre<self.last_pre:
                    self.learning_rate*=self.decay_rate

                self.pre=pre
                self.loss=0
        pass

if __name__=='__main__':
    
    model=BPR()
    model.train()
        
        
