
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as st
plt.rcParams['figure.figsize'] = (5.0, 5.0)



# In[1]:


class RidgeLinearRegression:
       def __init__(self,X_data,Y_data,l,l2,iterations): #initialize all parameters
           self.X=X_data
           self.Y=Y_data
           #weight and bias
           self.m=0
           self.c=0
           self.L=l #learning rate
           self.l2=l2 #regularization parameter
           self.iter=iterations #num of iterations
           self.n=float(len(self.X))  #size of data
       
       def cost(self,pred_y): #cost function
           cost=np.sum(np.square(self.Y-pred_y))/(2*self.n) + self.l2*np.sum(np.square(self.m))
           return(cost)
       
       def fit(self):
           self.history=np.zeros(self.iter)
           #updating values of m and c
           for i in range(self.iter):
               pred_y=self.m*self.X + self.c
               #print(pred_y)
               Dm= (-2/self.n)*(self.X*(self.Y-pred_y))+2*self.l2*self.m
               Dc= (-2/self.n)*(self.Y-pred_y)
               #update
               self.m=self.m-Dm*self.L
               self.c=self.c-Dc*self.L
               #cost is calculated for every iteration
               self.history[i]=self.cost(pred_y)
           self.mse=self.MSE(self.Y,pred_y)
               
               
               
      
       def MSE(self,pred_y,Y):
           errors=Y-pred_y #error is the difference between actual and predicted value
           mse=np.sum(np.square(errors))/self.n #mean of sum of square of erros
           return mse
       
       def results(self):
           fig=plt.figure(figsize=(14,14))
           a1=fig.add_subplot(211)

           plt.title('minimisation of errors across the iterations')
           a1.plot(self.history)


           #making predictions
           a2=fig.add_subplot(212)
           final_y=self.m*self.X +self.c 
           plt.scatter(self.X,self.Y)
           plt.title('regrssion line')
           a2.plot([min(self.X),max(self.X)],[min(final_y),max(final_y)],color='red') #plotting the red line 
           
           plt.show()
           
           print ('Mean Squared Error=',self.mse)



