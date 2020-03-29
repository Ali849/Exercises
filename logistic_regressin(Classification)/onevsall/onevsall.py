import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from  scipy.io import loadmat
import scipy.optimize as opt
import matplotlib.image as mpimg
from random import *



#################################----VARIABLES----#############################
data= loadmat('training_data.mat')
#print(data)
X=data['X']
y=data['y'].ravel()

y[y==10]=0

num_labels = 10


###############################################################################
##############################------PLOT-------################################

def display_image(X):

     fig, axis = plt.subplots(10,10,figsize=(8,8))


     for i in range(10):
        for j in range(10):
          axis[i,j].imshow(X[randint(0,5001),:].reshape(20,20,order="F"), cmap="Greys") 
          axis[i,j].axis("off")

###############################################################################


def testing():
    theta_t=np.array([-2,-1,1,2])

    y_t=np.array([[1],[0],[1],[0],[1]])

    ones=np.ones((5,1))

    x_t=np.array([i/10 for i in range(1,16)]).reshape(5,3, order='F')

    X_t= np.hstack((ones, x_t))

    lambd=3

    J,grad=cost_gradient(theta_t,X_t,y_t,lambd)

    print(J)
    print(grad)
###############################################################################
#############################-----FUNCTIONS------##############################
    

def sigmoid(z):
    return 1 / ( 1 + np.exp(-z) ) 



def cost_gradient(theta,X,y,lambd):
    m = y.size
    

    h = sigmoid(X @ (theta.T))
    

    theta[0] = 0
    
    J = (1 / m) * np.sum(  (-y) @ (np.log(h)) - (1 - y) @ (np.log(1 - h))) + (lambd / (2 * m)) * np.sum(np.square(theta))
    
    grad = (1 / m) * ( X.T  @ (h - y) ) 
    grad = grad + (lambd / m) * theta

        
    return J, grad
       
       
def onevsall(X,y,num_labels):
      m,n = X.shape
      all_theta= np.zeros((num_labels,n+1))
      X= np.hstack(  (np.ones((m,1))   ,  X) )
      initial_theta=np.zeros( X.shape[1] )
      
      for i in range(num_labels):
          
          result=opt.minimize(cost_gradient, 
                                initial_theta, 
                                (X, (np.where(y ==i,1,0)), 3), 
                                jac=True, 
                                method='CG')
          
          new_row = (result.x)[np.newaxis,:]
          
          all_theta[i,:]=new_row
          
      return all_theta

def  predict_onevsall(all_theta,X):
     m,n = X.shape
     p=np.zeros((m,1))
     
     X= np.hstack(  (np.ones((m,1)) ,  X) )
     
     p = np.argmax(sigmoid(X @ (all_theta.T)), axis = 1)
         
         
     
     return p    
     
     
     
     
     

###############################################################################
 
all_theta= onevsall(X,y,num_labels)

pred =  predict_onevsall(all_theta,X)

print(   "accuracy:" , np.mean((pred==y))    * 100     )









        