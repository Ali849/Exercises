import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from  scipy.io import loadmat
import scipy.optimize as opt
import matplotlib.image as mpimg
from random import *


data= loadmat('training_data2.mat')
data1=loadmat('training_data.mat')



X=data1['X']
y=data1['y']




theta1=data['Theta1']
theta2=data['Theta2']


###############################################################################
##############################------PLOT-------################################
def disp_image(X):
    
     m,n= X.shape
     
     fig, axis = plt.subplots(m,figsize=(3,3))


     axis.imshow(X.reshape(20,20,order="F"), cmap="Greys") 
     axis.axis("off")


          
          
###############################################################################
#############################-----FUNCTIONS------##############################

def sigmoid(z):
    return 1 / ( 1 + np.exp(-z) )



def predict(theta1,theta2,X):
    m,n= X.shape[0], X.shape[1]
    
    p= np.zeros((m,1))
    X=np.hstack( (np.ones((m,1))   , X) )
    
    
    a2=sigmoid(X @ theta1.T)
    a2=np.hstack((np.ones((a2.shape[0],1)) , a2  ))
    
    a3= sigmoid(a2 @ theta2.T)
    
    for i in np.arange(m):
        for_picture= a3[i,:]
        put= np.argmax(for_picture)
        p[i]=put
        
        
        
    
    return p+1
###############################################################################    



pred =  predict(theta1,theta2,X)


print(   "accuracy:" , np.mean(y==pred)    * 100     )




num=randrange(0,5001)
X_pred= (X[num,:])[np.newaxis,:]
disp_image(X_pred)
prd=predict(theta1,theta2,X_pred)
print('is it %s ?' %np.mod(prd[0,0],10)  )









