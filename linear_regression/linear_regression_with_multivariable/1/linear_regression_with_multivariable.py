import numpy as np
import matplotlib.pyplot as plt


data=open('training_data_multi.txt','r')
data_read=data.read()
#------------------------------VARIABLES--------------------------------------

training_data=(np.matrix(data_read))

x=(training_data[:,0:-1])
y=training_data[:,-1]


row=(np.shape(training_data))[0]
coloumn=(np.shape(training_data))[1]

theta=np.zeros((coloumn,1))
alpha=0.01
iterations=400
m=len(x)

mu=np.mean(x)
sigma=np.std(x)
###############################################################################
#------------------------------NORMAL EQUATION---------------------------------
###############################################################################


def normal_equation(theta,X,y):
    theta= np.linalg.pinv( (X.T)*X ) * X.T * y 
    return theta


###############################################################################
#-----------------------------------------------------------------------------
###############################################################################


#-----------------------------FUTURE NORMALIZE---------------------------------
def future_normalize(sigma,mu,x):
    
    return np.divide((x-mu),sigma) 
    

#-------------------------COMPUTE COST FUCTION---------------------------
def cost_function(theta,X,y,m):
    return (1/(2*m))* sum(np.square((X*theta)-y))
   
#----------------------GRADIENT DESCENT: FIND OPTIMAL THETA----------------

def gradient_descent(theta,X,y,m,alpha,iterations):
    count=np.ones(((len(theta)),1))
    theta=np.float64(theta)
    for i in range(0,iterations):
        for k in range(0,len(count)):
           count[k]=(sum(np.multiply(((X*theta)-(y)) , (X[:,k]))))
        
        
        theta-= ((alpha*(1/m)) * count)
    
    return theta

##############################################################################
x=future_normalize(sigma,mu,x)
X=np.concatenate(((np.ones((len(x),1))),x),axis=1)
     
J=cost_function(theta,X,y,m)
theta=gradient_descent(theta,X,y,m,alpha,iterations)
##############################################################################

value1=future_normalize(sigma,mu,np.matrix('1650,3'))
price= (np.matrix('1,%s'%(value1))) * theta

print(price)
