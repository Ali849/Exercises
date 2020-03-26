import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize as opt

data=pd.read_csv('training_data.txt',header=None)
#print(data)

#print(data.describe())

#------------------------------VARIABLES---------------------------------------


x = data.values[:,0:2]
X=np.concatenate(((np.ones((np.shape(x)[0],1))),x),axis=1)
y = data.values[:,-1:]



###############################################################################
###############################################################################
def plot_data():
   positive_value=data.loc[data[2] == 1]
   negative_value=data.loc[data[2] == 0]
   plt.plot(positive_value.values[:,0],positive_value.values[:,1],'k+',color='blue',label='positive')
   plt.plot(negative_value.values[:,0],negative_value.values[:,1],'ko',color='red',label='neagative')
   plt.xlabel('Microscop 1 test')
   plt.ylabel('microscope 2 test')
   plt.legend(loc='upper right')

def mapFeaturePlot(u,v):
    degree=6
    out = np.ones(1)
    for i in range(1,degree+1):
        for j in range(i+1):
            terms= (u**(i-j) * v**j)
            out= np.hstack((out,terms))
    return out


def contour_plot(theta):
    u_vals = np.linspace(-1,1.5,50)
    v_vals= np.linspace(-1,1.5,50)
    z=np.zeros((len(u_vals),len(v_vals)))
    for i in range(len(u_vals)):
        for j in range(len(v_vals)):
            z[i,j] =mapFeaturePlot(u_vals[i],v_vals[j]) @ theta        
    plt.contour(u_vals,v_vals,z.T,0)
    
    
    
###############################################################################
###############################################################################   

def sigmoid(z):
    return 1 / ( 1 + np.exp(-z) )
   
def mapFeature(X,degree):
    
    x1 = X[:,1]
    x2 = X[:,2]
    out = np.ones(len(x1)).reshape(len(x1),1)
    for i in range(1,degree+1):
        for j in range(i+1):
            terms= (x1**(i-j) * x2**j).reshape(len(x1),1)
            out= np.hstack((out,terms))
    return out

def cost_gradient(theta,X,y):
    lambd=1
    [m,n]= (np.shape(X))
    
    theta = theta[:,np.newaxis]
    
    h=sigmoid(np.dot(X,theta))

    J=(1/m) * np.sum(  ((-y) * np.log((h))) -  ((1-y) * np.log(1-(h))) + ( (lambd/(2*m))*np.sum(np.square(theta[1:])))      )
    
    gradient_init=np.zeros((theta.shape[0],1))
    
    gradient=gradient_init + ((1/m) * ( (np.transpose(X)) @ ((h)-y)))+((lambd/m)*theta)
    
    
    gradient[0]= ((1/m) * ( (np.transpose(X)) @ ((h)-y)))[0]
    
    return J,gradient





X = mapFeature(X,6)
theta_reg = np.zeros( X.shape[1] )

cost,grad=cost_gradient(theta_reg, X, y)


result = opt.fmin_tnc(func=cost_gradient, x0=theta_reg, args=(X, y))
optimal_theta=result[0]

plt.figure(figsize=(10,7))
plot_data()
contour_plot(optimal_theta)


#reg_logistic_regression = opt.minimize( fun = cost_gradient, x0 = theta_reg, 
#                                   args = (X, y), jac = gradient, 
#                                   options = {'maxiter' : 400} )



