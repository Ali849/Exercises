import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



data=pd.read_csv('training_data.txt',header=None)
print(data)

#--------------------------------------VARIABLE--------------------------------

x=(data.values[:,0:-1])
y=data.values[:,-1:]


X=np.concatenate(((np.ones((np.shape(x)[0],1))),x),axis=1)



m,n= (np.shape(X))


initial_theta=np.zeros((n))


###############################################################################
#-----------------------------------PLOTING------------------------------------
###############################################################################

def viewdata(data):
    positive_value= data.loc[data[2] == 1]
    negative_value=data.loc[data[2] ==0]
    
    plt.figure(figsize=(7,5))
    plt.plot(positive_value.values[:,0],positive_value.values[:,1],'k+',color='b',label='positive',)
    plt.plot(negative_value.values[:,0],negative_value.values[:,1], 'ko',color='r',label='negative')
    plt.legend(loc='lower left')
    plt.xlabel('1 Exam Results')
    plt.ylabel('2 Exam Results')

def decision_boundary(theta,X,y):
    if (np.shape(X))[1] <=3:
        plot_x = np.array([np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2])
        
        plot_y = (-1. / theta[2]) * (theta[1] * plot_x + theta[0])

        plt.plot(plot_x, plot_y)
     


###############################################################################
#-----------------------------FUNCTIONS----------------------------------------
###############################################################################

def sigmoid(z):
    
    return (1/(1+(np.exp(-z))))
     

def cost_gradient(theta,X,y):
    m,n= (np.shape(X))
    receive_theta = np.array(theta)[np.newaxis]
    theta = np.transpose(receive_theta)
    
    i=sigmoid(np.dot(X,theta))

    J=(1/m) * np.sum(  ((-y) * np.log((i))) - ((1-y) * np.log(1-(i))))

    gradient_init=np.zeros((n,1))


    gradient=gradient_init + ((1/m) * (np.dot( (np.transpose(X)) , ((i)-y))))
    
    return J,gradient

###############################################################################
###############################################################################


z=(np.dot(X,initial_theta))

J,gradient= cost_gradient(initial_theta,X,y) #FOR INITIAL THETA



###############################################################################
#--------------------------------OPTIMIZATION----------------------------------
###############################################################################
import scipy.optimize as opt

result = opt.fmin_tnc(func=cost_gradient, x0=initial_theta, args=(X, y))

optimal_theta=result[0]

J,gradient=(cost_gradient(optimal_theta,X,y)) #FOR OPTIMAL THETA
###############################################################################
###############################################################################

viewdata(data)

decision_boundary(optimal_theta,X,y)



