import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


data = open('training_data.txt', 'r')
data_read=data.read()
#------------------------------VARIABLES--------------------------------------
matrix=(np.matrix(data_read))

y=(matrix[:,1])
x=(matrix[:,0])
X=np.concatenate(((np.ones((len(x),1))),x),axis=1)
theta=np.matrix('0;0')
alpha=0.01
iterations=1500
m=len(x)

#--------------------------PLOTING DATA---------------------------------------
def data_plot(x,y,theta):
    plt.plot(x,y,'rx',label='training data')
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Popultion of City in 10,000s')
    plt.plot(X[:,1], X*theta, label='linear')
    plt.legend(loc='lower right')

#--------------------------CONTOUR PLOT----------------------------------   
def contour_plot():
     global theta0_vals 
     theta0_vals = np.linspace(-10, 10, 100)
     global theta1_vals
     theta1_vals = np.linspace(-1, 4, 100)

     J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))


     for i in range(0,len(theta0_vals)):
        for j in range(0,len(theta1_vals)):
           t=np.zeros((2,1))
           t[0]=theta0_vals[i]
           t[1]=theta1_vals[j]
           J_vals[i,j]=cost_function(t,X,y,m)


     global J_vals1
     J_vals1= np.transpose(J_vals)
     
     plt.contour(theta0_vals, theta1_vals, J_vals1, np.logspace(-2, 3, 20))
     plt.plot(theta[0], theta[1], 'rx')
     plt.ylabel('theta1')
     plt.xlabel('tehta0')
    
#-----------------------------surface 3d plot----------------------------

def surface_plot(theta0_vals,theta1_vals,J_vals1):
    ax = plt.axes(projection='3d')
    ax.plot_surface(theta0_vals, theta1_vals, J_vals1 ,cmap='viridis', edgecolor='none')
    plt.xlabel('theta0') 
    plt.ylabel('theta1')
     
     
#---------------------CALL IT FOR PREDICTION---------------------
def hypotes(theta,prediction):
    return ((prediction*theta)*10000)[0,0]

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

#-----------------------------------------------------------------------------

J= cost_function(theta,X,y,m)
theta=gradient_descent(theta,X,y,m,alpha,iterations)    

plt.figure(0)
data_plot(x,y,theta)


plt.figure(1)
contour_plot()

plt.figure(2)
surface_plot(theta0_vals,theta1_vals,J_vals1)

prediction=np.matrix('1,'+ str(float(input('enter city\'s papulation:  '))/10000))    

#--------------------------OUTPUT--------------------------------------------


print('your profit is $%.2f ' % (hypotes(theta,prediction)))
