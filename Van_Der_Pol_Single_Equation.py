

import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np
import matplotlib.pyplot as plt


## Load numerical solution
import scipy.io
mat = scipy.io.loadmat('Van_der_pol_mu2.mat')
VanDerPolmu2 = mat['data2']


mat = scipy.io.loadmat('Van_der_pol_mu15.mat')
VanDerPolmu15 = mat['data']






NN=50
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.regressor = nn.Sequential(nn.Linear(1, NN),
                                       nn.Tanh(),
                                       nn.Linear(NN, NN),
                                       nn.Tanh(),
                                       nn.Linear(NN, NN),
                                       nn.Tanh(),
                                       nn.Linear(NN, NN),
                                       nn.Tanh(),
                                       nn.Linear(NN, 1))
    def forward(self, x):
        output = self.regressor(x)
        return output


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)

LEARNING_RATE = 1e-3
MU = 2; # For Van der Pol equation
TRAIN_LIM = 10
COL_RES = 1000
EPOCHS = 1000

#Boundary Conditions
# t_bc = np.array([[0]])
# x_bc = np.array([[2,0]])

t_bc = np.array([[0]])
x_bc = np.array([[1]])

# Points and weight boundary vs ODE Loss
col_points = int(TRAIN_LIM*COL_RES)
boundary_points = len(x_bc)

F_WEIGHT = 1 #Physics Weight
B_WEIGHT = 1/(1*(boundary_points+col_points)) #Boundary Weight


#Define Net, apply to device and init weights
net = Net()
net = net.to(device)
net.apply(init_weights)


#Define Criteria
criterion = torch.nn.MSELoss() # Mean squared error
optimizer = torch.optim.Adam(net.parameters(),lr = LEARNING_RATE)


#ODE Driven loss function
def f(t,mu,net):
    x = net(t)
    ## Van der Pol Equation
    x_t = torch.autograd.grad(x.sum(), t, create_graph=True)[0]
    x_tt = torch.autograd.grad(x_t.sum(), t, create_graph=True)[0]
    ode = x_tt-mu*(1-x**2)*x_t+x
    return ode

def lossCalc(mse_u,mse_f,bp,cp,f_weight,b_weight,epoch = -1,beta = 1,betaLim = 1):
    #Function to combine loss from boundary and ODE or PDE
    if epoch*beta > betaLim or epoch == -1:
        loss = (b_weight*mse_u)/bp + (f_weight*mse_f/cp)
        epochBeta = betaLim
    else:
        loss = (b_weight*mse_u)/bp + (f_weight*mse_f/cp)*epoch*beta
        epochBeta = epoch*beta
    
    return loss,epochBeta


for epoch in range(EPOCHS):
    optimizer.zero_grad() # to make the gradients zero
    
    # Loss based on boundary conditions
    pt_t_bc = Variable(torch.from_numpy(t_bc).float(), requires_grad=False).to(device)
    pt_x_bc = Variable(torch.from_numpy(x_bc).float(), requires_grad=False).to(device)
    
    net_bc_out = net(pt_t_bc) # output of u(x,t)
    mse_u = criterion(input = net_bc_out, target = pt_x_bc)






    # Loss based on PDE
    t_collocation = np.random.uniform(low=0.0, high=TRAIN_LIM, size=(col_points,1))
    all_zeros = np.zeros((col_points,1))
    
    
    pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True).to(device)
    pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)
    
    ode = f(pt_t_collocation,MU,net) # output of f(x,t)
    mse_f = criterion(input = ode, target = pt_all_zeros)
    
    # Combining the loss functions
    loss,epochBeta = lossCalc(mse_u,mse_f,boundary_points,col_points,F_WEIGHT,B_WEIGHT)
    
    loss.backward() 
    optimizer.step() 

    with torch.autograd.no_grad():
        if epoch%10 == 0:
            print('Epoch:',epoch,"Traning Loss:",loss.data,'epochBeta:',epochBeta)
            print('Boundary Loss:',(B_WEIGHT*mse_u)/boundary_points,'ODE Loss: ',(F_WEIGHT*mse_f/col_points))
        

## Plot of solution within trained bounds
n = 1000
T_test = torch.linspace(0,TRAIN_LIM,n,requires_grad=True).to(device)
T_test = T_test.reshape(n,1)

score = net(T_test) 

x1_plot = score.cpu().detach().numpy()

T_plot = torch.linspace(0,TRAIN_LIM,n,requires_grad=False)
T_plot = T_test.reshape(n,1)
T_plot = T_plot.cpu().detach().numpy()

ode_residual = f(T_test,MU,net)
ode_residual = ode_residual.cpu().detach().numpy()
plt.figure()
plt.scatter(T_plot,x1_plot)
plt.title('Network Solution of x1')
plt.xlabel('t')
plt.ylabel('X1')
plt.legend()


plt.figure()
plt.title('X2')
plt.plot(T_plot,x1_plot)
plt.plot(VanDerPolmu2[:,0],VanDerPolmu2[:,1])
