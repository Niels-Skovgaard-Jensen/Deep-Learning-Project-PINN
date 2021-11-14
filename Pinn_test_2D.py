import torch
import torch.nn as nn
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.hidden_layer1 = nn.Linear(1,5)
#         self.hidden_layer2 = nn.Linear(5,5)
#         self.hidden_layer3 = nn.Linear(5,5)
#         self.hidden_layer4 = nn.Linear(5,5)
#         self.hidden_layer5 = nn.Linear(5,5)
#         self.output_layer = nn.Linear(5,1)

#     def forward(self,x):
#         inputs = x # combined two arrays of 1 columns each to one array of 2 columns
#         layer1_out = torch.sigmoid(self.hidden_layer1(inputs))
#         layer2_out = torch.sigmoid(self.hidden_layer2(layer1_out))
#         layer3_out = torch.sigmoid(self.hidden_layer3(layer2_out))
#         layer4_out = torch.sigmoid(self.hidden_layer4(layer3_out))
#         layer5_out = torch.sigmoid(self.hidden_layer5(layer4_out))
#         output = self.output_layer(layer5_out) ## For regression, no activation is used in output layer
#         return output

#     def predict(self, X):
#             X = torch.Tensor(X)
#             return self(X).detach().numpy().squeeze()
    
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-7
### (2) Model
net = nn.Sequential(
          nn.Linear(1,20),
          nn.Sigmoid(),
          nn.Linear(20,20),
          nn.LeakyReLU(),
          nn.Linear(20,200),
          nn.LeakyReLU(),
          nn.Linear(200,80),
          nn.LeakyReLU(),
          nn.Linear(80,20),
          nn.LeakyReLU(),
          nn.Linear(20,20),
          nn.Sigmoid(),
          nn.Linear(20,1),
        )
net = net.to(device)
net.apply(init_weights)
#mse_cost_function = torch.nn.MSELoss() # Mean squared error
mse_cost_function = torch.nn.MSELoss() # Mean squared error
optimizer = torch.optim.Adam(net.parameters(),lr = LEARNING_RATE,weight_decay= WEIGHT_DECAY)

## PDE as loss function. Thus would use the network which we call as u_theta
def f(x,net):
    u = net(x) # the dependent variable u is given by the network based on independent variables x,t
    ## Based on our f = du/dx - 2du/dt - u, we need du/dx and du/dt
    #u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    pde = u_xx-u_x+u
    return pde

def lossCalc(mse_u,mse_f,bp,cp,epoch = -1,beta = 0,betaLim = 1):
    if epoch*beta > betaLim or epoch == -1:
        loss = mse_u/boundary_points + (mse_f/col_points)
    else:
        loss = mse_u/boundary_points + (mse_f/col_points)*epoch*beta
    
    return loss,epoch*beta
    

## Create Data for boundary conditions

x_bc = np.array([[0],[1],[2]])
#t_bc = np.zeros((500,1))
# compute u based on BC
u_bc = np.array([[0],[3],[0]])

### (3) Training / Fitting
iterations = 20000
col_points = 100
boundary_points = len(u_bc)
#2.4087e-05
beta = 1e-4
for epoch in range(iterations):
    optimizer.zero_grad() # to make the gradients zero
    
    # Loss based on boundary conditions
    pt_x_bc = Variable(torch.from_numpy(x_bc).float(), requires_grad=False).to(device)
    #pt_t_bc = Variable(torch.from_numpy(t_bc).float(), requires_grad=False).to(device)
    pt_u_bc = Variable(torch.from_numpy(u_bc).float(), requires_grad=False).to(device)
    
    net_bc_out = net(pt_x_bc) # output of u(x,t)
    mse_u = mse_cost_function(net_bc_out, pt_u_bc)
    
    # Loss based on PDE
    x_collocation = np.random.uniform(low=0.0, high=2, size=(col_points,1))
    #t_collocation = np.random.uniform(low=0.0, high=1.0, size=(500,1))
    all_zeros = np.zeros((col_points,1))
    
    
    pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True).to(device)
    #pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True).to(device)
    pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)
    
    f_out = f(pt_x_collocation,net) # output of f(x,t)
    mse_f = mse_cost_function(f_out, pt_all_zeros)
    
    # Combining the loss functions
    #loss = mse_u/boundary_points + (mse_f/col_points)
    loss,epochBeta = lossCalc(mse_u,mse_f,boundary_points,col_points,epoch,beta)
    
    loss.backward() # This is for computing gradients using backward propagation
    optimizer.step() # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta

    with torch.autograd.no_grad():
        if epoch%100 == 0:
            print('Epoch:',epoch,"Traning Loss:",loss.data,'epochBeta:',epochBeta)
        


import matplotlib.pyplot as plt
import numpy as np

n = 1000
X_test = torch.linspace(0,2,n,requires_grad=True).to(device)
X_test = X_test.reshape(n,1)

#X_test2 = torch.linspace(0*np.pi,2*np.pi,n).to(device)
#X_test2 = X_test2.reshape(n,1)


score = net(X_test) 

#y_x = torch.autograd.grad(score.sum(),X_test)[0]
#y_x = y_x.cpu().detach().numpy()


y_plot = score.cpu().detach().numpy()
residual = f(X_test,net)
x_plot = X_test.cpu().detach().numpy()

#x_plot2 = X_test2.cpu().detach().numpy()

#y_plot_validation = np.exp(x_plot2)



#print(score)
plt.figure
limits = [0,3,0,2]

#plt.scatter(x_plot2,y_plot_validation, label='Sin(x)')
plt.scatter(x_plot,y_plot,label = 'Model Approximation')
plt.legend()
plt.figure()
plt.scatter(x_plot,residual.detach().numpy(),label = 'Residual of ODE')
plt.legend()
#plt.axis(limits)
#plt.scatter(x_plot,y_x,label ='Diff')


