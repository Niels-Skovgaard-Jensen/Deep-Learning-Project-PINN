

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
NN=100
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
                                       nn.Linear(NN, NN),
                                       nn.Tanh(),
                                       nn.Linear(NN, NN),
                                       nn.Tanh(),
                                       nn.Linear(NN, 2))
    def forward(self, x):
        output = self.regressor(x)
        return output


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)

LEARNING_RATE = 5e-3
EPOCHS = 1000
MU = 2
TRAIN_LIM = 50
col_points = 3000

#Boundary Conditions
# t_bc = np.array([[0]])
# x_bc = np.array([[2,0]])
data = VanDerPolmu2 ##TO CHOOSE DATA FOR

t_bc = data[:,0]
t_bc = t_bc[..., None] 
x_bc = data[:,1:3]

#DataPoints

point_count = 100
boundary_points = np.random.randint(0,t_bc.size, size=point_count)
t_bc = t_bc[boundary_points, :]
x_bc = x_bc[boundary_points,:]

def f(t,mu,net):
    x = net(t) 
    x1 = x[:,0]
    x2 = x[:,1]
    ## Van der Pol Equation
    x1_t = torch.autograd.grad(x1.sum(), t, create_graph=True)[0]
    x2_t = torch.autograd.grad(x2.sum(), t, create_graph=True)[0]
    ode1 = x1_t-x2
    ode2 = mu*(1-x1**2)*x2-x1-x2_t
    return ode1,ode2

#Define Net, apply to device and init weights
net = Net()
net = net.to(device)
net.apply(init_weights)

#Define Criteria
criterion = torch.nn.MSELoss() # Mean squared error
optimizer = torch.optim.Adam(net.parameters(),lr = LEARNING_RATE)

mse_d_list = []
mse_f_list = []
loss_list = []

for epoch in range(EPOCHS):
    optimizer.zero_grad() # to make the gradients zero
    
    # Loss based on boundary conditions
    pt_t_bc = Variable(torch.from_numpy(t_bc).float(), requires_grad=False).to(device)
    pt_x_bc = Variable(torch.from_numpy(x_bc).float(), requires_grad=False).to(device)
    
    net_bc_out = net(pt_t_bc) # output of u(x,t)
    mse_d = criterion(input = net_bc_out, target = pt_x_bc)

    #Physics Loss based on ODE
    t_collocation = np.random.uniform(low=0.0, high=TRAIN_LIM, size=(col_points,1))
    all_zeros = np.zeros((col_points,1))
    
    
    pt_t_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True).to(device)
    pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)
    
    ode1,ode2 = f(pt_t_collocation,MU,net) # output of f(x,t)
    
    mse_f = criterion(input = ode1, target = pt_all_zeros)+criterion(input = ode2, target = pt_all_zeros)
    
    loss = mse_d+0*mse_f
    
    mse_d_list.append(mse_d)
    mse_f_list.append(mse_d)
    loss_list.append(loss)
    loss.backward() 
    optimizer.step() 

    with torch.autograd.no_grad():
        if epoch%10 == 0:
            print('Epoch:',epoch,"Total Loss:",loss.data,'ODE Loss',mse_f.data,'Data Loss:',mse_d.data)
        

## Plot of solution within trained bounds

t_plot = torch.linspace(data[0,0],data[-1,0],steps = 1000)
t_plot = t_plot[..., None] 
score = net(t_plot) 

x1_plot = score[:,0].cpu().detach().numpy()
x2_plot = score[:,1].cpu().detach().numpy()
t_bc_plot = t_bc
x1_bc = x_bc[:,0]
x2_bc = x_bc[:,1]



plt.figure()
plt.title('Net X1')
plt.plot(t_plot,x1_plot)
plt.figure()
plt.title('Net X2')
plt.plot(t_plot,x2_plot)

plt.figure()
plt.title('X1')
plt.plot(data[:,0],data[:,1],label = 'Numerical')
plt.plot(t_plot,x1_plot,label = 'Net')
plt.scatter(t_bc,x1_bc)
plt.legend()

plt.figure()
plt.title('X2')
plt.plot(data[:,0],data[:,2],label = 'Numerical')
plt.plot(t_plot,x2_plot,label = 'Net')
plt.scatter(t_bc,x2_bc,marker = 'X',c='k')
plt.legend()

plt.figure()

mse_d_list_plot = [x.item() for x in mse_d_list]
mse_f_list_plot = [x.item() for x in mse_f_list]
loss_plot = [x.item() for x in loss_list]

plt.plot(mse_d_list_plot,label = 'Boundary Loss')
plt.plot(mse_f_list_plot,label  = 'Physics loss')
plt.plot(loss_plot,label = 'Total Loss')
plt.legend()

# plt.figure()
# plt.title('X1')
# plt.plot(VanDerPolmu15[:,0],VanDerPolmu15[:,1])
# plt.figure()
# plt.title('X2')
# plt.plot(VanDerPolmu15[:,0],VanDerPolmu15[:,2])


