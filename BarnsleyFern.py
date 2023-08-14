import torch
import numpy as np
import matplotlib.pyplot as plt

#centre line/ main stem
def f1(x, y):
    transformation_matrix = torch.tensor([[0.00, 0.00], [0.00, 0.16]], device=device)
    translation_vector = torch.tensor([[0.00], [0.00]], device=device)
    new_coordinates = torch.matmul(transformation_matrix, torch.tensor([[x], [y]], device=device)) + translation_vector #matrix multiplication
    
    #extract new_coordinate value and return as float
    #new_coordinates [0,0]indexing returns the value in the top-left corner of the matrix- extracting x
    #new_coordinates [0,1]- extracting y
    #using tensor to perform matrix multiplication with GPU (parallelism)
    return new_coordinates[0, 0], new_coordinates[1, 0]

#rightward lean of successive ferns
def f2(x, y):
    transformation_matrix = torch.tensor([[0.85, 0.04], [-0.04, 0.85]], device=device)
    translation_vector = torch.tensor([[0.00], [1.60]], device=device)
    new_coordinates = torch.matmul(transformation_matrix, torch.tensor([[x], [y]], device=device)) + translation_vector
    return new_coordinates[0, 0], new_coordinates[1, 0]

#left lean of leaves- first leaf on left
def f3(x, y):
    transformation_matrix = torch.tensor([[0.20, -0.26], [0.23, 0.22]], device=device)
    translation_vector = torch.tensor([[0.00], [1.60]], device=device)
    new_coordinates = torch.matmul(transformation_matrix, torch.tensor([[x], [y]], device=device)) + translation_vector
    return new_coordinates[0, 0], new_coordinates[1, 0]

#right lean of leaves- first leaf on right
def f4(x, y):
    transformation_matrix = torch.tensor([[-0.15, 0.28], [0.26, 0.24]], device=device)
    translation_vector = torch.tensor([[0.00], [0.44]], device=device)
    new_coordinates = torch.matmul(transformation_matrix, torch.tensor([[x], [y]], device=device)) + translation_vector
    return new_coordinates[0, 0], new_coordinates[1, 0]

functions = [f1, f2, f3, f4]

points = 100000 #no.iterations
x, y = 0.0, 0.0 #initial config./ starting point

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load into PyTorch tensors
x = torch.tensor(x, device=device) #applying device config. for subsequent operations w/ tensors
y = torch.tensor(y, device=device)

#store coordinates
x_list = []
y_list = []

for i in range(points):
    function = np.random.choice(functions, p=[0.01, 0.85, 0.07, 0.07]) #random selection of function, w/ correspondinf probabilities
    x, y = function(x, y) #apply chosen transformation to current (x,y)
    x_list.append(x) #update list
    y_list.append(y)

# Transfer x_list and y_list to GPU for plotting
x_plot = torch.tensor(x_list, device=device)
y_plot = torch.tensor(y_list, device=device)

plt.scatter(x_plot.cpu(), y_plot.cpu(), s=0.28, color='green')
plt.show()

#Iterative Function System (IFS)- self similar
#Simulating variant of Chaos Game by repeatedly applying a sequence of affine transformations to a point
#Each affine transformation is represented by a matrix multiplication followed by a translation- each function chosen probablistically

