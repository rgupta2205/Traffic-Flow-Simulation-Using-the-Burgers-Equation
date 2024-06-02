import io
from flask import Flask, render_template, request
import math
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
# %matplotlib inline
import math
from tkinter import *
from base64 import b64encode
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import base64
import io
import numpy                       #here we load numpy
from matplotlib import pyplot, cm
# import cairosvg
from io import BytesIO
Lx=2
Ly=6
nx = 41
ny = 41
nt = 200
c = 1
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
sigma = .01
nu = 1
dt = sigma * dx * dy / nu

v_max=1
rho_max=1
ratio=rho_max/v_max
rho0=0.2
u0=v_max*(1-rho0/rho_max)


x = numpy.linspace(0, Lx, nx)
y = numpy.linspace(0, Ly, ny)

u = u0*numpy.ones((ny, nx))  # create a 1xn vector of 1's
v = 0*numpy.ones((ny, nx))
un = numpy.ones((ny, nx)) 
vn = numpy.ones((ny, nx))
comb = numpy.ones((ny, nx))

rho_vector=np.arange(rho0, 1.1, 0.1)
u0_vector = v_max * (1 - rho_vector / rho_max)



num_squares = np.random.randint(30, size=1)[0]
square_size = 5
square_locations = np.random.randint(1, 40-square_size+1, size=(num_squares, 2))

for i in range(num_squares):
    random_index = np.random.randint(len(u0_vector))
    u[square_locations[i,0]:square_locations[i,0]+square_size-1, square_locations[i,1]:square_locations[i,1]+square_size-1] = u0_vector[random_index]
    print(u0_vector[random_index])

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = range(len(u))
y = range(len(u[0]))
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, u, cmap='viridis')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Velocity')

plt.title('Velocity Profile in X-Direction')
plt.show()

rho = np.ones(u.shape) - (ratio * u)

plt.figure()
plt.subplot(1, 3, 1)
plt.contourf(x, y, rho)
plt.xlabel('x-direction')
plt.ylabel('y-direction')
plt.title('Density profile')
plt.colorbar()

c = ['b', 'r', 'g', 'k','m']
cnt = 1
for i in range(2, 40, 10):
    plt.subplot(1, 3, 2)
    plt.plot(x, rho[i,:], c[cnt])
    plt.subplot(1, 3, 3)
    plt.plot(y, rho[:,i], c[cnt])
    # plt.subplot(1, 4, 4)
    # plt.axvline(x=x[i], color=c[cnt])
    # plt.axhline(y=y[i], color=c[cnt])
    cnt += 1

plt.subplot(1, 3, 2)
plt.plot(x, rho[2, :], 'b', label='1st')
plt.plot(x, rho[12, :], 'r', label='2nd')
plt.plot(x, rho[22, :], 'g', label='3rd')
plt.plot(x, rho[32, :], 'k', label='4th')
plt.legend()
plt.title('y-direction')
plt.ylabel('density')

plt.subplot(1, 3, 3)
plt.plot(y, rho[:, 2], 'b', label='1st')
plt.plot(y, rho[:, 12], 'r', label='2nd')
plt.plot(y, rho[:, 22], 'g', label='3rd')
plt.plot(y, rho[:, 32], 'k', label='4th')
plt.legend()
plt.title('x-direction')
plt.ylabel('density')

plt.show()


X, Y = np.meshgrid(x,y)
for n in range(nt+1):
    for i in range(1,ny-1):
        for j in range(1,nx-1):
            u[i,j] = u[i,j] - (dt/dx) * u[i,j] * (u[i,j]-u[i-1,j]) - (dt/dy) * v[i,j] * (u[i,j]-u[i,j-1]) + (nu*dt/dx**2) * (u[i-1,j] - 2*u[i,j] + u[i+1,j]) + (nu*dt/dy**2) * (u[i,j-1] - 2*u[i,j] + u[i,j+1])
            v[i,j] = v[i,j] - (dt/dx) * u[i,j] * (v[i,j]-v[i-1,j]) - (dt/dy) * v[i,j] * (v[i,j]-v[i,j-1]) + (nu*dt/dx**2) * (v[i-1,j] - 2*v[i,j] + v[i+1,j]) + (nu*dt/dy**2) * (v[i,j-1] - 2*v[i,j] + v[i,j+1])
            
            u[0:ny-1, 0] = 1
            u[0, 0:nx-1] = 1
            u[ny-1, 0:nx-1] = 1
            u[0:nx-1, ny-1] = 1
            
            v[0:ny-1, 0] = 1
            v[0, 0:nx-1] = 1
            v[ny-1, 0:nx-1] = 1
            v[0:nx-1, ny-1] = 1
            
rho = np.ones_like(u) - (ratio * u)

import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize=(12,4))

# Subplot 1
rho = np.ones(u.shape) - ratio*u
plt.subplot(1,3,1)
plt.contourf(x, y, rho)
plt.xlabel('x-direction')
plt.ylabel('y-direction')
plt.title('Density profile')
plt.colorbar()

# Subplot 2
c = ['b', 'r', 'g', 'k','m']
cnt = 1
for i in range(3, 41,10):
    plt.subplot(1,3,2)
    plt.plot(x, rho[i, :], c[cnt])
    plt.subplot(1,3,3)
    plt.plot(y, rho[:, i], c[cnt])
    # plt.subplot(1,4,4)
    # plt.axvline(x=x[i], linestyle='--', color=c[cnt])
    # plt.axhline(y=y[i], linestyle='--', color=c[cnt])
    cnt += 1

plt.subplot(1,3,2)
plt.legend(['1st', '2nd', '3rd', '4th'])
plt.title('y-direction')
plt.ylabel('density')

plt.subplot(1,3,3)
plt.legend(['1st', '2nd', '3rd', '4th'])
plt.title('x-direction')
plt.ylabel('density')

plt.show()

"""
# Create the squares by setting the appropriate elements of the space to density value
# for i in range(1,num_squares):
#     random_index = np.random(len(u0_vector))
#     u(square_locations(i,1):(square_locations(i,1)+square_size-1), square_locations(i,2):(square_locations(i,2)+square_size-1)) =  u0_vector(random_index);
#     u0_vector(random_index)
# end

###Assign initial conditions

##set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
u[int(.5 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1)] = 2 
##set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
v[int(.5 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1)] = 2


for n in range(nt + 1): ##loop across number of time steps
    un = u.copy()
    vn = v.copy()

    u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                dt / dx * un[1:-1, 1:-1] * 
                (un[1:-1, 1:-1] - un[1:-1, 0:-2]) - 
                dt / dy * vn[1:-1, 1:-1] * 
                (un[1:-1, 1:-1] - un[0:-2, 1:-1]) + 
                nu * dt / dx**2 * 
                (un[1:-1,2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) + 
                nu * dt / dy**2 * 
                (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1]))

    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] - 
                    dt / dx * un[1:-1, 1:-1] *
                    (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                    dt / dy * vn[1:-1, 1:-1] * 
                    (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) + 
                    nu * dt / dx**2 * 
                    (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                    nu * dt / dy**2 *
                    (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1]))

    u[0, :] = 5
    u[-1, :] = 5
    u[:, 0] = 3
    u[:, -1] = 3

    v[0, :] = 2
    v[-1, :] = 2
    v[:, 0] = 1
    v[:, -1] = 1

    fig2 = pyplot.figure(figsize=(20, 15), dpi=100)
    ax2 = fig2.gca(projection='3d')
    X, Y = numpy.meshgrid(x, y)
    ax2.plot_surface(X, Y, u, cmap=cm.viridis, rstride=1, cstride=1)
    ax2.plot_surface(X, Y, v, cmap=cm.viridis, rstride=1, cstride=1)
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$y$')
    fig1.show()
    fig2.savefig("abc1.png")


    file_name1="abc.png"
    file_name2="abc1.png"    
    # fig1.savefig(file_name1)
    # fig2.savefig(file_name2)


    # inherit figures' dimensions, partially
    h1, h2 = [int(np.ceil(fig.get_figheight())) for fig in (fig1, fig2)]
    wmax = int(np.ceil(max([fig.get_figwidth() for fig in (fig1, fig2)])))

    fig, axes = plt.subplots(h1 + h2, figsize=(wmax, h1 + h2))

    # make two axes of desired height proportion
    gs = axes[0].get_gridspec()
    for ax in axes.flat:
        ax.remove()
    ax1 = fig.add_subplot(gs[:h1])
    ax2 = fig.add_subplot(gs[h1:])

    ax1.imshow(plt.imread(file_name1))
    ax2.imshow(plt.imread(file_name2))

    for ax in (ax1, ax2):
        for side in ('top', 'left', 'bottom', 'right'):
            ax.spines[side].set_visible(False)
        ax.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)

    fig.savefig("abcd")

"""    