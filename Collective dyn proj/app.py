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


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/indexa.html', methods=['GET', 'POST'])
def indexa():
    if request.method == 'POST':
        
        file = request.form['upload-file']
        file1 = request.form['upload-filea']
        file2 = request.form['upload-fileb']
        file4 = request.form['upload-filed']
        print(file)
        print(file1)
        print(file2)
        print(file4)
        nx=int(file)
        nt=int(file1)
        c=int(file2)
        eq=file4

        def linear_Convection1D(nx1,nt1,c1):
            nx = nx1  # try changing this number from 41 to 81 and Run All ... what happens?
            dx = 2 / (nx-1)
            nt = nt1    #nt is the number of timesteps we want to calculate
            dt = .025  #dt is the amount of time each timestep covers (delta t)
            c = c1      #assume wavespeed of c = 1
            u = numpy.ones(nx)      #numpy function ones()
            u[int(.5 / dx):int(1 / dx + 1)] = 2  #setting u = 2 between 0.5 and 1 as per our I.C.s
            print(u)
            fig, ax = plt.subplots(figsize=(16,8))
            ax.plot(numpy.linspace(0, 2, nx), u)
            ax.set_title("Plot")
            fig.savefig("abcd.png")
            print("1-D Linear Convection")
            
            
        def nonlinear_Convection(nx1,nt1,c1):
            nx = nx1
            dx = 2 / (nx - 1)
            nt = nt1    #nt is the number of timesteps we want to calculate
            dt = .025  #dt is the amount of time each timestep covers (delta t)
            u = numpy.ones(nx)      #as before, we initialize u with every value equal to 1.
            u[int(.5 / dx) : int(1 / dx + 1)] = 2  #then set u = 2 between 0.5 and 1 as per our I.C.s
            c = c1      #assume wavespeed of c = 1
            un = numpy.ones(nx) #initialize our placeholder array un, to hold the time-stepped solution
            for n in range(nt):  #iterate through time
                un = u.copy() ##copy the existing values of u into un
                for i in range(1, nx):  ##now we'll iterate through the u array                
                ###This is the line from Step 1, copied exactly.  Edit it for our new equation.
                ###then uncomment it and run the cell to evaluate Step 2                
                    u[i] = un[i] - c * dt / dx * (un[i] - un[i-1])
            fig, ax = plt.subplots(figsize=(16,8))
            ax.plot(numpy.linspace(0, 2, nx), u)
            ax.set_title("Plot")
            fig.savefig("abcd.png")
            # print("1-D Linear Convection")            
                      

        def cFL_Condition(nx1,nt1,c1):            
            nx = nx1
            dx = 2 / (nx - 1)
            nt = nt1    #nt is the number of timesteps we want to calculate
            c = c1
            sigma = .5            
            dt = sigma * dx
            u = numpy.ones(nx) 
            u[int(.5/dx):int(1 / dx + 1)] = 2
            un = numpy.ones(nx)
            for n in range(nt):  #iterate through time
                un = u.copy() ##copy the existing values of u into un
                for i in range(1, nx):
                    u[i] = un[i] - c * dt / dx * (un[i] - un[i-1])  
            fig, ax = plt.subplots(figsize=(16,8))
            ax.plot(numpy.linspace(0, 2, nx), u)
            ax.set_title("Plot")
            fig.savefig("abcd.png")
            print("1-D Linear Convection")
                    
            
            
            
        def burgers_Equation_in_2D(nx1,nt1,c1):
            nx = nx1
            ny = 41
            nt = nt1
            c = c1
            dx = 2 / (nx - 1)
            dy = 2 / (ny - 1)
            sigma = .0009
            nu = 0.01
            dt = sigma * dx * dy / nu


            x = numpy.linspace(0, 2, nx)
            y = numpy.linspace(0, 2, ny)

            u = numpy.ones((ny, nx))  # create a 1xn vector of 1's
            v = numpy.ones((ny, nx))
            un = numpy.ones((ny, nx)) 
            vn = numpy.ones((ny, nx))
            comb = numpy.ones((ny, nx))

            ###Assign initial conditions

            ##set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
            u[int(1 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1)] = 2 
            ##set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
            v[int(1 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1)] = 2
            
            ###(plot ICs)
            fig1 = pyplot.figure(figsize=(20, 15), dpi=100)
            ax1 = fig1.gca(projection='3d')
            X, Y = numpy.meshgrid(x, y)
            ax1.plot_surface(X, Y, u[:], cmap=cm.viridis, rstride=1, cstride=1)
            ax1.plot_surface(X, Y, v[:], cmap=cm.viridis, rstride=1, cstride=1)
            ax1.set_xlabel('$x$')
            ax1.set_ylabel('$y$');
            fig1.savefig("abc.png")
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



        if(eq=='a'):
            linear_Convection1D(nx,nt,c)

        elif(eq=='b'):
            nonlinear_Convection(nx,nt,c)
            # encoded = fig_to_base64(img)    
            # myhtml='<img src="data:image/png;base64, {}">'.format(encoded.decode('utf-8'))
        elif(eq=='c'):
            cFL_Condition(nx,nt,c)
            # encoded = fig_to_base64(img)    
            # myhtml='<img src="data:image/png;base64, {}">'.format(encoded.decode('utf-8'))
        elif(eq=='d'):
            burgers_Equation_in_2D(nx,nt,c)
            # encoded = fig_to_base64(img)    
            # myhtml='<img src="data:image/png;base64, {}">'.format(encoded.decode('utf-8'))

        im = Image.open("abcd.png")
        data = io.BytesIO()
        im.save(data, "PNG")
        encoded_img_data = base64.b64encode(data.getvalue())

            
            
            
            
            
            
        
    

        # fig, ax = plt.subplots(figsize=(16, 8)) 
        # ax.set_title("Operational balance Chart")
        # ax.set_xlabel("Operation Number")
        # #plt.xticks(ind, ('T1', 'T2', 'T3', 'T4', 'T5'))
        # res = [i / j for i, j in zip(BT, OQ)]
        # ax.bar(index, res, 0.6, color="lightskyblue", label='Cycle Time')

        # # horizontal line indicating the threshold
        # ax.plot([0, len(BT)+1], [T, T], "k--", label='Takt Time')
        # ax.legend(loc='upper right')
        # fig.savefig("OperatorBalanceChart.png")
        # im = Image.open("OperatorBalanceChart.png")
        # data = io.BytesIO()
        # im.save(data, "PNG")
        # encoded_img_data = base64.b64encode(data.getvalue())
        
        
        return render_template('indexa.html',user_image = encoded_img_data.decode('utf-8'))   
        # return render_template('indexa.html')   


if __name__ == '__main__':
    app.run(debug=True)
