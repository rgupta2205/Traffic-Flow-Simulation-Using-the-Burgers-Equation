import io
from flask import Flask, render_template, request
import pandas as pd
# %matplotlib inline
import math
import pandas as pd
import pandas as pd 
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
from matplotlib import pyplot      #here we load matplotlib
import time, sys  
from matplotlib import pyplot, cm


app = Flask(__name__)

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template("index.html")


@app.route("/result", methods=['GET', 'POST'])
def result():
    output = request.form.to_dict()
    Svalue = output[""]

    return render_template("index.html", Svalue = Svalue)


if __name__ == '__main__':
    app.run(debug=True)
