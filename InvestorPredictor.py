# -*- coding: utf-8 -*-
"""
Data Set is from Aug 16, 2017 - Aug 15, 2018.
Retrieved from finance.yahoo.com
I am aware that this one feature is pretty much useless

@author: Michael Ehnes
"""

# Importing Libs
import os
import numpy as np
import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt 

dataFilePath = os.getcwd() + "\\Data\\GOOGL.csv"

df = pd.read_csv(dataFilePath)

def predict_prices(df, x):
    
    # Extracting the dates and opening prices from the data frame
    # Values in the "Date" column are objects so they need to be converted to strings so I can remove the "-" inorder for them to be converted to int's
    dates = df[["Date"]].astype(str).replace('-', '', regex=True).astype(int)    
    
    prices = df[["Open"]]
    prices = prices.values.ravel()
    
    # Creating models and fitting data
    svr_lin = SVR(kernel = "linear", C = 1e3)
    print("Fitting data")
    svr_lin.fit(dates, prices)
    print("Done fitting data")
    
    svr_poly = SVR(kernel = "poly", C = 1e3, degree = 2)
    svr_poly.fit(dates, prices)
    
    svr_rbf = SVR(kernel = "rbf", C = 1e3, gamma = 0.1)
    svr_rbf.fit(dates, prices)
    
    # Creating the graph for visualization of data
    plt.scatter(dates, prices, c = "black", label = "Data")
    
    # Starting predictions
    print("Predicting linear")
    plt.plot(dates, svr_lin.predict(dates), color = "blue", label = "Linear")
    
    print("Predicting Poly")
    plt.plot(dates, svr_poly.predict(dates), color = "magenta", label = "Poly")
    
    print("Predicting RBF")
    plt.plot(dates, svr_rbf.predict(dates), color = "red", label = "RBF")
    
    # Labeling axis and giving title
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Which model is best?")
    plt.legend()
    plt.show()
    
    return svr_lin.predict(x)[0], svr_poly.predict(x)[0], svr_rbf.predict(x)[0]

print("Calling predict method")    
predicted_price = predict_prices(df, 29) 
print(predicted_price)
    
    