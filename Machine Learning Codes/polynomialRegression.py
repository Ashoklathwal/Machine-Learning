# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np
 
f, n = map(int, input().split())
x_matrix = []
y_matrix = []
for i in range(n):
    #inp = [float(d) for d in input().split(' ')] 
    inp = list(map(float, input().split(' ')))
    x_matrix.append(inp[0:f])
    y_matrix.append(inp[f])
polyFeatures = preprocessing.PolynomialFeatures()
x_matrix=polyFeatures.fit_transform(x_matrix)    
lin = LinearRegression()
lin.fit(x_matrix, y_matrix)
t = (int)(input())
for i in range(t):
    #x = [float(d) for d in input().split(' ')]
    x = list(map(float, input().split(' ')))
    x = polyFeatures.fit_transform(x)
    print("%.2f" % lin.predict(x))
    
