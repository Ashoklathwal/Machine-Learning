from sklearn.linear_model import LinearRegression
import numpy as np

f, n = map(int, input().split())
x_matrix = []
y_matrix = []
for i in range(n):
    #inp = [float(d) for d in input().split(' ')] 
    inp = list(map(float, input().split(' ')))
    x_matrix.append(inp[0:f])
    y_matrix.append(inp[f])
lin = LinearRegression()
lin.fit(x_matrix, y_matrix)
t = (int)(input())
for i in range(t):
    #x = [float(d) for d in input().split(' ')]
    x = list(map(float, input().split(' ')))
    #x = map(float, input().split())
    print("%.2f" % lin.predict(x))
    