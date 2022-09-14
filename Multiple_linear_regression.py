# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 17:57:15 2021

@author: monte
"""



import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
   
df = pd.read_csv("FuelConsumptionCo2.csv") #lectura del dataset
#df.head()#LECTURA DEL TITULO DE CADA COLUMNA

#Podemos seleccionar algunas caracteristicas del dataset
cdf=df[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB","CO2EMISSIONS"]]


 #CREAMOS UN SET DE ENTREMANIENTO Y DE PRUEBA(TRAIN/SPLIT).

msk=np.random.rand(len(df)) < 0.80
train=cdf[msk]#Set de entrenamiento
test=cdf[~msk]#set de prueba

#MODELO DE REGRESION SIMPLE

#### Entrenar distribución de los datos"
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

#MODELAMOS LA REGRESION LINEAL MULTIPLE E IMPRIMIMOS COEFICIENTES DE REGRESION
###############ENTRENAMIENTO DEL MODELO(TRAIN)
from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE',"CYLINDERS","FUELCONSUMPTION_COMB"]])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (x, y).predict(x)


# Imprimimos los coeficientes de regresión.
print ('Coefficients:', regr.coef_)


######################################
####METODO DE MINIMOS CUADRADOS ORDINARIOS
#####################PRUEBA DEL MODELO(TEST)

y_hat = regr.predict(test[[ 'ENGINESIZE',"CYLINDERS","FUELCONSUMPTION_COMB"]])
x = np.asanyarray(test[['ENGINESIZE',"CYLINDERS","FUELCONSUMPTION_COMB"]])
y = np.asanyarray(test[['CO2EMISSIONS']])

print("suma residual de los cuadrados:%.2f"
      % np.mean((y_hat-y)**2))

print("Varianza:%.2f"% regr.score(x,y))


