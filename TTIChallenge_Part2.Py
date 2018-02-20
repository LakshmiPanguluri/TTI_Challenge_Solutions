#TTi Data Challenge
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#part 2
# Reading "Programming_Part1_TTI_Challenge.csv' data into pandas dataframe
Part1_data = pd.read_csv('E:\TTIChallenge\Programming_Part1_TTI_Challenge.csv', sep= '\t')
Part1_data.to_csv('Part1_data.csv')
#2.2
Part1_data.describe()

def geo_mean(X):
     '''x: 1darray(numeric)
     Returns: Gemotric mean of given elements'''

    a = np.array(X)
    return a.prod()**(1.0/len(a))

geo_mean(Part1_data.X)
geo_mean(Part1_data.Y)

#2.3 writing dataframe to SQL table
#importing related libraries to connect to MS sql
#Code
import pyodbc
from sqlalchemy import *
#from sql alchemy import create_engine, Table, Integer, Sequence, Column, MetaData

engine= create_engine('mssql+pyodbc://user_name:password@host:port/database?driver=SQL+Server')
conn= engine.connect()
#creating ‘PartNo’ database table
metadata= MetaData()
PartNo =Table('PartNo', metadata,
    Column('Class', String(20), nullable=False),
    Column('Part_No', String(20), primary_key=True),
    Column('X', Integer, nullable=False),
    Column('Y', Integer, nullable=False)
)
PartNo.create(engine, checkfirst=True)
#Exporting data from ‘Part1_data’ dataframe to ‘PartNo’ table in MSsql
Part1_data.to_sql(name='[servername].[dbname].[PartNo]',con=engine, if_exists='append')


#Scatter plot of X vs Y
plt.scatter(Part1_data.X, Part1_data.Y )
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
# Plotting a linear regression of order 1 between 'X' and 'Y'
sns.regplot(x='X', y='Y', data=Part1_data, scatter=None, color='grey', label='order 1')

# Plotting in green a linear regression of order 2,3,4,5,6 between 'X' and 'Y'
sns.regplot(x='X', y='Y', data=Part1_data,  scatter=None, order=2, color='green', label='order 2')
sns.regplot(x='X', y='Y',  data=Part1_data, scatter=None, order=3, color='purple', label='order 3')
sns.regplot(x='X', y='Y',  data=Part1_data, scatter=None, order=4, color='yellow', label='order 4')
sns.regplot(x='X', y='Y',  data=Part1_data, scatter=None, order=5, color='orange', label='order 5')
sns.regplot(x='X', y='Y',  data=Part1_data, scatter=None, order=6, color='blue', label='order 6')


#Fitting Ploynomial Regression to the dataset
X=  Part1_data.iloc[:,2:3].values
y=  Part1_data.iloc[:,3].values

#generating polynomial features for X orders ranging from 2 to 10
K = 9
from sklearn.preprocessing import PolynomialFeatures
X_poly = {}
for i in range(2,K):
    poly_reg = PolynomialFeatures(i)
    X_poly[i] = poly_reg.fit_transform(X)

# Splitting the dataset into the Training set and Test set for each dataset
X_train = {}
X_test = {}
from sklearn.model_selection import train_test_split
for i in range(2,K):
    X_temp_arr = X_poly[i]
    X_train[i], X_test[i], y_train, y_test= train_test_split(X_temp_arr, y, test_size = 0.25, random_state = 0)

# Fitting Linear regression for each order of X from 2 to K to the training sets
lin_reg = {}
from sklearn.linear_model import LinearRegression
for i in range(2,K):
    lin_reg[i] =LinearRegression()
    lin_reg[i].fit(X_train[i], y_train)

# Predicting the Test set results and finding the mse and rmse for each ordered polynomial
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
y_pred = {}
error = {}
r2_coef={}
mse={}
rmse={}
for i in range(2,K):
    y_pred[i] = lin_reg[i].predict(X_test[i])
    error[i] = y_pred[i]-y_test
    r2_coef[i]=r2_score(y_test, y_pred[i])
    mse[i]= mean_squared_error(y_test, y_pred[i])
    rmse[i]=math.sqrt(mse[i])

print(lin_reg[3].coef_)
print(lin_reg[4].coef_)
print(lin_reg[5].coef_)

#2.5Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = {}
for i in range(2,K):
    accuracies[i] = cross_val_score(estimator = lin_reg[i], X = X_train[i], y = y_train, cv = 10)
    print("Cross validation number i: ", i)
    print (accuracies[i].mean())
    print (accuracies[i].std())



