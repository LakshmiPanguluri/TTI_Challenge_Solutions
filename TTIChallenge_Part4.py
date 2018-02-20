import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#Part4.i-Importing the .xlsx file to Pandas Dataframe
data2= pd.read_excel('E:\TTIChallenge\ModelingDataSet.xlsx', sheet_name='Transactions')
data2.columns= data2.columns.str.replace(' ','')
X1= data2.iloc[:, 5:]

plt.hist(data2['Margin%'])

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X1)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#Fitting K-Means to the Margin of the dataset and appending formed clusternumbers(0 to 5) to dataset
kmeans_4 = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans_4= kmeans_4.fit_predict(X1)

#Assigning  the clusters created  to the dataset
data2['Optimal_Clusters4'] = y_kmeans_4
     
#Distribution of extended costs across different clusters formed
sns.set_style("whitegrid")
ax = sns.boxplot(x= "Optimal_Clusters4", y="Extended_cost", data=data2)
plt.show()    
#Distribution of extended costs(0,6000$) across different clusters formed
sns.set_style("whitegrid")
ax = sns.boxplot(x= "Optimal_Clusters4", y="Extended_cost", data=data2)
plt.ylim(0, 6000)
plt.show()

#Part 4.i.1
#Random Forest classification
X = data2.iloc[:, 3:4].values                                 
y = data2.iloc[:, 6].values
                
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier_RF = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier_RF.fit(X_train, y_train)

# Predicting the Test set results
y_pred_RF = classifier_RF.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm_RF = confusion_matrix(y_test, y_pred_RF)
ac_RF = accuracy_score(y_test, y_pred_RF)
     
    
#Part 4.ii.b      
# Fitting K-Means to the Margin of the dataset and assigning the appending formed clusternumbers to dataset
kmeans={}
for nb_b in range(3,11):
    kmeans[nb_b] = KMeans(n_clusters = nb_b, init = 'k-means++', random_state = 42)
    data2['cluster{0}'.format(nb_b)] = kmeans[nb_b].fit_predict(X1)

#Displaying the results of number of bins and clusters and cluster average margin % and coefficient of variation for each cluster.       
avg_margin={}
coef_var={}
for nb_b in range(3,11):
    print("cluster with",str(nb_b),"bins")
    print("clusternumber","average_margin%","coef_of_var")
    avg_margin_temp = [data2[data2['cluster{0}'.format(nb_b)] == value]['Margin%'].mean() for value in range(nb_b)]
    coef_var_temp = [data2[data2['cluster{0}'.format(nb_b)] == value]['Margin%'].std()*100/avg_margin_temp[value] for value in range(nb_b)]
    for i in range(nb_b):
        print("cluster", str(i), avg_margin_temp[i], coef_var_temp[i])
    avg_margin[nb_b] = avg_margin_temp
    coef_var[nb_b] = coef_var_temp

a=[]
for i in range(3,11):
    for _ in range(i):
        a.append(i)
 
b=[]
for i in range(3,11):
    for val in avg_margin[i]:
        b.append(val)
plt.scatter(x=a, y=b)
plt.xlabel('NoofBins')
plt.ylabel('avg_margin')

#Part 4.ii.b
#Plotting scatter plot for numerical data
from pandas.plotting import scatter_matrix
scatter_matrix(data2)
#Finding if there is any correlation between the variables               
correlations= data2.corr(method='spearman')
mask = np.zeros_like(correlations, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
     # Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(correlations, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

#Identifying the variables to segment margins clusters 
#2.Data Preprocessing and Random Forest Regressions using Extended_cost and Revenue
names1= list(data2.columns[3:5])
X_Final1 = data2.iloc[:, 3:5]                                
y_Final1 = data2.iloc[:, 6]
                
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X_Final1 = StandardScaler()
X_Final1 = sc_X_Final1.fit_transform(X_Final1)
                
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train11, X_test11, y_train11, y_test11 = train_test_split(X_Final1, y_Final1, test_size = 0.2, random_state = 0) 

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier_RF_11 = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier_RF_11.fit(X_train11, y_train11)

# Predicting the Test set results
y_pred_RF11 = classifier_RF_11.predict(X_test11)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm_RF11 = confusion_matrix(y_test11, y_pred_RF11)
ac_RF11 = accuracy_score(y_test11, y_pred_RF11)

classifier_RF_11.feature_importances_
print(sorted(zip(map(lambda x: round(x, 4), classifier_RF_11.feature_importances_), names1), 
             reverse=True))

#Data Preprocessing and Random Forest Regressions using Quantity, Extended_cost and Revenue
names2= list(data2.columns[2:5])
X_Final2 = data2.iloc[:, 2:5]                                
y_Final2 = data2.iloc[:, 6]
                
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X_Final2 = StandardScaler()
X_Final2 = sc_X_Final2.fit_transform(X_Final2)
                
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train12, X_test12, y_train12, y_test12 = train_test_split(X_Final2, y_Final2, test_size = 0.2, random_state = 0) 

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier_RF_12 = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier_RF_12.fit(X_train12, y_train12)

# Predicting the Test set results
y_pred_RF12 = classifier_RF_12.predict(X_test12)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm_RF12 = confusion_matrix(y_test12, y_pred_RF12)
ac_RF12 = accuracy_score(y_test12, y_pred_RF12)

classifier_RF_12.feature_importances_
print(sorted(zip(map(lambda x: round(x, 4), classifier_RF_12.feature_importances_), names2), 
             reverse=True))

#Random Forest Regressions using Unit_cost, Quantity, Extended_cost and Revenue
names3= list(data2.columns[1:5])
X_Final3 = data2.iloc[:, 1:5]                                
y_Final3 = data2.iloc[:, 6]
                
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X_Final3 = StandardScaler()
X_Final3 = sc_X_Final2.fit_transform(X_Final3)
                
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train13, X_test13, y_train13, y_test13 = train_test_split(X_Final3, y_Final3, test_size = 0.2, random_state = 0) 

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier_RF_13 = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier_RF_13.fit(X_train13, y_train13)

# Predicting the Test set results
y_pred_RF13 = classifier_RF_13.predict(X_test13)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm_RF13 = confusion_matrix(y_test13, y_pred_RF13)
ac_RF13 = accuracy_score(y_test13, y_pred_RF13)

classifier_RF_13.feature_importances_
print(sorted(zip(map(lambda x: round(x, 4), classifier_RF_13.feature_importances_), names3), 
             reverse=True))




