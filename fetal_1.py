import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('fetal_health.csv')

data.head()
data.shape
data.info()
data.isnull().sum()
data.describe()

data['fetal_health'].value_counts()
sns.histplot(data=data['fetal_health'], color='green', edgecolor='green')
plt.title('Histogram of fetal health categories')
plt.show()

plt.figure(figsize=(15,10))
sns.boxplot(data)
plt.title('Boxplot before handling outliers and scaling')
plt.xticks(rotation=45)
plt.show()

corr_matrix = data.corr()
plt.figure(figsize=(15,10))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True)
plt.title('Correlation matrix of features')
plt.show()

data.hist()
plt.show()

#sns.pairplot(data)
#plt.show()

data['fetal_health'].value_counts().plot(kind='pie',autopct='%1.0f%%', labels=['Normal','Suspect','Pathological'])
plt.title('Pies chart of fetal health categories')
plt.show()

from pandas.plotting import scatter_matrix
#scat_matrix = scatter_matrix(data, figsize=(50,50), color='blue', marker='.')
#plt.show()
 

