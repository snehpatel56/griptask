#!/usr/bin/env python
# coding: utf-8

# # Hello This is sneh patel.

# # Task 2 : Prediction using unsupervised ML

# ![download%20%281%29.png](attachment:download%20%281%29.png)

# In[62]:


#import the warnings.
import warnings
warnings.filterwarnings("ignore")


# In[63]:


#import the useful libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.cluster import KMeans


# In[81]:


inp0 = pd.read_csv("Iris.csv")


# In[82]:


inp0


# In[66]:


inp0.shape


# In[67]:


inp0.head()


# In[68]:


inp0.tail()


# In[69]:


inp0.info()


# In[70]:


inp0.describe()


# In[71]:


inp0.isnull().sum()


# In[83]:


inp0.drop(['Id' , 'Species'] , axis = 1 , inplace = True)


# In[84]:


inp0.duplicated().sum()


# In[85]:


inp0.drop_duplicates(inplace=True)


# In[86]:


inp0.duplicated().sum()


# In[87]:


inp0


# In[88]:


sns.scatterplot(data=inp0,x="SepalLengthCm" ,y="PetalLengthCm")
plt.show()


# In[89]:


sns.scatterplot(data=inp0,x="SepalWidthCm" ,y="PetalWidthCm")
plt.show()


# In[90]:


sns.heatmap(inp0.corr() , annot=True)
plt.show()


# In[91]:


scup =[]

for i in range(1,11):
    Kmeans=KMeans(n_clusters=i ,random_state=40, max_iter = 300 , n_init = 10 )
    Kmeans.fit(inp0.values)
    scup.append(Kmeans.inertia_)


# In[106]:


sns.lineplot(x= range(1,11), y=scup, marker='o', linestyle='-', color='g')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()


# In[94]:


Kmeans=KMeans(n_clusters=3 ,random_state=0, max_iter = 300 , n_init = 10 )


# In[97]:


y_kmeans = kmeans.fit_predict(inp0.values)


# In[103]:


sns.scatterplot(x = inp0.values[y_kmeans == 0, 0] , y = inp0.values[y_kmeans == 0, 1],markers = '-')
sns.scatterplot(x = inp0.values[y_kmeans == 1, 0] , y = inp0.values[y_kmeans == 1, 1])
sns.scatterplot(x = inp0.values[y_kmeans == 2, 0] , y = inp0.values[y_kmeans == 2, 1])
sns.scatterplot(x = kmeans.cluster_centers_[:, 0] , y = kmeans.cluster_centers_[:,1],label='Centroids' ,c="red",s=300)
plt.title('K-means Clustering (k=3) for Iris Dataset')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.legend()
plt.show()
     


# In[ ]:





# In[ ]:





# In[ ]:




