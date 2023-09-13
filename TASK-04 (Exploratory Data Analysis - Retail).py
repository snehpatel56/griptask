#!/usr/bin/env python
# coding: utf-8

# # Hello This is sneh patel.

# # TASK-04 (Exploratory Data Analysis - Retail)
# 
# Problem Statement :
# 1) Perform 'Exploratory data analysis' on dataset 'SampleSuperstore'
# 2) As a business manager, try to find out the weak areas where you can work to make more profit.
# 3) What business problem you can derive by exploring data?

# ![download%20%281%29.png](attachment:download%20%281%29.png)

# In[1]:


#import the warnings.
import warnings
warnings.filterwarnings("ignore")


# In[2]:


#import the useful libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')



# In[3]:


inp0 = pd.read_csv("SampleSuperstore.csv")


# In[5]:


inp0


# In[6]:


inp0.head()


# In[7]:


inp0.tail()


# In[8]:


inp0.shape


# In[9]:


inp0.isnull().sum()


# In[17]:


inp0.info()


# In[10]:


inp0.describe()


# In[11]:


inp0.duplicated().sum()


# In[15]:


inp0.shape


# In[12]:


inp0.drop_duplicates(inplace=True)


# In[19]:


compare = inp0.loc[:,["Sales","Quantity" ,"Discount" ,"Profit" ]]
correlation =compare.corr()
sns.heatmap(correlation,xticklabels=correlation.columns,yticklabels=compare.columns,annot=True)
plt.show()


# In[13]:


sns.boxplot(data=inp0)


# In[14]:


sns.pairplot(data=inp0, hue="Region")


# In[20]:


sns.displot(inp0["Sales"])
sns.displot(inp0["Profit"])
sns.displot(inp0["Discount"])
sns.displot(inp0["Quantity"])


# In[44]:


Sales = inp0.groupby("State").Sales.sum()
Profit =inp0.groupby("State").Profit.sum()


# In[37]:


Sales.plot(kind="bar")

plt.title("state vs sales")
plt.figure(figsize=(8,10))
plt.xlabel("state")
plt.ylabel("total sales")
plt.show()


# In[43]:


Profit.plot(kind="bar")
plt.title("state vs total profit")
plt.figure(figsize=(8,10))
plt.xlabel("state")
plt.ylabel("total profit")
plt.show()


# In[45]:


inp0["Ship Mode"].unique()


# In[56]:


inp0.groupby("Ship Mode")["Sales"].sum().plot.bar(color="red")


# In[51]:


inp0.groupby("Ship Mode")["Profit"].sum().plot.bar()


# In[57]:


inp0.groupby("Ship Mode")["Discount"].sum().plot.bar(color="green")


# In[58]:


inp0["Segment"].unique()


# In[64]:


inp0.groupby("Segment")["Sales"].sum().plot.bar(color="red")


# In[62]:


inp0.groupby("Segment")["Profit"].sum().plot.bar()


# In[65]:


inp0.groupby("Segment")["Discount"].sum().plot.bar(color="green")


# In[66]:


inp0["Region"].unique()


# In[67]:


inp0.groupby("Region")["Sales"].sum().plot.bar(color="red")


# In[68]:


inp0.groupby("Region")["Profit"].sum().plot.bar()


# In[69]:


inp0.groupby("Region")["Discount"].sum().plot.bar(color="green")


# In[70]:


inp0["Category"].unique()


# In[71]:


inp0.groupby("Category")["Sales"].sum().plot.bar(color="red")


# In[72]:


inp0.groupby("Category")["Profit"].sum().plot.bar()


# In[74]:


inp0.groupby("Category")["Discount"].sum().plot.bar(color="green")


# In[87]:


plt.figure(figsize=(20,10))
sns.countplot(data=inp0 ,x="Sub-Category",orient='h',order=inp0["Sub-Category"].value_counts().index)
plt.show()


# In[85]:


plt.figure(figsize=(14,8))
sns.countplot(data=inp0,x="Sub-Category" ,hue="Segment")
plt.show()


# # conclusion:

# 1) standard class has highest profit and sales and they are providing lowest discount on same day
# 2) segment no 1 is consumar type
# 3) highest sales & profit is in west region and highest discount in central region in this case they should start campaign and more advertise in south region if south region people will get more discount then it will increase sales
# 4) they have higher sales and profit in technology and they should provide higher discount in furnither
# 5) 70% of product in sub category are in  most demand
# 6)  in all segment consumer rate is in top

# # solution:
#     

# 1) they should provide discount offer in south region they should focus on  product which are in most demand and most selling
# 2) profit is best when discount is 0-15% but after its decreasing
