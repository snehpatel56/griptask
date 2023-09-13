#!/usr/bin/env python
# coding: utf-8

# 
# # Hello This is sneh patel.

# # TASK-03 (Exploratory Data Analysis - terrorism)
# 
# Problem Statement :
# 1) Perform 'Exploratory data analysis' on dataset 'globalterrorismdb'
# 2) try to find out insight as to analysis attacks 
# 

# ![download%20%281%29.png](attachment:download%20%281%29.png)

# In[165]:


#import the warnings.
import warnings
warnings.filterwarnings("ignore")


# In[122]:


#import the useful libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[56]:


inp0 = pd.read_csv("globalterrorismdb.csv" ,encoding="latin-1")


# In[57]:


inp0


# In[58]:


inp0.shape


# In[59]:


inp0.head()


# In[60]:


inp0.tail()


# In[61]:


inp0.describe()


# In[62]:


inp0.columns.values


# In[67]:


inp0.rename(columns={'iyear':'Year','imonth':'Month','iday':"day",'gname':'Group','country_txt':'Country','region_txt':'Region','provstate':'State','city':'City','latitude':'latitude',
    'longitude':'longitude','summary':'summary','attacktype1_txt':'Attacktype','targtype1_txt':'Targettype','weaptype1_txt':'Weapon','nkill':'kill',
     'nwound':'Wound'},inplace=True)
inp0.columns.values


# In[69]:


inp1=inp0[['Year','Month','day','Country','State','Region','City','latitude','longitude',"Attacktype",'kill','Wound','target1','summary','Group','Targettype','Weapon','motive']]
inp1


# In[70]:


inp1.info()


# In[76]:


inp1.isnull().sum()


# In[75]:


inp1 = inp1[~inp1.kill.isnull()]
inp1 = inp1[~inp1.Wound.isnull()]


# In[92]:


year = inp1["Year"].unique()
year_count = inp1["Year"].value_counts(dropna = True).sort_index()
plt.figure(figsize=(18,10))
plt.xticks(rotation = 50)
plt.xlabel('Attacking Year',fontsize=20)
plt.ylabel('Number of Attacks Each Year',fontsize=20)
plt.title('Attacks In Years From 1970 to 2017',fontsize=30)
sns.barplot(x= year ,y=year_count)
plt.show()


# In[129]:


pd.crosstab(inp1.Year , inp1.Region).plot(kind ="area" ,figsize=(20,10))
plt.title('Region Wise Terrorist Activities For Each Year',fontsize=25)
plt.ylabel('Number of Attacks',fontsize=20)
plt.xlabel("Year",fontsize=20)
plt.show()


# In[134]:


attack = inp1.Country.value_counts()
attack.head(10)


# In[136]:


inp1.Group.value_counts()[1:10]


# In[147]:


df = inp1[["Year","kill"]].groupby(["Year"]).sum()
df.plot(kind="bar",alpha=0.7)
plt.figure(figsize=(20,10))


# In[149]:


inp1['City'].value_counts().to_frame().sort_values('City',axis=0,ascending=False).head(10).plot(kind='bar',figsize=(20,10),color='#698B22')
plt.xticks(rotation = 50)
plt.xlabel("City",fontsize=15)
plt.ylabel("Number of attack",fontsize=15)
plt.title("Top 10 most effected city",fontsize=20)
plt.show()


# In[151]:


inp1[['Attacktype','kill']].groupby(["Attacktype"],axis=0).sum().plot(kind='bar',figsize=(20,10),color=['#CD6600'])
plt.xticks(rotation=50)
plt.title("Number of killed ",fontsize=20)
plt.ylabel('Number of people',fontsize=15)
plt.xlabel('Attack type',fontsize=15)
plt.show()


# In[155]:


inp1[['Attacktype','Wound']].groupby(["Attacktype"],axis=0).sum().plot(kind='bar',figsize=(20,10),color=["#9A32CD"])
plt.xticks(rotation=50)
plt.title("Number of wounded ",fontsize=20)
plt.ylabel('Number of people',fontsize=15)
plt.xlabel('Attack type',fontsize=15)
plt.show()


# In[156]:


inp1['Group'].value_counts().to_frame().drop('Unknown').head(10).plot(kind='bar',color='#00688B',figsize=(20,10))
plt.title("Top 10 terrorist group attack",fontsize=20)
plt.xlabel("terrorist group name",fontsize=15)
plt.ylabel("Attack number",fontsize=15)
plt.show()


# In[157]:


inp1['Year'].plot.hist() 
plt.xlabel('year')
plt.ylabel('number of attacks')
plt.title('terrorism attacks over years')
plt.show()


# In[158]:


df1=inp1[['Group','Country','kill']]
df1=df1.groupby(['Group','Country'],axis=0).sum().sort_values('kill',ascending=False).drop('Unknown').reset_index().head(10)
df1


# In[161]:


typeKill = inp1.pivot_table(columns='Attacktype', values='kill', aggfunc='sum')
typeKill


# In[163]:


countryKill = inp1.pivot_table(columns='Country', values='kill', aggfunc='sum')
countryKill


# # conclusion:
# 1) Country with the most attacks are occured in Iraq
# 2) City with the most attacks Were Occured in Baghdad
# 3) Region with the most attacks were Occured Middle East & North Africa
# 4) Year with the most attacks was 2016
# 5) Taliban organization is responsible for highest number of attacks
# 6) Most Attack Type was Bombing/Explosion
# 7) also Number of people that Wounded is maximum due to Bombing and Explosion
