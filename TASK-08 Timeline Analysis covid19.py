#!/usr/bin/env python
# coding: utf-8

# # Hello This is sneh patel.

# # TASK-08 (Timeline Analysis:- covid19)

# problem statement:
# 
# 1) Create a storyboard showing spread of Covid-19 cases in your country or any region (Asia, Europe, BRICS etc) 
# 
# 2) Use animation, timeline and annotations to create attractive and interactive dashboards and story
# 
# 3) Identify interesting patterns and possible reasons helping Covid-19 spread  with basic as well as advanced charts.
# 
# 

# ![download%20%281%29.png](attachment:download%20%281%29.png)

# In[3]:


#import the warnings.
import warnings
warnings.filterwarnings("ignore")


# In[1]:


# All basic required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import seaborn as sns


# In[2]:


#importing our dataset from local machine

df = pd.read_csv('owid-covid-data.csv')
df.head()


# In[3]:


#Take a look of dataset from below

df.tail()


# In[4]:


df.shape  #shape of our data


# In[5]:


df.columns # all the columns inside the dataset


# In[6]:


df.info() # overall information about the dataset


# In[7]:


#basic statisical details 
df.describe()


# In[8]:


df.count() #Used to count the number of rows


# In[9]:


duplicate_df = df[df.duplicated()] # this is way to check duplicate values

print("Number of duplicate rows: ", duplicate_df.shape)


# In[10]:


df = df.drop_duplicates() # drop duplicate values

df.head()


# In[11]:


df.isna().sum() # checking the null values


# ### Data visualization & Analysis report

# In[12]:


#using histogram to represent the relationship between all features

df.hist(figsize=(35,40))
plt.show()


# In[29]:


# Most effected continent

df['continent'].value_counts()


# In[65]:


top_10_countries_by_most_deaths = df.sort_values("total_deaths",ascending =False)
top_10_countries_by_most_deaths ['location'].tail(10)


# In[14]:


last_cases = df[df["date"] == "2021-10-08"] # to check the last day casesof each country 
last_cases.head() # to print only first five


# In[15]:


max_cases = last_cases.sort_values(by = "total_cases" , ascending = False) # this is the last day max cases of each country
max_cases


# In[16]:


max_cases[1:6]  # Top 5 Countries of having maximum cases


# In[17]:


plt.figure(figsize=(10,5))  #bar plot of first  10 countries having maximum cases
sns.barplot(x = 'location' , y = 'total_cases' , data = max_cases[1:11] , hue = 'date')
plt.xticks(rotation=90)
#plt.style.use('dark_background')
plt.show()


# ### Analysis of COVID Cases in Asia

# In[18]:


data_asia = df[df['continent'] =='Asia']
data_asia = data_asia[~data_asia['location'].isin(['World','International'])]
data_asia.head()


# In[19]:


df_asia=data_asia[["date","total_cases","new_cases","total_deaths","new_tests","total_tests",
                  "location",'new_deaths_per_million','total_tests_per_thousand','new_tests_per_thousand',
                  'positive_rate','new_deaths']]
df_asia.head()


# In[20]:


# Total Test per day
sns.set(rc = {"figure.figsize" : (10,5)})
sns.lineplot(x = "date" , y = "total_tests" , data = df_asia)
plt.show()


# In[67]:


plt.figure(figsize=(10,5))
sns.lineplot(x='total_cases',y='total_deaths',data=df_asia)
plt.title("Total Cases vs Total Deaths")


# #### From the above graph we can say that as the total cases increases the total deaths are also increasing.

# In[61]:


plt.figure(figsize=(10,5)) # total tests vs positive rate
sns.lineplot(x='total_tests',y='positive_rate',data=df_asia)
plt.title("Total Tests vs Positive Rate")


# #### From the above line graph we can say that, in the beginning as the number of tests were being done the positive rate also had increased but it decreased after around 1 le8 and after that it has been gradualy increasing.

# In[62]:


plt.figure(figsize=(10,5))
sns.lineplot(x='total_deaths',y='new_deaths_per_million',data=df_asia)
plt.title("Total Deaths vs New Deaths per million")


# #### From the above graph we can say that, new deaths per million population had increased very steeply in the beginning. Around 4000 cases approximately number of deaths started increasing slowly and after 41000 deaths approximately it started falling.

# In[21]:


plt.figure(figsize= (20,10)) #Heatmap Showing Correlation of dataset
sns.heatmap(np.round(df_asia.corr(),2),annot= True,cmap='Blues')


# #### India

# In[22]:


# Covid Cases in INDIA
india = df[df["location"] == "India"]
india.head()


# In[23]:


india.tail() # last 5 rows of the dataset


# In[24]:


# Total cases per day
sns.set(rc = {"figure.figsize" : (10,5)})
sns.lineplot(x = 'date' , y = 'total_cases' , data = india)
plt.show()


# In[25]:


india_last_5_days = india.tail() # Total cases last 5 days
sns.set(rc = {"figure.figsize" : (10,5)})
sns.lineplot(x = "date" , y = "total_cases" , data = india_last_5_days)
plt.show()


# In[26]:


# Total Test per day
sns.set(rc = {"figure.figsize" : (10,5)})
sns.lineplot(x = "date" , y = "total_tests" , data = india)
plt.show()


# In[27]:


# New cases per day
sns.set(rc = {"figure.figsize" : (10,5)})
sns.lineplot(x = "date" , y = "new_cases" , data = india)
plt.show()


# In[54]:


# New deaths per day
sns.set(rc = {"figure.figsize" : (10,5)})
sns.lineplot(x = "date" , y = "new_deaths" , data = india)
plt.show()


# In[59]:


# New cases per day
sns.set(rc = {"figure.figsize" : (10,5)})
sns.lineplot(x = "date" , y = "total_deaths", data = india)
plt.show()


# In[58]:


# New cases per day
sns.set(rc = {"figure.figsize" : (10,5)})
sns.lineplot(x = "date" , y = "people_vaccinated", data = india)
plt.show()


# In[47]:


india.columns


# In[53]:


new=df[df['location']=='India'][['date','total_cases','new_cases',]]
new[:10]
# This is the top 10 info about total cases and new cases similarly you can other info also

