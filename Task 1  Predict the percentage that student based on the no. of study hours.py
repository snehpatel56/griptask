#!/usr/bin/env python
# coding: utf-8

# # Hello This is sneh patel.

# # Task 1 : Predict the percentage that student based on the no. of study hours.

# ![download%20%281%29.png](attachment:download%20%281%29.png)

# In[65]:


#import the warnings.
import warnings
warnings.filterwarnings("ignore")


# In[111]:


#import the useful libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LinearRegression


# In[112]:


inp0 = pd.read_csv("student_scores - student_scores.csv")


# In[113]:


inp0


# In[69]:


inp0.head()


# In[70]:


inp0.tail()


# In[71]:


inp0.shape


# In[72]:


inp0.info()


# In[73]:


inp0.describe()


# In[74]:


inp0.isnull().sum()


# # for visulization we are using bar

# In[75]:


inp0.plot(x='Hours',y='Scores',style='1')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage score')
plt.show()


# In[162]:


sns.scatterplot(data=inp0 ,x="Hours" ,y="Scores")
plt.show()


# In[160]:


inp0.plot.pie(x="Hours",y="Scores")
plt.show()


# In[161]:


sns.barplot(data=inp0 ,x="Hours" ,y="Scores")
plt.show()


# In[146]:


study_hours_mean = inp0['Hours'].mean()
percentage_score_mean = inp0['Scores'].mean()
print(study_hours_mean)
print(percentage_score_mean)


# # splitting the data

# In[130]:


x = inp0[['Hours']].values
y = inp0['Scores'].values 


# In[131]:


from sklearn.model_selection  import train_test_split
x_train, x_test,y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[133]:


model = LinearRegression()


# In[143]:


model.fit=(x_train, y_train)


# In[141]:


model.fit=(x_test ,y_test)


# In[145]:


#Prediction of score
model=LinearRegression().fit(x_train,y_train) # initialize & fit the model
y_pred=model.predict(x_test) # now p


# In[147]:


print(y_test)
print("Prediction of scores")
y_pred=model.predict(x_test)
print(y_pred)


# In[152]:


inp1=pd.DataFrame({'Actual':y_test,'Predicited':y_pred})
inp1


# # evaluate model

# In[153]:


plt.figure(figsize=(10, 6))
sns.regplot(x='Hours', y='Scores', data=inp0, color='b', scatter_kws={'s': 50})
plt.title('Regression Plot of Study Hours vs. Percentage Scores')
plt.xlabel('Study Hours')
plt.ylabel('Percentage Scores')
plt.grid(True)
plt.show()


# In[154]:


hours=[[9.25]]
pred=model.predict(hours)
print(pred)


# In[155]:


from sklearn import metrics
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))


# # conclusion

# 1) as increse of hour study is also increase
# 2) This study has practical implications for both educators and students, offering valuable guidance on optimizing study strategies for improved academic outcomes. By recognizing the pivotal role of study habits, educational stakeholders can implement targeted interventions to enhance learning experiences and achievements.

# In[ ]:




