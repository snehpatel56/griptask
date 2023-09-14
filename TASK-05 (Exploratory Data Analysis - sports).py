#!/usr/bin/env python
# coding: utf-8

# # Hello This is sneh patel.

# # TASK-05 (Exploratory Data Analysis - sports)
# 
# Problem Statement:
# 1) Perform 'Exploratory Data Analysis' on data set ' Indian Premier League'
# 
# 2) As a Sports Analysts, find out the most successful teams, players and factors contributing win or lose of teams.
# 
# 3) Suggest teams or players a company should endorse for its products.

# ![download%20%281%29.png](attachment:download%20%281%29.png)

# In[49]:


#import the warnings.
import warnings
warnings.filterwarnings("ignore")


# In[50]:


#import the useful libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[51]:


match = pd.read_csv("matches.csv")


# In[52]:


deli = pd.read_csv("deliveries.csv")


# In[53]:


match.head()


# In[54]:


match.shape


# In[55]:


match.info()


# In[56]:


match.describe()


# In[57]:


match.columns


# In[58]:


match.isnull().sum()


# In[59]:


deli.head()


# In[60]:


deli.shape


# In[29]:


deli.info()


# In[61]:


deli.describe()


# In[62]:


deli.isnull().sum()


# In[63]:


deli.columns


# In[65]:


season_data = match[["id","season","winner"]]
season_data


# In[66]:


final_data = deli.merge(season_data,how="inner",left_on="match_id" , right_on="id")


# In[40]:


final_data


# In[68]:


winning_team = match.groupby("season")["winner"].value_counts()
winning_team


# In[71]:


plt.figure(figsize=(18,10))
sns.countplot(data=match ,x="season")
plt.title("number of matches played in each season")
plt.xlabel("season")
plt.ylabel("matches")
plt.show()


# In[74]:


plt.figure(figsize=(18,10))
sns.countplot(data=final_data ,x="winner")
plt.title("number of matches won by team")
plt.xticks(rotation=50)
plt.xlabel("teams")
plt.ylabel("no of wins")
plt.show()


# In[100]:


match['win_by'] = match['win_by_runs'].apply(lambda x: 'Bat first' if x > 0 else 'Bowl first')


# In[101]:


win=match.win_by.value_counts()
labels=pd.array(win.index)
sizes = win.values
colors = ['#00C957', '#B22222']
plt.figure(figsize = (10,8))
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True,startangle=90)
plt.title('Match Result',fontsize=20)
plt.axis('equal')
plt.show()


# In[91]:


Toss=match.toss_decision.value_counts()
labels= pd.array(Toss.index)
sizes = Toss.values
colors = ['#FFBF00', '#FA8072']
plt.figure(figsize = (10,8))
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True,startangle=90)
plt.title('Toss result',fontsize=20)
plt.axis('equal')
plt.show()


# In[102]:


plt.figure(figsize = (18,10))
sns.countplot(data=match ,x='season',hue='win_by')
plt.title("Numbers of matches won by batting and bowling first ",fontsize=20)
plt.xlabel("Season",fontsize=15)
plt.ylabel("Count",fontsize=15)
plt.show()


# In[108]:


plt.figure(figsize = (18,10))
sns.countplot(data=match ,x='season',hue='toss_decision',palette="Greens")
plt.title("Numbers of matches won by toss result ",fontsize=20)
plt.xlabel("Season",fontsize=15)
plt.ylabel("Count",fontsize=15)
plt.show()


# In[120]:


plt.figure(figsize=(38,20))
sns.countplot(data= match,x="venue")
plt.title("Number Of Matches Played On Each Ground ")
plt.xlabel("Venue")
plt.ylabel("Matches")
plt.xticks(rotation=50)
plt.show()


# In[121]:


final_matches=match.drop_duplicates(subset=['season'], keep='last')

final_matches[['season','winner']].reset_index(drop=True).sort_values('season')


# In[130]:


plt.figure(figsize=(18,10))
top_players = match.player_of_match.value_counts()[:10]
sns.barplot(x=top_players.index ,y=top_players)
plt.xlabel("player")
plt.show()


# In[131]:


final_matches["winner"].value_counts()


# In[133]:


final_matches[["toss_winner","toss_decision","winner"]].reset_index(drop=True)


# In[134]:


final_matches[['winner','player_of_match']].reset_index(drop=True)


# In[146]:


four=final_data[final_data['batsman_runs']==4]
four.groupby('batting_team')['batsman_runs'].agg([('runs by fours','sum'),('fours','count')])


# In[147]:


six=final_data[final_data['batsman_runs']==6]
six.groupby('batting_team')['batsman_runs'].agg([('runs by fours','sum'),('fours','count')])


# In[142]:


deli.groupby('bowler')['dismissal_kind'].agg(['count']).reset_index().sort_values('count',ascending=False).reset_index(drop=True).iloc[:10,:]


# # conclusion:

# 1) The highest number of match played in IPL season was 2011,2012,2013.
# 
# 2) The highest number of match won by Mumbai Indians 4 matches with maximum runs of 146 in the game against Delhi Daredevils.
# 
# 3) Major Games are won by bowl first as Bowl first has higher chances of winning then the team which bat first.
# 
# 4) Team decide Fielding First After Winning The Toss with probability of (61%).
# 
# 5) In finals most teams after winning toss decide to do fielding first as it increses chance of winning .
# 
# 6) CH gayle, AB de villers are Top player of match winning.
# 
# 7) It is interesting that out of 12 IPL finals,9 times the team that won the toss was also the winner of IPL.
# 
# 8) Shikar Dhawan hit highest number of four.
# 
# 9) The highest number of six hit by player is CH gayle.
# 
# 10) Top run scorer in IPL are Virat kholi, SK Raina, RG Sharma.
# 
# 11) SL Malinga takes highest wicket IPL 
