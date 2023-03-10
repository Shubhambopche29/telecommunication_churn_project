#!/usr/bin/env python
# coding: utf-8

# ## Business objectives
# Customer churn is a big problem for telecommunications companies. Indeed, their annual churn rates are usually higher than 10%. For that reason, they develop strategies to keep as many clients as possible. This is a classification project since the variable to be predicted is binary (churn or loyal customer). The goal here is to model churn probability, conditioned on the customer features.

# ### 1. Importing Libraries

# In[1]:


#Importing the required packages
import pandas as pd
import numpy as np
#importing Visualization Packages
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder


# ### 2. Import Dataset

# In[2]:


churn_data = pd.read_csv("churn.csv")
churn_data


# In[3]:


churn_data.head()


# In[4]:


churn_data.tail()


# ### 3. Data Understanding

# #### Initial Analysis

# In[5]:


churn_data.shape


# In[6]:


print('features of data set :',churn_data.columns)


# In[7]:


churn_data.describe()


# In[8]:


churn_data.dtypes


# In[9]:


churn_data.select_dtypes(include=['object'])


# In[10]:


churn_data.isna().sum()


# In[11]:


churn_data.info()


# In[12]:


churn_data.iloc[10:15]


# In[13]:


churn_data.iloc[101:111]


# ### Here we can see both "Day Charges" and "Eve Minutes" are idenitfied as object datatype
# * Even though data inside both is in float there is Nan Value
# * so we need to convert string Nan to np.Nan
# * And drop the Nan values
# * Change the Datatypes

# In[14]:


nan_value = np.nan
type(nan_value)


# In[15]:


churn_data["day.charge"] = churn_data["day.charge"].replace("Nan",nan_value)


# In[16]:


churn_data["eve.mins"] = churn_data["eve.mins"].replace("Nan",nan_value)


# In[17]:


churn_data.isna().sum()


# In[18]:


churn_data.dropna(inplace=True)


# In[19]:


churn_data.isna().sum()


# In[20]:


# Dropping Unwanted Columns
churn_data.drop("Unnamed: 0",axis=1,inplace=True)


# In[21]:


#converting day.charges and eve.mins from object to float
churn_data["day.charge"] = churn_data["day.charge"].astype(float)
churn_data["eve.mins"] = churn_data["eve.mins"].astype(float)


# In[22]:


churn_data.dtypes


# In[23]:


churn_data.shape


# #### There were total of 31 null values present in our data

# In[24]:


churn_data["churn"].value_counts()


# ### Data is Cleaned

# In[25]:


churn_data.nunique()


# In[26]:


churn_data.describe(include='all').T


# In[27]:


churn_data[churn_data.duplicated()]


# In[28]:


# % Missing Value visualization

missing = pd.DataFrame((churn_data.isnull().sum())*100/churn_data.shape[0]).reset_index()
plt.figure(figsize=(16,5))
ax = sns.pointplot('index',0,data=missing)
plt.xticks(rotation =90,fontsize =7)
plt.title("Percentage of Missing values")
plt.ylabel("PERCENTAGE")
plt.show


# In[29]:


len(churn_data[churn_data.duplicated()])


# * As of now There are 4969 rows and 20 columns in above dataset.
# 
# * out of which there are 8 float data type,
# 
# * 7 integer data type,
# 
# * 5 object data type i.e catagarical value are there.
# 
# * There are no missing value present so no need to do the missing value imputation,
# 
# * And also there are no duplicate value present.

# ## Missing Data - Initial Intuition
# - Here, we don't have any missing data.
# General Thumb Rules:
# 
# - For features with less missing values- can use regression to predict the missing values or fill with the mean of the values present, depending on the feature.
# - For features with very high number of missing values- it is better to drop those columns as they give very less insight on analysis.
# - As there's no thumb rule on what criteria do we delete the columns with high number of missing values, but generally you can delete the columns, if you have more than 30-40% of missing values.

# In[30]:


churn_data.dtypes


# ###  Correlation Analysis

# In[31]:


churn_data.corr()


# In[32]:


plt.figure(figsize=(20,10))
sns.heatmap(churn_data.corr(),annot=True)
plt.show()


# In[33]:


sns.pairplot(data=churn_data)


# ## Exploratory Data Analysis (EDA)
# 

# In[34]:


#unique value inside 'churn' column
churn_data['churn'].unique()


# In[35]:


#count of true & false in ' churn'
churn_data.churn.value_counts()


# In[36]:


#pie chart to analysis to churn
churn_data['churn'].value_counts().plot.pie(explode=[0.05,0.05],autopct='%1.1f%%',startangle=90,shadow=True,figsize=(8,8))
plt.title('pie chart for churn')
plt.show()


# In[37]:


#To get the Donut Plot to analyze churn
data = churn_data['churn'].value_counts()
explode = (0, 0.2)
plt.pie(data, explode = explode,autopct='%1.1f%%',shadow=True,radius = 2.0, labels = ['Not churned customer','churned customer'],colors=['royalblue' ,'lime'])
circle = plt.Circle( (0,0), 1, color='white')
p=plt.gcf()
p.gca().add_artist(circle)
plt.title('Donut Plot for churn')
plt.show()


# In[38]:


#let's see churn by using countplot
sns.countplot(x=churn_data.churn)
plt.show()


# - After analyzing the churn column, we had little to say like almost 15% of customers have churned.
# - let's see what other features say to us and what relation we get after correlated with churn

# In[39]:


#printing the unique value of state column
churn_data['state'].nunique()


# In[40]:


#Separating churn and non churn customers
churn_df     = churn_data[churn_data["churn"] == bool(True)]
not_churn_df = churn_data[churn_data["churn"] == bool(False)]


# ## Analyzing *Account Length* Column
# 

# In[41]:


#Account length vs Churn
sns.distplot(churn_data['account.length'])
plt.show()


# In[42]:


#comparison of churned account length and not churned account length 
sns.distplot(churn_data['account.length'],color = 'blue',label="All")
sns.distplot(churn_df['account.length'],color = "black",hist=True,label="Churned")
sns.distplot(not_churn_df['account.length'],color = 'red',hist= True,label="Not churned")
plt.legend()
plt.show()


# ## Analyzing *State* Column
# 

# In[43]:


#printing the unique value of state column
churn_data['state'].nunique()


# In[44]:


#Comparison churn with state by using countplot
sns.set(style="darkgrid")
plt.figure(figsize=(15,8))
ax = sns.countplot(x='state', hue="churn", data=churn_data)
plt.show()


# In[45]:


label_encode=LabelEncoder()
labels=label_encode.fit_transform(churn_data['churn'])
churn_data['Churn']=labels
churn_data.drop(columns='churn',axis=1,inplace=True)


# In[46]:


plt.rcParams['figure.figsize'] = (12, 7)
color = plt.cm.copper(np.linspace(0, 0.5, 20))
((churn_data.groupby(['state'])['Churn'].mean())*100).sort_values(ascending = False).head(6).plot.bar(color = ['violet','indigo','b','g','y','orange','r'])
plt.title(" State with most churn percentage", fontsize = 20)
plt.xlabel('state', fontsize = 15)
plt.ylabel('percentage', fontsize = 15)
plt.show()


# In[47]:


#calculate State vs Churn percentage
State_data = pd.crosstab(churn_data["state"],churn_data["Churn"])
State_data['Percentage_Churn'] = State_data.apply(lambda x : x[1]*100/(x[0]+x[1]),axis = 1)
print(State_data)


# In[48]:


#show the most churn state of top 10 b ascending the above list
churn_data.groupby(['state'])['Churn'].mean().sort_values(ascending = False).head(10)


# *There is 51 unique state present who have different churn rate.*
# 
# *From the above analysis CA, NJ, WA, TX, MT, MD are the ones who have a higher churn rate of more than 21.*
# 
# *The reason for this churn rate from a particular state may be due to the low coverage of the cellular network.*

# In[49]:


#Show count value of 'yes','no'
churn_data['intl.plan'].value_counts()


# In[50]:


#Calculate the International Plan vs Churn percentage 
International_Plan_data = pd.crosstab(churn_data["intl.plan"],churn_data["Churn"])
International_Plan_data['Percentage Churn'] = International_Plan_data.apply(lambda x : x[1]*100/(x[0]+x[1]),axis = 1)
print(International_Plan_data)


# In[51]:


#To get the Donut Plot to analyze International Plan
plt.figure(figsize=(10,5))
data = churn_data['intl.plan'].value_counts()
explode = (0, 0.2)
plt.pie(data, explode = explode,autopct='%1.1f%%',shadow=True,radius = 2.0, labels = ['No','Yes'],colors=['skyblue' ,'orange'])
circle = plt.Circle( (0,0), 1, color='white')
p=plt.gcf()
p.gca().add_artist(circle)
plt.title('Donut Plot for International plan')
plt.show()


# - There are 5000 people
# - 495 have a International Plan
# - 4505 do not have International Plan

# In[52]:


#Analysing by using countplot
sns.countplot(x='intl.plan',hue="Churn",data = churn_data)
plt.show()


# - This is a count plot which shows the churned and not churned customer respective to their international plan
# 
# - From the above data we get
# 
# - There are 4505 customers who dont have a international plan.
# 
# - There are 495 customers who have a international plan.
# 
# - Among those who have a international plan 42.07 % people churn.
# 
# - Whereas among those who dont have a international plan only 11.2 % people churn.
# 
# - So basically the people who bought International plans are churning in big numbers.
# 
# - Probably because of connectivity issues or high call charge.

# ## Analyzing "Voice Mail Plan" column

# In[53]:


#show the unique value of the "Voice mail plan" column
churn_data["voice.plan"].unique()


# In[54]:


#Calculate the Voice Mail Plan vs Churn percentage
Voice_mail_plan_data = pd.crosstab(churn_data["voice.plan"],churn_data["Churn"])
Voice_mail_plan_data['Percentage Churn'] = Voice_mail_plan_data.apply(lambda x : x[1]*100/(x[0]+x[1]),axis = 1)
print(Voice_mail_plan_data)


# In[55]:


#To get the Donut Plot to analyze Voice mail plan
plt.figure(figsize=(7,3))
data = churn_data['voice.plan'].value_counts()
explode = (0, 0.2)
plt.pie(data, explode = explode,autopct='%1.1f%%',startangle=90,shadow=True,radius = 2.0, labels = ['NO','YES'],colors=['skyblue','red'])
circle = plt.Circle( (0,0), 1, color='white')
p=plt.gcf()
p.gca().add_artist(circle)
plt.title('Donut Plot for Voice mail plan')
plt.show()


# - There are 4969 people,
# - 1317 having Voicemail plan,
# - 3652 do not have any Voicemail plan.

# In[56]:


#Analysing by using countplot
sns.countplot(x='voice.plan',hue="Churn",data = churn_data)
plt.show()


# - As we can see there is are no clear relation between voice mail plan and churn so we can't clearly say anything so let's move to the next voice mail feature i.e number of voice mail, let's see what it gives to us.
# - This plot shows churn corresponding with the subscription of voicemail plan Out of 1317 people having Voicemail plan, 7.7% are Churn.

# ## Analyzing "voice messages" column
# 

# In[57]:


#show the data of 'Number vmail messages' 
churn_data['voice.messages'].unique()


# In[58]:


#Printing the data of 'Number vmail messages'
churn_data['voice.messages'].value_counts()    #The output will be sorted in descending order.


# In[59]:


#Analysing by using displot diagram
sns.distplot(churn_data['voice.messages'])
plt.show()


# In[60]:


#Analysing by using boxplot diagram between 'number vmail messages' and 'churn'
fig = plt.figure(figsize =(10, 8)) 
churn_data.boxplot(column='voice.messages', by='Churn')
fig.suptitle('number of voice message', fontsize=14, fontweight='bold')
plt.show()


# *After analyzing the above voice mail feature data we get an insight that when there are more than 20 voice-mail messages then there is a churn*
# 
# *For that, we need to improve the voice mail quality.*

# ## Analyzing "Customer service calls" column

# In[61]:


#Printing the data of customer service calls 
churn_data['customer.calls'].value_counts()


# In[62]:


#Calculating the Customer service calls vs Churn percentage
Customer_service_calls_data = pd.crosstab(churn_data['customer.calls'],churn_data["Churn"])
Customer_service_calls_data['Percentage_Churn'] = Customer_service_calls_data.apply(lambda x : x[1]*100/(x[0]+x[1]),axis = 1)
print(Customer_service_calls_data)


# - This table mapping number of customer calls to the churn percentage
# - It’s clear that after 4 calls at least 44% of the subscribers churn.
# - Customers with more than 4 service calls their probability of leaving is more

# In[63]:


#Analysing using countplot
sns.countplot(x='customer.calls',hue="Churn",data = churn_data)
plt.show()


# *  it is observed from the above analysis that, mostly because of bad customer service, people tend to leave the operator.
# The above data indicating that those customers who called the service center 5 times or above those customer churn percentage is higher than 60%,And customers who have called once also have a high churn rate indicating their issue was not solved in the first attempt.
# So operator should work to improve the service call.

# ## Analyzing all calls minutes,all calls, all calls charge together

# * D A Y

# In[64]:


#Print the mean value of churned and not churned customer 
print(churn_data.groupby(["Churn"])['day.calls'].mean())


# In[65]:


#Print the mean value of churned and not churned customer 
print(churn_data.groupby(["Churn"])['day.mins'].mean())


# In[66]:


#Print the mean value of churned and not churned customer 
print(churn_data.groupby(["Churn"])['day.charge'].mean())


# In[67]:


#show the relation using scatter plot
sns.scatterplot(x="day.mins", y="day.charge", hue="Churn", data=churn_data,palette='hls')
plt.show()


# * In this graph we can see the frequent red-dotted line till 250 and above it we can see frequent blue-dotted Lines
# * That means customer who are spending less than 250 minutes is happy with our telecom 
# * And customer spending above 250 minutes is not satisfied with our telecom network so tend to churn to competitors

# * E V E N I N G

# In[68]:


#Print the mean value of churned and not churned customer 
print(churn_data.groupby(["Churn"])['eve.calls'].mean())


# In[69]:


#Print the mean value of churned and not churned customer 
print(churn_data.groupby(["Churn"])['eve.mins'].mean())


# In[70]:


#Print the mean value of churned and not churned customer 
print(churn_data.groupby(["Churn"])['eve.charge'].mean())


# In[71]:


#show the relation using scatter plot
sns.scatterplot(x="eve.mins", y="eve.charge", hue="Churn", data=churn_data,palette='hls')
plt.show()


# * Most of the customers are happy with the evening charges that we can only see less people churning away because of the evening charge issue 

# * N I G H T

# In[72]:


#Print the mean value of churned and not churned customer 
print(churn_data.groupby(["Churn"])['night.calls'].mean())


# In[73]:


#Print the mean value of churned and not churned customer 
print(churn_data.groupby(["Churn"])['night.mins'].mean())


# In[74]:


#Print the mean value of churned and not churned customer 
print(churn_data.groupby(["Churn"])['night.charge'].mean())


# In[75]:


#show the relation of  using scatter plot
sns.scatterplot(x="night.mins", y="night.charge", hue="Churn", data=churn_data,palette='hls')
plt.show()


# * I N T E R N A T I O N A L 

# In[76]:


#Print the mean value of churned and not churned customer 
print(churn_data.groupby(["Churn"])['intl.calls'].mean())


# In[77]:


#Print the mean value of churned and not churned customer 
print(churn_data.groupby(["Churn"])['intl.mins'].mean())


# In[78]:


#Print the mean value of churned and not churned customer 
print(churn_data.groupby(["Churn"])['intl.charge'].mean())


# In[79]:


#show the relation of  using scatter plot
sns.scatterplot(x="intl.mins", y="intl.charge", hue="Churn", data=churn_data,palette='hls')
plt.show()


# In[80]:


# converting call charge by category to call minutes   
daychrg_perminute = churn_data['day.charge'].mean()/churn_data['day.mins'].mean()
evechrg_perminute = churn_data['eve.charge'].mean()/churn_data['eve.mins'].mean()
nightchrg_perminute = churn_data['night.charge'].mean()/churn_data['night.mins'].mean()
intlchrg_perminute= churn_data['intl.charge'].mean()/churn_data['intl.mins'].mean()


# In[81]:


print("Day Charge Per Minute :",round(daychrg_perminute,3))
print("Evening Charge Per Minute :",round(evechrg_perminute,3))
print("Night Charge Per Minute :",round(nightchrg_perminute,3))
print("International Charge Per Minute :",round(intlchrg_perminute,3))


# In[82]:


sns.barplot(x=['Day','Evening','Night','International'],y=[daychrg_perminute,evechrg_perminute,nightchrg_perminute,intlchrg_perminute])
plt.show()


# * As the graph shows international plans charges highly so that there is a possibility that customers who use internaltional calls to churn out
# * And also daycharges is also high compared to others

# ### Displot for each numerical column present in the data set

# In[83]:


dist1=churn_data.select_dtypes(exclude=['object','bool'])
for column in dist1:
        plt.figure(figsize=(17,1))
        sns.displot(data=dist1, x=column)

plt.show()


# #### Data Visualization End

# In[84]:


import streamlit as st


# In[ ]:




