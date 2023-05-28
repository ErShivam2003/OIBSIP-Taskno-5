#!/usr/bin/env python
# coding: utf-8

# # By : Shivam Kumar Patel

# ## Sales Prediction with Python

# ### Importing Libraries

# In[58]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score


# ### Importing Dataset

# In[59]:


df=pd.read_csv(r"C:\Users\sp924\Downloads\Advertising.csv")
df.head()


# In[60]:


df.sample(5)


# In[61]:


df.shape


# In[62]:


df.info()


# In[63]:


df.isnull().sum()


# In[64]:


df.duplicated().sum()


# In[65]:


df=df.drop('Unnamed: 0',axis=1)


# In[66]:


df.describe()


# In[67]:


plt.figure(figsize=(10,15))
plt.subplot(1,3,1)
sns.scatterplot(x='TV',y='Sales',data=df)

plt.subplot(1,3,2)
sns.scatterplot(x='Newspaper',y='Sales',data=df)

plt.subplot(1,3,3)
sns.scatterplot(x='Radio',y='Sales',data=df)

plt.show()


# In[68]:


sns.pairplot(df)


# In[69]:


plt.boxplot(data=df,x=df['Newspaper'])
plt.show()


# In[70]:


df=df[df['Newspaper'] <=100]


# In[71]:


sns.heatmap(df.corr(),annot=True)


# ### Model Training

# In[72]:


x=df.iloc[:,:-1]


# In[73]:


print(x.shape)


# In[74]:


print(x.sample(5))


# In[75]:


y=df.iloc[:,-1]


# In[76]:


print(y.shape)


# In[77]:


print(y.sample(5))


# In[78]:


from sklearn.model_selection import train_test_split


# In[79]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# In[80]:


lr = LinearRegression()


# In[81]:


lr.fit(x_train,y_train)


# In[82]:


y_pred = lr.predict(x_test)


# In[83]:


lr.coef_


# In[84]:


lr.intercept_


# In[85]:


r2_score(y_test,y_pred)*100


# ### Model 2

# In[87]:


rfr =RandomForestRegressor(n_estimators=10,random_state=42)


# In[88]:


rfr.fit(x_train,y_train)


# In[89]:


y_pred_rfr=rfr.predict(x_test)


# In[90]:


y_pred_rfr


# In[91]:


r2_score(y_test,y_pred_rfr)*100

