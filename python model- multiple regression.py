#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:


data_df=pd.read_excel(r"C:\Users\user\Desktop\Folds5x2_pp.xlsx")


# In[3]:


data_df.head()


# In[4]:


x=data_df.drop(["PE"],axis=1).values
y=data_df["PE"].values


# In[5]:


print(x) 


# In[6]:


print(y)


# In[7]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


# In[8]:


from sklearn.linear_model import LinearRegression
ml=LinearRegression()
ml.fit(x_train,y_train)


# In[9]:


y_pred=ml.predict(x_test)
print(y_pred)


# In[10]:


ml.predict([[14.96,41.76,1024.07,73.17]])


# In[11]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[12]:


import matplotlib.pyplot as plt
plt.figure (figsize=(15,10))
plt.scatter(y_test,y_pred)
plt.xlabel("Actual")
plt.ylabel("predicteed")
plt.title("Actual vs. predicted")


# In[15]:


pred_y_df=pd.DataFrame({"Actual Value":y_test,"predicted Value":y_pred,"Difference":y_test-y_pred})
pred_y_df[0:20]


# In[ ]:




