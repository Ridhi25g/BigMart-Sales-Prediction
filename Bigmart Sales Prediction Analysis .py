#!/usr/bin/env python
# coding: utf-8

# 
# # <font color=blue>BigMart Sales Prediction</font>

# ## Dataset Information
# 

# The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. Also, certain attributes of each product and store have been defined. The aim is to build a predictive model and find out the sales of each product at a particular store.
# 
# Using this model, BigMart will try to understand the properties of products and stores which play a key role in increasing sales.
# 
# 
# Variable | Description
# ----------|--------------
# Item_Identifier | Unique product ID
# Item_Weight | Weight of product
# Item_Fat_Content | Whether the product is low fat or not
# Item_Visibility | The % of total display area of all products in a    store allocated to the particular product
# Item_Type | The category to which the product belongs
# Item_MRP | Maximum Retail Price (list price) of the product
# Outlet_Identifier | Unique store ID
# Outlet_Establishment_Year | The year in which store was established
# Outlet_Size | The size of the store in terms of ground area covered
# Outlet_Location_Type | The type of city in which the store is located
# Outlet_Type | Whether the outlet is just a grocery store or some sort of supermarket
# Item_Outlet_Sales | Sales of the product in the particulat store. This is the outcome variable to be predicted.

# ## Import modules

# In[48]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# ## Loading the dataset

# In[3]:


df = pd.read_csv('Train.csv')
df.head()


# In[4]:


# statistical info
df.describe()


# In[5]:


# datatype of attributes
df.info()


# In[6]:


# check unique values in dataset
df.apply(lambda x: len(x.unique()))


# ## Preprocessing the dataset

# In[7]:


# check for null values
df.isnull().sum()


# In[8]:


# check for categorical attributes
cat_col = []
for x in df.dtypes.index:
    if df.dtypes[x] == 'object':
        cat_col.append(x)
cat_col


# In[9]:


cat_col.remove('Item_Identifier')
cat_col.remove('Outlet_Identifier')
cat_col


# In[10]:


# print the categorical columns
for col in cat_col:
    print(col)
    print(df[col].value_counts())
    print()


# In[11]:


# fill the missing values
item_weight_mean = df.pivot_table(values = "Item_Weight", index = 'Item_Identifier')
item_weight_mean


# In[12]:


miss_bool = df['Item_Weight'].isnull()
miss_bool


# In[20]:


for i, item in enumerate(df['Item_Identifier']):
    if miss_bool[i]:
        if item in item_weight_mean:
            df['Item_Weight'][i] = item_weight_mean.loc[item]['Item_Weight']
        else:
            df['Item_Weight'][i] = np.mean(df['Item_Weight'])


# In[16]:


df['Item_Weight'].isnull().sum()


# In[17]:


outlet_size_mode = df.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))
outlet_size_mode


# In[18]:


miss_bool = df['Outlet_Size'].isnull()
df.loc[miss_bool, 'Outlet_Size'] = df.loc[miss_bool, 'Outlet_Type'].apply(lambda x: outlet_size_mode[x])


# In[19]:


df['Outlet_Size'].isnull().sum()


# In[21]:


sum(df['Item_Visibility']==0)


# In[22]:


# replace zeros with mean
df.loc[:, 'Item_Visibility'].replace([0], [df['Item_Visibility'].mean()], inplace=True)


# In[23]:


sum(df['Item_Visibility']==0)


# In[24]:


# combine item fat content
df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF':'Low Fat', 'reg':'Regular', 'low fat':'Low Fat'})
df['Item_Fat_Content'].value_counts()


# ## Creation of New Attributes

# In[25]:


df['New_Item_Type'] = df['Item_Identifier'].apply(lambda x: x[:2])
df['New_Item_Type']


# In[26]:


df['New_Item_Type'] = df['New_Item_Type'].map({'FD':'Food', 'NC':'Non-Consumable', 'DR':'Drinks'})
df['New_Item_Type'].value_counts()


# In[27]:


df.loc[df['New_Item_Type']=='Non-Consumable', 'Item_Fat_Content'] = 'Non-Edible'
df['Item_Fat_Content'].value_counts()


# In[28]:


# create small values for establishment year
df['Outlet_Years'] = 2013 - df['Outlet_Establishment_Year']


# In[29]:


df['Outlet_Years']


# In[30]:


df.head()


# ## Exploratory Data Analysis

# In[31]:


sns.distplot(df['Item_Weight'])


# In[32]:


sns.distplot(df['Item_Visibility'])


# In[33]:


sns.distplot(df['Item_MRP'])


# In[34]:


sns.distplot(df['Item_Outlet_Sales'])


# In[35]:


# log transformation
df['Item_Outlet_Sales'] = np.log(1+df['Item_Outlet_Sales'])


# In[36]:


sns.distplot(df['Item_Outlet_Sales'])


# In[37]:


sns.countplot(df["Item_Fat_Content"])


# In[57]:


# plt.figure(figsize=(15,5))
l = list(df['Item_Type'].unique())
chart = sns.countplot(df["Item_Type"])
chart.set_xticklabels(labels=l, rotation=90)


# In[58]:


sns.countplot(df['Outlet_Establishment_Year'])


# In[59]:


sns.countplot(df['Outlet_Size'])


# In[60]:


sns.countplot(df['Outlet_Location_Type'])


# In[61]:


sns.countplot(df['Outlet_Type'])


# ## Coorelation Matrix
# 
# 

# In[62]:


corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')


# In[63]:


df.head()


# ## Label Encoding

# In[64]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Outlet'] = le.fit_transform(df['Outlet_Identifier'])
cat_col = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'New_Item_Type']
for col in cat_col:
    df[col] = le.fit_transform(df[col])


# ## Onehot Encoding

# In[65]:


df = pd.get_dummies(df, columns=['Item_Fat_Content', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'New_Item_Type'])
df.head()


# ## Input Split

# In[66]:


X = df.drop(columns=['Outlet_Establishment_Year', 'Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'])
y = df['Item_Outlet_Sales']


# ## Model Training

# In[77]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
def train(model, X, y):
    # train the model
    model.fit(X, y)
    
    # predict the training set
    pred = model.predict(X)
    
    # perform cross-validation
    cv_score = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
    cv_score = np.abs(np.mean(cv_score))
    
    print("Model Report")
    print("MSE:",mean_squared_error(y,pred))
    print("CV Score:", cv_score)


# ## Linear Regression

# In[78]:


from sklearn.linear_model import LinearRegression, Ridge, Lasso
model = LinearRegression(normalize=True)
train(model, X, y)
coef = pd.Series(model.coef_, X.columns).sort_values()
coef.plot(kind='bar', title="Model Coefficients")


# ## Ridge Regression

# In[79]:


model = Ridge(normalize=True)
train(model, X, y)
coef = pd.Series(model.coef_, X.columns).sort_values()
coef.plot(kind='bar', title="Model Coefficients")


# ## Lasso Regression

# In[81]:


model = Lasso()
train(model, X, y)
coef = pd.Series(model.coef_, X.columns).sort_values()
coef.plot(kind='bar', title="Model Coefficients")


# Attained an impressive Cross-Validation score through proficient application of Lasso Regression.
