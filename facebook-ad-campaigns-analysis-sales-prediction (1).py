#!/usr/bin/env python
# coding: utf-8

# # **Sales Conversion Optimization by analysing ad campaigns Tracked by Facebook and Prediction using Random Forest Regressor**

# **Data:**
# 
# The data used in this project is from an anonymous organisation’s social media ad campaign.
# 
# 1.) ad_id: an unique ID for each ad.
# 
# 2.) xyzcampaignid: an ID associated with each ad campaign of XYZ company.
# 
# 3.) fbcampaignid: an ID associated with how Facebook tracks each campaign.
# 
# 4.) age: age of the person to whom the ad is shown.
# 
# 5.) gender: gender of the person to whim the add is shown
# 
# 6.) interest: a code specifying the category to which the person’s interest belongs (interests are as mentioned in the person’s Facebook public profile).
# 
# 7.) Impressions: the number of times the ad was shown.
# 
# 8.) Clicks: number of clicks on for that ad.
# 
# 9.) Spent: Amount paid by company xyz to Facebook, to show that ad.
# 
# 10.) Total conversion: Total number of people who enquired about the product after seeing the ad.
# 
# 11.) Approved conversion: Total number of people who bought the product after seeing the ad.

# **Aim:** 
# 
# To optimize Sales conversion and predict future sales

# **Approach:** 
# 
# Exploratory Data Analysis using matplotlib and seaborn and model training using Random Forest Regressor

# # **Importing Libraries**

# In[ ]:


import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # **Loading Data**

# In[55]:


df=pd.read_csv("/kaggle/input/clicks-conversion-tracking/KAG_conversion_data.csv")


# In[56]:


df.head()


# **Checking for null values**

# In[57]:


df.info()


# # **Exploratory Data Analysis**

# In[58]:


df.shape


# In[59]:


df.describe()


# In[60]:


import matplotlib.pyplot as plt
import seaborn as sns


# **Correlation Matrix**

# In[61]:


g=sns.heatmap(df[["Impressions","Clicks","Spent","Total_Conversion","Approved_Conversion"]].corr()
              ,annot=True ,fmt=".2f", cmap="coolwarm")


# Here it's clear, "Impressions" and "Total_Conversion" are more correlated with "Approved_Conversion" than "Clicks" and "Spent". 

# # **Campaigns**

# In[62]:


df["xyz_campaign_id"].unique()


# Here, we see there are 3 different ad campaigns for xyz company.
# 
# Now we'll replace their names with campaign_a, campaign_b and campaign_c for better visualisation which creates problem with integer values

# In[63]:


df["xyz_campaign_id"].replace({916:"campaign_a",936:"campaign_b",1178:"campaign_c"}, inplace=True)


# In[64]:


df.head()


# In[65]:


# count plot on single categorical variable 
sns.countplot(x ='xyz_campaign_id', data = df) 
# Show the plot 
plt.show() 


# This shows campaign_c has most number of ads.

# In[66]:


#Approved_Conversion
# Creating our bar plot
plt.bar(df["xyz_campaign_id"], df["Approved_Conversion"])
plt.ylabel("Approved_Conversion")
plt.title("company vs Approved_Conversion")
plt.show()


# It's clear from both the above graphs that compaign_c has better Approved_conversion count, i.e. most people bought products in campaign_c.
# 
# Let's see the distribution with age.

# **Age**

# In[67]:


# count plot on single categorical variable 
sns.countplot(x ='age', data = df) 
# Show the plot 
plt.show() 


# In[68]:


import seaborn as sns
sns.set(style="whitegrid")
tips = sns.load_dataset("tips")
sns.barplot(x=df["xyz_campaign_id"], y=df["Approved_Conversion"], hue=df["age"], data=tips)


# It's interesting to note that  in campaign_c and campaign_b, the age group of 30-34 shows more interest, whereas in campaign_a the age group of 40-44 shows more interest.
# 
# Let's see the distribution with gender.

# **Gender**

# In[69]:


# count plot on single categorical variable 
sns.countplot(x ='gender', data = df) 
# Show the plot 
plt.show() 


# In[70]:


import seaborn as sns
sns.set(style="whitegrid")
tips = sns.load_dataset("tips")
sns.barplot(x=df["xyz_campaign_id"], y=df["Approved_Conversion"], hue=df["gender"], data=tips)


# Both the genders shows similar interests in all three campaigns.

# **Interest**

# In[71]:


# count plot on single categorical variable 
fig_dims = (15,6)
fig, ax = plt.subplots(figsize=fig_dims)
sns.countplot(x ='interest', data = df) 
# Show the plot 
plt.show() 


# In[72]:


plt.scatter(df["interest"], df["Approved_Conversion"])
plt.title("interest vs. Approved_Conversion")
plt.xlabel("interest")
plt.ylabel("Approved_Conversion")
plt.show()


# It's interesting to note that, although the count of interest after 100 is less,there is a rise of users after 100 who actually bought the product. Rest of the distribution is according to what was expected.

# In[73]:


g = sns.FacetGrid(df, col="gender")
g.map(plt.scatter, "interest", "Approved_Conversion", alpha=.4)
g.add_legend();


# In[74]:


g = sns.FacetGrid(df, col="age")
g.map(plt.scatter, "interest", "Approved_Conversion", alpha=.4)
g.add_legend();


# **Spent**

# In[75]:


plt.hist(df['Spent'], bins = 25)
plt.xlabel("Spent")
plt.ylabel("Frequency")
plt.show()


# In[76]:


plt.scatter(df["Spent"], df["Approved_Conversion"])
plt.title("Spent vs. Approved_Conversion")
plt.xlabel("Spent")
plt.ylabel("Approved_Conversion")
plt.show()


# We can see, as the amount of money spent increases, no of product bought increases.

# In[77]:


g = sns.FacetGrid(df, col="gender")
g.map(plt.scatter, "Spent", "Approved_Conversion", alpha=.4)
g.add_legend();


# In[78]:


g = sns.FacetGrid(df, col="age")
g.map(plt.scatter, "Spent", "Approved_Conversion", alpha=.4)
g.add_legend();


# **Impressions**

# In[79]:


plt.hist(df['Impressions'], bins = 25)
plt.xlabel("Impressions")
plt.ylabel("Frequency")
plt.show()


# In[80]:


plt.scatter(df["Impressions"], df["Approved_Conversion"])
plt.title("Impressions vs. Approved_Conversion")
plt.xlabel("Impressions")
plt.ylabel("Approved_Conversion")
plt.show()


# There is a sudden rise in Approved conversions after a certain point in Impressions.

# # **People who actually bought the product**

# **After Clicking the ad ?**

# Let's see people who actually went from clicking to buying the product.

# In[81]:


g = sns.FacetGrid(df, col="gender")
g.map(plt.scatter, "Clicks", "Approved_Conversion", alpha=.4)
g.add_legend();


# It seems men tend to click more than women but women buy more products than men after clicking the add.

# In[82]:


g = sns.FacetGrid(df, col="age")
g.map(plt.scatter, "Clicks", "Approved_Conversion", alpha=.4)
g.add_legend();


# People in age group 30-34 has more tendency to buy product after clicking the add.

# **After enquiring the product?**

# Let's see people who actually went from enquiring to buying the product.

# In[83]:


g = sns.FacetGrid(df, col="gender")
g.map(plt.scatter, "Total_Conversion", "Approved_Conversion", alpha=.4)
g.add_legend();


# It seems women buys more products than men after enquiring the product. However men tends to enquire more about the product.

# In[84]:


g = sns.FacetGrid(df, col="age")
g.map(plt.scatter, "Total_Conversion", "Approved_Conversion",alpha=.5)
g.add_legend()


# It seems people in age group 30-34 are more likely to buy the product after enquiring the product.

# # **Zooming into campaign_c(campaign with most approved_conversion)**

# In[85]:


a=[]
b=[]
c=[]
for i,j,k in zip(df.xyz_campaign_id, df.fb_campaign_id, df.Approved_Conversion):
    if i=="campaign_c":
      a.append(i),b.append(j),c.append(k)


# In[86]:


d={'campaign_name':a, 'fb_campaign_id':b, 'Approved_Conversion':c}     
campaign_c=pd.DataFrame(d)
campaign_c.head()


# **Distribution of fb_campaign_id with Approved_Conversion for campaign_c**

# In[87]:


plt.figure(figsize=(20,5))
plt.scatter(campaign_c["fb_campaign_id"], campaign_c["Approved_Conversion"])
plt.title("fb_campaign_id vs. Approved_Conversion for campaign_c")
plt.xlabel("fb_campaign_id")
plt.ylabel("Approved_Conversion")
plt.show()


# We can see fb_campaign_ids around 145000 have more Approved_Conversion than around 180000 for campaign_c

# # **Summary**

# **(Just for reminder : (916,  936, 1178) xyz_campaign_ids were replaced by campaign_a, campaign_b and campaign_c)**

# **Correlations:**
# * "Impressions" and "Total_Conversion" are more correlated with "Approved_Conversion" than "Clicks" and "Spent".
# 
# **Campaign_C:**
# 1. campaign_c has most number of ads.
# 2. compaign_c has better Approved_conversion count, i.e. most people bought products in campaign_c.
# 
# **age_group:**
# 3. In campaign_c and campaign_b, the age group of 30-34 shows more interest, whereas in campaign_a the age group of 40-44 shows more interest.
# 
# **gender:**
# 4. Both the genders shows similar interests in all three campaigns.
# 
# **interest:**
# 5. Although the count of interest after 100 is less,there is a rise of users after 100 who actually bought the product. Rest of the distribution is according to what was expected.
# 
# **money spent:**
# 6. As the amount of money spent increases, no of product bought increases.
# 7. There is a sudden rise in the Approved_Conversion after a certain point in Impressions.
# 
# **Product bought after clicking the ad:**
# 8. It seems men tend to click more than women but women buy more products than men after clicking the add.
# 9. People in age group 30-34 has more tendency to buy product after clicking the add.
# 
# **Product bought after enquiring the ad:**
# 10. It seems women buys more products than men after enquiring the product. However men tends to enquire more about the product.
# 11. It seems people in age group 30-34 are more likely to buy the product after enquiring the product.
# 
# **Instructive_conclusion:**
# 
# 12. For campaign_c, fb_campaign_ids around 145000 have more Approved_Conversion than around 180000
# 
# 

# **Business Questions**
# 
# **1)How to optimize the social ad campaigns for the highest conversion rate possible. (Attain best Reach to Conversion ratios/Click to Conversion ratios)**
# 
# **=>** Since highest conversion rate was attained in campaign_c, we can consider the factors contributed in this campaign:
# 
# * The number of ad counts should be more for better reach.
# 
# * The age group of 30-34 should be the main aim.
# 
# * People with interest types after 100 should be given more attention
# 
# * More the number of times the add is shown i.e. "impression", more approved_conversion rate is achieved.
# 
# **2)Finding the perfect target demographics with the appropriate clickthrough rates**
# 
# * Women tend to buy the product more often after clicking the ad than men.
# * Also the age group 30 to 34 buy the product more often after clicking the ad
# 
# **3)Understanding the ideal turnaround/decision making time per age group to convert and retarget future social campaigns**
# 
# * Age group 30-34 tend to take less decision making time followed by 35 to 39 and 40-44. 
# * Age group 45-49 take the most time to decide.
# 
# **4)Comparing the individual campaign performance so the best creative/campaign can be run again with adjusted audiences.**
# 
# * clearly campaign_c wins the battle due to highest approved_conversion rate.
# * Also campaign_a does pretty well , considering the number of ads it involves. With less no of ads, it has managed to peform better than campaign_b with large no of ads.

# # **Modelling**

# **Replacing xyz_campaign_ids again with actual ids for modelling**

# In[88]:


df["xyz_campaign_id"].replace({"campaign_a":916 ,"campaign_b":936 ,"campaign_c":1178}, inplace=True)


# **Encoding the Labels 'gender' and 'age' for better modelling**

# In[89]:


#encoding gender
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
encoder.fit(df["gender"])
df["gender"]=encoder.transform(df["gender"])
print(df["gender"])


# In[90]:


#encoding age
encoder.fit(df["age"])
df["age"]=encoder.transform(df["age"])
print(df["age"])


# In[91]:


df.head()


# **Removing "Approved_Conversion" and "Total_Conversion" from dataset**

# In[92]:


x=np.array(df.drop(labels=["Approved_Conversion","Total_Conversion"], axis=1))
y=np.array(df["Total_Conversion"])


# In[93]:


x


# In[94]:


y


# In[95]:


y=y.reshape(len(y),1)
y


# **Feature Scaling**

# In[96]:


from sklearn.preprocessing import StandardScaler
sc_x= StandardScaler()
x = sc_x.fit_transform(x)


# **splitting Data into testset and trainset**

# In[97]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)


# # **Random Forest Regressor to predict Total_Conversion**

# In[98]:


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 10, random_state = 0)
rfr.fit(x_train, y_train)


# **Predicting Total Conversion in test_set and rounding up values**

# In[99]:


y_pred=rfr.predict(x_test)
y_pred=np.round(y_pred)


# In[100]:


y_pred


# # **Evaluation**

# In[101]:


from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
mae=mean_absolute_error(y_test, y_pred)
mse=mean_squared_error(y_test, y_pred)
rmse=np.sqrt(mse)
r2_score=r2_score(y_test, y_pred)


# In[102]:


mae


# The mean absolute error achieved is 0.99.

# In[103]:


#R-squred value
r2_score


# we have got 0.753 of R-squared value which means 75.3% of the data fits the regression model.

# Please, upvote my work if it could help.
# Thank you!
