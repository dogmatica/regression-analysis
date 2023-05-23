#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports and housekeeping
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
sns.set_theme(style="darkgrid")


# In[2]:


# Import the main dataset
df = pd.read_csv('Steel_industry_data.csv',dtype={'locationid':np.int64})


# In[3]:


df


# In[4]:


# Display dataset info
df.info()


# In[6]:


# Check data for duplicated rows
df.duplicated().sum()


# In[7]:


# Display summary statistics for entire dataset - continuous variables
df.describe()


# In[8]:


# Display summary statistics for entire dataset - categorical variables
df.describe(include = object)


# In[9]:


# Initialize figure size settings
plt.rcParams['figure.figsize'] = [10, 10]


# In[10]:


# Display histogram plots for distribution of continuous variables
df.hist()


# In[12]:


# Reassign data types
for col in df:
    if df[col].dtypes == 'object':
        df[col] = df[col].astype('category')


# In[13]:


# Use cat.codes for label encoding of 3 categorical variables
df['WeekStatus_cat'] = df['WeekStatus'].cat.codes
df['Day_of_week_cat'] = df['Day_of_week'].cat.codes
df['Load_Type_cat'] = df['Load_Type'].cat.codes


# In[14]:


df.head


# In[15]:


# Display regplots for bivariate statistical analysis of continuous variables - dependent variable = Bandwidth_GB_Year
fig, ax = plt.subplots(figsize = (20,20), ncols = 3, nrows = 3)
sns.regplot(x="Lagging_Current_Reactive.Power_kVarh",
            y="Usage_kWh",
            data=df,
            ax = ax[0][0],
            ci=None)
sns.regplot(x="Leading_Current_Reactive_Power_kVarh",
            y="Usage_kWh",
            data=df,
            ax = ax[0][1],
            ci=None)
sns.regplot(x="CO2(tCO2)",
            y="Usage_kWh",
            data=df,
            ax = ax[0][2],
            ci=None)
sns.regplot(x="Lagging_Current_Power_Factor",
            y="Usage_kWh",
            data=df,
            ax = ax[1][0],
            ci=None)
sns.regplot(x="Leading_Current_Power_Factor",
            y="Usage_kWh",
            data=df,
            ax = ax[1][1],
            ci=None)
sns.regplot(x="NSM",
            y="Usage_kWh",
            data=df,
            ax = ax[1][2],
            ci=None)
sns.regplot(x="WeekStatus_cat",
            y="Usage_kWh",
            data=df,
            ax = ax[2][0],
            ci=None)
sns.regplot(x="Day_of_week_cat",
            y="Usage_kWh",
            data=df,
            ax = ax[2][1],
            ci=None)
sns.regplot(x="Load_Type_cat",
            y="Usage_kWh",
            data=df,
            ax = ax[2][2],
            ci=None)


# In[20]:


# Check the remaining columns for null values
df.isnull().sum()


# In[27]:


df.rename(columns={"Lagging_Current_Reactive.Power_kVarh": "Lagging_Current_Reactive_Power_kVarh", "CO2(tCO2)": "CO2_tCO2"}, inplace = True)


# In[28]:


df.info()


# In[35]:


df.corr()


# In[36]:


# Create initial model and display summary
mdl_Usage_kWh_vs_CO2_tCO2 = ols("Usage_kWh ~ CO2_tCO2", data=df).fit()
print(mdl_Usage_kWh_vs_CO2_tCO2.summary())


# In[39]:


# Display MSE, RSE, Rsquared and Adjusted Rsquared for initial model
mse_all = mdl_Usage_kWh_vs_CO2_tCO2.mse_resid
print('MSE of original model: ', mse_all)
rse_all = np.sqrt(mse_all)
print('RSE of original model: ', rse_all)
print('Rsquared of original model: ', mdl_Usage_kWh_vs_CO2_tCO2.rsquared)
print('Rsquared Adjusted of original model: ', mdl_Usage_kWh_vs_CO2_tCO2.rsquared_adj)


# In[40]:


# Create initial model and display summary
mdl_Usage_kWh_vs_both = ols("Usage_kWh ~ CO2_tCO2 + Lagging_Current_Reactive_Power_kVarh", data=df).fit()
print(mdl_Usage_kWh_vs_both.summary())


# In[41]:


# Display MSE, RSE, Rsquared and Adjusted Rsquared for initial model
mse_all = mdl_Usage_kWh_vs_both.mse_resid
print('MSE of original model: ', mse_all)
rse_all = np.sqrt(mse_all)
print('RSE of original model: ', rse_all)
print('Rsquared of original model: ', mdl_Usage_kWh_vs_both.rsquared)
print('Rsquared Adjusted of original model: ', mdl_Usage_kWh_vs_both.rsquared_adj)


# In[37]:


# Plots for independent variable Children
fig = plt.figure(figsize=(14, 8))
fig = sm.graphics.plot_regress_exog(mdl_Usage_kWh_vs_CO2_tCO2, 'CO2_tCO2', fig=fig)


# In[42]:


# Plots for independent variable Children
fig = plt.figure(figsize=(14, 8))
fig = sm.graphics.plot_regress_exog(mdl_Usage_kWh_vs_both, 'CO2_tCO2', fig=fig)


# In[43]:


# Plots for independent variable Children
fig = plt.figure(figsize=(14, 8))
fig = sm.graphics.plot_regress_exog(mdl_Usage_kWh_vs_both, 'Lagging_Current_Reactive_Power_kVarh', fig=fig)


# In[44]:


# Create initial model and display summary
mdl_CO2_tCO2_vs_both = ols("CO2_tCO2 ~ Usage_kWh + Lagging_Current_Reactive_Power_kVarh", data=df).fit()
print(mdl_CO2_tCO2_vs_both.summary())


# In[45]:


# Display MSE, RSE, Rsquared and Adjusted Rsquared for initial model
mse_all = mdl_CO2_tCO2_vs_both.mse_resid
print('MSE of original model: ', mse_all)
rse_all = np.sqrt(mse_all)
print('RSE of original model: ', rse_all)
print('Rsquared of original model: ', mdl_CO2_tCO2_vs_both.rsquared)
print('Rsquared Adjusted of original model: ', mdl_CO2_tCO2_vs_both.rsquared_adj)


# In[46]:


# Plots for independent variable Children
fig = plt.figure(figsize=(14, 8))
fig = sm.graphics.plot_regress_exog(mdl_CO2_tCO2_vs_both, 'Usage_kWh', fig=fig)


# In[47]:


# Plots for independent variable Children
fig = plt.figure(figsize=(14, 8))
fig = sm.graphics.plot_regress_exog(mdl_CO2_tCO2_vs_both, 'Lagging_Current_Reactive_Power_kVarh', fig=fig)


# In[48]:


# Q-Q plot for final model
sm.qqplot(mdl_CO2_tCO2_vs_both.resid, line='s')


# In[ ]:




