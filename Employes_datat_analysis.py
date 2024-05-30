#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv("employee_survey_data.csv")


# In[3]:


df.head(50)


# In[4]:


print(df.describe())


# In[5]:


print(df.isnull().sum())


# In[10]:


# Plot distributions
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.histplot(df['EnvironmentSatisfaction'], kde=True, label='Environment Satisfaction')
sns.histplot(df['JobSatisfaction'], kde=True, label='Job Satisfaction', color='red')
sns.histplot(df['WorkLifeBalance'], kde=True, label='Work Life Balance', color='green')
plt.legend()
plt.title('Distribution of Satisfaction Metrics')
plt.show()

# Pairplot to visualize relationships
sns.pairplot(df[['EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance']])
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[['EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[11]:


plt.figure(figsize=(12, 8))

# Environment vs Job Satisfaction
plt.subplot(2, 2, 1)
plt.scatter(df['EnvironmentSatisfaction'], df['JobSatisfaction'], color='blue')
plt.title('Environment vs Job Satisfaction')
plt.xlabel('Environment Satisfaction')
plt.ylabel('Job Satisfaction')

# Environment vs Work-Life Balance
plt.subplot(2, 2, 2)
plt.scatter(df['EnvironmentSatisfaction'], df['WorkLifeBalance'], color='green')
plt.title('Environment vs Work-Life Balance')
plt.xlabel('Environment Satisfaction')
plt.ylabel('Work-Life Balance')

# Job Satisfaction vs Work-Life Balance
plt.subplot(2, 2, 3)
plt.scatter(df['JobSatisfaction'], df['WorkLifeBalance'], color='red')
plt.title('Job Satisfaction vs Work-Life Balance')
plt.xlabel('Job Satisfaction')
plt.ylabel('Work-Life Balance')

plt.tight_layout()
plt.show()


# In[12]:


import numpy as np

correlation_matrix = df[['EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance']].corr()

plt.figure(figsize=(8, 6))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none', aspect='auto')
plt.colorbar()
plt.xticks(range(len(correlation_matrix)), correlation_matrix.columns, rotation=45)
plt.yticks(range(len(correlation_matrix)), correlation_matrix.columns)
plt.title('Correlation Matrix')

for i in range(len(correlation_matrix)):
    for j in range(len(correlation_matrix)):
        plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', ha='center', va='center', color='black')

plt.show()


# In[13]:


# Correlation matrix
correlation_matrix = df[['EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance']].corr()
print(correlation_matrix)

# Mean and standard deviation
mean_satisfaction = df[['EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance']].mean()
std_satisfaction = df[['EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance']].std()
print(f"Mean Satisfaction Scores:\n{mean_satisfaction}")
print(f"Standard Deviation of Satisfaction Scores:\n{std_satisfaction}")


# In[14]:


# Summary statistics for each satisfaction metric
summary_stats = df[['EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance']].describe()
print(summary_stats)

# Visualize distributions with histograms
df[['EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance']].hist(bins=5, figsize=(12, 6))
plt.show()


# In[15]:


from scipy.stats import f_oneway

# One-way ANOVA
f_stat, p_val = f_oneway(df['EnvironmentSatisfaction'], df['JobSatisfaction'], df['WorkLifeBalance'])
print(f"ANOVA F-statistic: {f_stat}, p-value: {p_val}")

if p_val < 0.05:
    print("There is a significant difference between the satisfaction metrics.")
else:
    print("There is no significant difference between the satisfaction metrics.")


# In[16]:


plt.figure(figsize=(12, 6))

# Plot Environment Satisfaction over EmployeeID
plt.subplot(1, 3, 1)
plt.plot(df['EmployeeID'], df['EnvironmentSatisfaction'], marker='o')
plt.title('Environment Satisfaction over EmployeeID')
plt.xlabel('EmployeeID')
plt.ylabel('Rating')

# Plot Job Satisfaction over EmployeeID
plt.subplot(1, 3, 2)
plt.plot(df['EmployeeID'], df['JobSatisfaction'], marker='o', color='red')
plt.title('Job Satisfaction over EmployeeID')
plt.xlabel('EmployeeID')
plt.ylabel('Rating')

# Plot Work-Life Balance over EmployeeID
plt.subplot(1, 3, 3)
plt.plot(df['EmployeeID'], df['WorkLifeBalance'], marker='o', color='green')
plt.title('Work-Life Balance over EmployeeID')
plt.xlabel('EmployeeID')
plt.ylabel('Rating')

plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




