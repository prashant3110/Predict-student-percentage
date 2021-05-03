#!/usr/bin/env python
# coding: utf-8

# # Name - Prashant
# 
# # The Spark Foundation - Data Science & Business Analytics Internship.
# # TASK-1 : Prediction Using Supervised Machine Learning.

# In this task it is required to predict the percentage of a student based on number of study hours using the Linear Regression supervised Machine Learning Algorithm.

# # Steps :
#  1. Importing the dataset.
#  2. Visualizing the dataset.
#  3. Data preparation.
#  4. Training the algorithm.
#  5. Visualizing the model.
#  6. Making predictions.
#  7. Evaluting the model.

# In[3]:


# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # step 1: importing the dataset

# In[4]:


# reading data from remote link
url="http://bit.ly/w-data"
df=pd.read_csv(url)


# In[5]:


df


# In[6]:


df.head()


# # Step 2: Visualizing the Dataset.

# In[7]:


df.plot(x='Hours',y='Scores',style='o')
plt.xlabel("No. of Hours Studied")
plt.ylabel("percentage scored")
plt.title("Hours VS Percentage")
plt.grid()
plt.show()


# From the above graph, we can observe that there is a linear relation between "Hours Studied" and "Percentage Scored". So, we can use the linear regression supervised machine learning model.

# # Step 3: Data preparation
# 
# we will divide the data into "attributes"(input) and "labels"(output).After that we will split the whole dataset into 2 parts - testing data and training data.

# In[8]:


# dividing the data into "attributes"(inputs) and "labels"(outputs).
X=df.iloc[:,:-1].values
Y=df.iloc[:,1].values


# In[10]:


from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


# In[16]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,Y_train)

print("Training is completed.")


# # Step 5: Visualizing the model
# 

# In[17]:


# plotting the regression line
line= model.coef_*X+model.intercept_

# plotting for the trainig data
plt.scatter(X_train,Y_train)
plt.plot(X,line,color="green");
plt.grid()
plt.show()


# In[18]:


# plotting for the testing data
plt.scatter(X_test,Y_test)
plt.plot(X,line,color="green");
plt.xlabel("Hours Studied")
plt.ylabel("Percentage Scored")
plt.grid()
plt.show()


# # Step 6: Making Predicitions

# In[19]:


Y_predicted =model.predict(X_test) # predicting the scores


# In[23]:


# comparing Actual with Predicted
df2=pd.DataFrame({'Actual': Y_test ,'Predicted':Y_predicted})


# In[24]:


df2


# In[32]:


# Now we can also test with our own data.
hours=9.25
own_pred=model.predict([[hours]])
print('No. of Hours={}'.format(hours))
print("The predicted percentage score if a student studied for",hours,"hours is",format(own_pred[0]))


# # Step 7 :Evaluting the model
# In the last step , we are going to evalute pur trained model by calculating mean absolute error

# In[33]:


from sklearn import metrics
print("Mean Absolute error:",metrics.mean_absolute_error(Y_test,Y_predicted))


# In[ ]:




