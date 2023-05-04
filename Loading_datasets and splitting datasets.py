#!/usr/bin/env python
# coding: utf-8

# In[5]:


from sklearn.datasets import fetch_california_housing

housing_data = fetch_california_housing()
X = housing_data.data # represent the feature matrix
y = housing_data.target # represent the response vector/target

feature_names = housing_data.feature_names
target_names = housing_data.target_names

print('Feature names: ', feature_names)
print('\nTarget names: ', target_names, '(Median house value for households)') 
print("\nFirst 5 rows of X:\n", X[:5])
print('\nShape of dataset', X.shape)


# In[6]:


from sklearn.datasets import fetch_california_housing

housing_data = fetch_california_housing()
X = housing_data.data # represent the feature matrix
y = housing_data.target # represent the response vector/target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1) # split the data

# importing the linearRegression class
from sklearn.linear_model import LinearRegression

regressor = LinearRegression() # instantiate the Linear Regression model
regressor.fit(X_train, y_train) # training the model

# expose the model to new values and predict the target vector
y_predictions = regressor.predict(X_test)
print('Predictions:', y_predictions)
# get the coefficients and intercept
print("Coefficients:\n", regressor.coef_)
print('Intercept:\n', regressor.intercept_)


# In[ ]:





# In[ ]:




