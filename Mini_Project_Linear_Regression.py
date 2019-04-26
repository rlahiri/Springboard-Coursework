#!/usr/bin/env python
# coding: utf-8

# # Regression in Python
# 
# ***
# This is a very quick run-through of some basic statistical concepts, adapted from [Lab 4 in Harvard's CS109](https://github.com/cs109/2015lab4) course. Please feel free to try the original lab if you're feeling ambitious :-) The CS109 git repository also has the solutions if you're stuck.
# 
# * Linear Regression Models
# * Prediction using linear regression
# 
# Linear regression is used to model and predict continuous outcomes with normal random errors. There are nearly an infinite number of different types of regression models and each regression model is typically defined by the distribution of the prediction errors (called "residuals") of the type of data. Logistic regression is used to model binary outcomes whereas Poisson regression is used to predict counts. In this exercise, we'll see some examples of linear regression as well as Train-test splits.
# 
# The packages we'll cover are: `statsmodels`, `seaborn`, and `scikit-learn`. While we don't explicitly teach `statsmodels` and `seaborn` in the Springboard workshop, those are great libraries to know.
# ***

# <img width=600 height=300 src="https://imgs.xkcd.com/comics/sustainable.png"/>
# ***

# In[37]:


# special IPython command to prepare the notebook for matplotlib and other libraries
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn

import seaborn as sns
import pylab

# special matplotlib argument for improved plots
from matplotlib import rcParams
sns.set_style("whitegrid")
sns.set_context("poster")


# ***
# # Part 1: Introduction to Linear Regression
# ### Purpose of linear regression
# ***
# <div class="span5 alert alert-info">
# 
# <p> Given a dataset containing predictor variables $X$ and outcome/response variable $Y$, linear regression can be used to: </p>
# <ul>
#   <li> Build a <b>predictive model</b> to predict future values of $\hat{Y}$, using new data $X^*$ where $Y$ is unknown.</li>
#   <li> Model the <b>strength of the relationship</b> between each independent variable $X_i$ and $Y$</li>
#     <ul>
#       <li> Many times, only a subset of independent variables $X_i$ will have a linear relationship with $Y$</li>
#       <li> Need to figure out which $X_i$ contributes most information to predict $Y$ </li>
#     </ul>
#    <li>It is in many cases, the first pass prediction algorithm for continuous outcomes. </li>
# </ul>
# </div>
# 
# ### A Brief Mathematical Recap
# ***
# 
# [Linear Regression](http://en.wikipedia.org/wiki/Linear_regression) is a method to model the relationship between a set of independent variables $X$ (also knowns as explanatory variables, features, predictors) and a dependent variable $Y$.  This method assumes the relationship between each predictor $X$ is **linearly** related to the dependent variable $Y$. The most basic linear regression model contains one independent variable $X$, we'll call this the simple model. 
# 
# $$ Y = \beta_0 + \beta_1 X + \epsilon$$
# 
# where $\epsilon$ is considered as an unobservable random variable that adds noise to the linear relationship. In linear regression, $\epsilon$ is assumed to be normally distributed with a mean of 0. In other words, what this means is that on average, if we know $Y$, a roughly equal number of predictions $\hat{Y}$ will be above $Y$ and others will be below $Y$. That is, on average, the error is zero. The residuals, $\epsilon$ are also assumed to be "i.i.d.": independently and identically distributed. Independence means that the residuals are not correlated -- the residual from one prediction has no effect on the residual from another prediction. Correlated errors are common in time series analysis and spatial analyses.
# 
# * $\beta_0$ is the intercept of the linear model and represents the average of $Y$ when all independent variables $X$ are set to 0.
# 
# * $\beta_1$ is the slope of the line associated with the regression model and represents the average effect of a one-unit increase in $X$ on $Y$.
# 
# * Back to the simple model. The model in linear regression is the *conditional mean* of $Y$ given the values in $X$ is expressed a linear function.  
# 
# $$ y = f(x) = E(Y | X = x)$$ 
# 
# ![conditional mean](images/conditionalmean.png)
# http://www.learner.org/courses/againstallodds/about/glossary.html
# 
# * The goal is to estimate the coefficients (e.g. $\beta_0$ and $\beta_1$). We represent the estimates of the coefficients with a "hat" on top of the letter.  
# 
# $$ \hat{\beta}_0, \hat{\beta}_1 $$
# 
# * Once we estimate the coefficients $\hat{\beta}_0$ and $\hat{\beta}_1$, we can use these to predict new values of $Y$ given new data $X$.
# 
# $$\hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x_1$$
# 
# * Multiple linear regression is when you have more than one independent variable and the estimation involves matrices
#     * $X_1$, $X_2$, $X_3$, $\ldots$
# 
# 
# * How do you estimate the coefficients? 
#     * There are many ways to fit a linear regression model
#     * The method called **least squares** is the most common method
#     * We will discuss least squares
# 
# $$ Y = \beta_0 + \beta_1 X_1 + \ldots + \beta_p X_p + \epsilon$$ 
#     
# ### Estimating $\hat\beta$: Least squares
# ***
# [Least squares](http://en.wikipedia.org/wiki/Least_squares) is a method that can estimate the coefficients of a linear model by minimizing the squared residuals: 
# 
# $$ \mathscr{L} = \sum_{i=1}^N \epsilon_i^2 = \sum_{i=1}^N \left( y_i - \hat{y}_i \right)^2  = \sum_{i=1}^N \left(y_i - \left(\beta_0 + \beta_1 x_i\right)\right)^2 $$
# 
# where $N$ is the number of observations and $\epsilon$ represents a residual or error, ACTUAL - PREDICTED.  
# 
# #### Estimating the intercept $\hat{\beta_0}$ for the simple linear model
# 
# We want to minimize the squared residuals and solve for $\hat{\beta_0}$ so we take the partial derivative of $\mathscr{L}$ with respect to $\hat{\beta_0}$ 

# Using this new information, we can compute the estimate for $\hat{\beta}_1$ by taking the partial derivative of $\mathscr{L}$ with respect to $\hat{\beta}_1$.

# $
# \begin{align}
# \frac{\partial \mathscr{L}}{\partial \hat{\beta_1}} &= \frac{\partial}{\partial \hat{\beta_1}} \sum_{i=1}^N \epsilon^2 \\
# &= \frac{\partial}{\partial \hat{\beta_1}} \sum_{i=1}^N \left( y_i - \hat{y}_i \right)^2 \\
# &= \frac{\partial}{\partial \hat{\beta_1}} \sum_{i=1}^N \left( y_i - \left( \hat{\beta}_0 + \hat{\beta}_1 x_i \right) \right)^2 \\
# &= 2 \sum_{i=1}^N \left( y_i - \left( \hat{\beta}_0 + \hat{\beta}_1 x_i \right) \right) \left( -x_i \right) \hspace{25mm}\mbox{(by chain rule)} \\
# &= -2 \sum_{i=1}^N x_i \left( y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i \right) \\
# &= -2 \sum_{i=1}^N x_i (y_i - \hat{\beta}_0 x_i - \hat{\beta}_1 x_i^2) \\
# &= -2 \sum_{i=1}^N x_i (y_i - \left( \bar{y} - \hat{\beta}_1 \bar{x} \right) x_i - \hat{\beta}_1 x_i^2) \\
# &= -2 \sum_{i=1}^N (x_i y_i - \bar{y}x_i + \hat{\beta}_1\bar{x}x_i - \hat{\beta}_1 x_i^2) \\
# &= -2 \left[ \sum_{i=1}^N x_i y_i - \bar{y} \sum_{i=1}^N x_i + \hat{\beta}_1\bar{x}\sum_{i=1}^N x_i - \hat{\beta}_1 \sum_{i=1}^N x_i^2 \right] \\
# &= -2 \left[ \hat{\beta}_1 \left\{ \bar{x} \sum_{i=1}^N x_i - \sum_{i=1}^N x_i^2 \right\} + \left\{ \sum_{i=1}^N x_i y_i - \bar{y} \sum_{i=1}^N x_i \right\}\right] \\
# & 2 \left[ \hat{\beta}_1 \left\{ \sum_{i=1}^N x_i^2 - \bar{x} \sum_{i=1}^N x_i \right\} + \left\{ \bar{y} \sum_{i=1}^N x_i - \sum_{i=1}^N x_i y_i \right\} \right] = 0 \\
# & \hat{\beta}_1 = \frac{-\left( \bar{y} \sum_{i=1}^N x_i - \sum_{i=1}^N x_i y_i \right)}{\sum_{i=1}^N x_i^2 - \bar{x}\sum_{i=1}^N x_i} \\
# &= \frac{\sum_{i=1}^N x_i y_i - \bar{y} \sum_{i=1}^N x_i}{\sum_{i=1}^N x_i^2 - \bar{x} \sum_{i=1}^N x_i} \\
# & \boxed{\hat{\beta}_1 = \frac{\sum_{i=1}^N x_i y_i - \bar{x}\bar{y}n}{\sum_{i=1}^N x_i^2 - n \bar{x}^2}}
# \end{align}
# $

# The solution can be written in compact matrix notation as
# 
# $$\hat\beta =  (X^T X)^{-1}X^T Y$$ 
# 
# We wanted to show you this in case you remember linear algebra, in order for this solution to exist we need $X^T X$ to be invertible. Of course this requires a few extra assumptions, $X$ must be full rank so that $X^T X$ is invertible, etc. Basically, $X^T X$ is full rank if all rows and columns are linearly independent. This has a loose relationship to variables and observations being independent respective. **This is important for us because this means that having redundant features in our regression models will lead to poorly fitting (and unstable) models.** We'll see an implementation of this in the extra linear regression example.

# ***
# # Part 2: Exploratory Data Analysis for Linear Relationships
# 
# The [Boston Housing data set](https://archive.ics.uci.edu/ml/datasets/Housing) contains information about the housing values in suburbs of Boston.  This dataset was originally taken from the StatLib library which is maintained at Carnegie Mellon University and is now available on the UCI Machine Learning Repository. 
# 
# 
# ## Load the Boston Housing data set from `sklearn`
# ***
# 
# This data set is available in the [sklearn](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html#sklearn.datasets.load_boston) python module which is how we will access it today.  

# In[38]:


from sklearn.datasets import load_boston
import pandas as pd

boston = load_boston()


# In[39]:


boston.keys()


# In[40]:


boston.data.shape


# In[41]:


# Print column names
print(boston.feature_names)


# In[42]:


# Print description of Boston housing data set
print(boston.DESCR)


# Now let's explore the data set itself. 

# In[43]:


bos = pd.DataFrame(boston.data)
bos.head()


# There are no column names in the DataFrame. Let's add those. 

# In[44]:


bos.columns = boston.feature_names
bos.head()


# Now we have a pandas DataFrame called `bos` containing all the data we want to use to predict Boston Housing prices.  Let's create a variable called `PRICE` which will contain the prices. This information is contained in the `target` data. 

# In[45]:


print(boston.target.shape)


# In[46]:


bos['PRICE'] = boston.target
bos.head()


# ## EDA and Summary Statistics
# ***
# 
# Let's explore this data set.  First we use `describe()` to get basic summary statistics for each of the columns. 

# In[47]:


bos.describe()


# ### Scatterplots
# ***
# 
# Let's look at some scatter plots for three variables: 'CRIM' (per capita crime rate), 'RM' (number of rooms) and 'PTRATIO' (pupil-to-teacher ratio in schools).  

# In[12]:


plt.scatter(bos.CRIM, bos.PRICE)
plt.xlabel("Per capita crime rate by town (CRIM)")
plt.ylabel("Housing Price")
plt.title("Relationship between CRIM and Price")


# <div class="span5 alert alert-info">
# <h3>Part 2 Checkup Exercise Set I</h3>
# 
# <p><b>Exercise:</b> What kind of relationship do you see? e.g. positive, negative?  linear? non-linear? Is there anything else strange or interesting about the data? What about outliers?</p>
# 
# 
# <p><b>Exercise:</b> Create scatter plots between *RM* and *PRICE*, and *PTRATIO* and *PRICE*. Label your axes appropriately using human readable labels. Tell a story about what you see.</p>
# 
# <p><b>Exercise:</b> What are some other numeric variables of interest? Why do you think they are interesting? Plot scatterplots with these variables and *PRICE* (house price) and tell a story about what you see.</p>
# 
# </div>

# **What kind of relationship do you see? Is there anything else strange or interesting about the data? What about outliers?**
# 
# 
# A weak negative linear correlation can be seen between the two variables 'CRIM' and 'PRICE'. Based on the scatterplot alone, one could make the assumption that the crime per capita rate is close to 0 irrespective of housing price which, in reality, is highly unlikely.

# In[13]:


# your turn: scatter plot between *RM* and *PRICE*

plt.scatter(bos.RM, bos.PRICE)
plt.xlabel("Average Number of Rooms Per Dwelling (RM)")
plt.ylabel("Housing Price")
plt.title("Relationship between RM and Price")


# There is a positive linear relationship between the average number of rooms per dwelling and housing price.

# In[14]:


# your turn: scatter plot between *PTRATIO* and *PRICE*

plt.scatter(bos.PTRATIO, bos.PRICE)
plt.xlabel("Pupil-to-Teacher Ratio (PTRATIO)")
plt.ylabel("Housing Price")
plt.title("Relationship between PTRATIO and Price")


# There is a negative linear relationship between pupil-to-teacher ratio and housing price.

# In[15]:


# your turn: create some other scatter plots

## scatter plot between *INDUS* and *PRICE*

plt.scatter(bos.INDUS, bos.PRICE)
plt.xlabel("Proportion of Non-retail Business Acres Per Town (INDUS)")
plt.ylabel("Housing Price")
plt.title("Relationship between INDUS and Price")


# There appears to a weak negative linear relationship between the proportion of non-retail business acres per town and housing price.

# In[16]:


# your turn: create some other scatter plots

## scatter plot between *DIS* and *PRICE*

plt.scatter(bos.DIS, bos.PRICE)
plt.xlabel("Weighted Distances to Five Boston Employment Centres (DIS)")
plt.ylabel("Housing Price")
plt.title("Relationship between DIS and Price")


# There appears to be a weak positive linear relationship between the weighted distances to five Boston employment centres and price.

# ### Scatterplots using Seaborn
# ***
# 
# [Seaborn](https://stanford.edu/~mwaskom/software/seaborn/) is a cool Python plotting library built on top of matplotlib. It provides convenient syntax and shortcuts for many common types of plots, along with better-looking defaults.
# 
# We can also use [seaborn regplot](https://stanford.edu/~mwaskom/software/seaborn/tutorial/regression.html#functions-to-draw-linear-regression-models) for the scatterplot above. This provides automatic linear regression fits (useful for data exploration later on). Here's one example below.

# In[17]:


sns.regplot(y="PRICE", x="RM", data=bos, fit_reg = True)


# In[18]:


sns.regplot(y="PRICE", x="INDUS", data=bos, fit_reg = True)


# ### Histograms
# ***
# 

# In[19]:


plt.hist(np.log(bos.CRIM))
plt.title("CRIM")
plt.xlabel("Crime rate per capita")
plt.ylabel("Frequency")
plt.show()


# <div class="span5 alert alert-info">
# <h3>Part 2 Checkup Exercise Set II</h3>
# 
# <p><b>Exercise:</b> In the above histogram, we took the logarithm of the crime rate per capita. Repeat this histogram without taking the log. What was the purpose of taking the log? What do we gain by making this transformation? What do you now notice about this variable that is not obvious without making the transformation?
# 
# <p><b>Exercise:</b> Plot the histogram for *RM* and *PTRATIO* against each other, along with the two variables you picked in the previous section. We are looking for correlations in predictors here.</p>
# </div>

# **In the above histogram, we took the logarithm of the crime rate per capita. Repeat this histogram without taking the log.**

# In[20]:


#your turn

plt.hist(bos.CRIM)
plt.title("CRIM")
plt.xlabel("Crime rate per capita")
plt.ylabel("Frequency")
plt.show()


# **What was the purpose of taking the log? What do we gain by making this transformation? What do you now notice about this variable that is not obvious without making the transformation?**
# 
# The purpose of taking the log was to avoid the issue of a positive skew in the graph. By making the logarithmic transformation, it becomes clear that a significant portion of the data falls below 0% crime rate per capita, while also following a more normal distribution. 

# **Plot the histogram for RM and PTRATIO against each other, along with the two variables you picked in the previous section. We are looking for correlations in predictors here.**

# In[21]:


# Histogram for 'RM' vs. 'PTRATIO'

_ = plt.hist(bos.RM)
_ = plt.hist(bos.PTRATIO)

plt.title("Average Number of Rooms Per Dwelling (RM) vs. Pupil-to-Teacher Ratio (PTRATIO)")
plt.ylabel("Frequency")
plt.legend(["RM", "PTRATIO"], loc=9)

plt.show()


# In[22]:


# Histogram for 'RM' vs. 'PTRATIO'

_ = plt.hist(bos.INDUS)
_ = plt.hist(bos.DIS)

plt.title("Proportion of Non-retail Business Acres Per Town (INDUS) vs. Weighted Distances to Five Boston Employment Centres (DIS)")
plt.ylabel("Frequency")
plt.legend(["INDUS", "DIS"], loc=9)

plt.show()


# ## Part 3: Linear Regression with Boston Housing Data Example
# ***
# 
# Here, 
# 
# $Y$ = boston housing prices (called "target" data in python, and referred to as the dependent variable or response variable)
# 
# and
# 
# $X$ = all the other features (or independent variables, predictors or explanatory variables)
# 
# which we will use to fit a linear regression model and predict Boston housing prices. We will use the least-squares method to estimate the coefficients.  

# We'll use two ways of fitting a linear regression. We recommend the first but the second is also powerful in its features.

# ### Fitting Linear Regression using `statsmodels`
# ***
# [Statsmodels](http://statsmodels.sourceforge.net/) is a great Python library for a lot of basic and inferential statistics. It also provides basic regression functions using an R-like syntax, so it's commonly used by statisticians. While we don't cover statsmodels officially in the Data Science Intensive workshop, it's a good library to have in your toolbox. Here's a quick example of what you could do with it. The version of least-squares we will use in statsmodels is called *ordinary least-squares (OLS)*. There are many other versions of least-squares such as [partial least squares (PLS)](https://en.wikipedia.org/wiki/Partial_least_squares_regression) and [weighted least squares (WLS)](https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares).

# In[48]:


# Import regression modules
import statsmodels.api as sm
from statsmodels.formula.api import ols


# In[49]:


# statsmodels works nicely with pandas dataframes
# The thing inside the "quotes" is called a formula, a bit on that below
m = ols('PRICE ~ RM',bos).fit()
print(m.summary())


# #### Interpreting coefficients
# 
# There is a ton of information in this output. But we'll concentrate on the coefficient table (middle table). We can interpret the `RM` coefficient (9.1021) by first noticing that the p-value (under `P>|t|`) is so small, basically zero. This means that the number of rooms, `RM`, is a statistically significant predictor of `PRICE`. The regression coefficient for `RM` of 9.1021 means that *on average, each additional room is associated with an increase of $\$9,100$ in house price net of the other variables*. The confidence interval gives us a range of plausible values for this average change, about ($\$8,279, \$9,925$), definitely not chump change. 
# 
# In general, the $\hat{\beta_i}, i > 0$ can be interpreted as the following: "A one unit increase in $x_i$ is associated with, on average, a $\hat{\beta_i}$ increase/decrease in $y$ net of all other variables."
# 
# On the other hand, the interpretation for the intercept, $\hat{\beta}_0$ is the average of $y$ given that all of the independent variables $x_i$ are 0.

# ####  `statsmodels` formulas
# ***
# This formula notation will seem familiar to `R` users, but will take some getting used to for people coming from other languages or those who are new to statistics.
# 
# The formula gives instruction for a general structure for a regression call. For `statsmodels` (`ols` or `logit`) calls you need to have a Pandas dataframe with column names that you will add to your formula. In the below example you need a pandas data frame that includes the columns named (`Outcome`, `X1`,`X2`, ...), but you don't need to build a new dataframe for every regression. Use the same dataframe with all these things in it. The structure is very simple:
# 
# `Outcome ~ X1`
# 
# But of course we want to to be able to handle more complex models, for example multiple regression is done like this:
# 
# `Outcome ~ X1 + X2 + X3`
# 
# In general, a formula for an OLS multiple linear regression is
# 
# `Y ~ X1 + X2 + ... + Xp`
# 
# This is the very basic structure but it should be enough to get you through the homework. Things can get much more complex. You can force statsmodels to treat variables as categorical with the `C()` function, call numpy functions to transform data such as `np.log` for extremely-skewed data, or fit a model without an intercept by including `- 1` in the formula. For a quick run-down of further uses see the `statsmodels` [help page](http://statsmodels.sourceforge.net/devel/example_formulas.html).
# 

# Let's see how our model actually fit our data. We can see below that there is a ceiling effect, we should probably look into that. Also, for large values of $Y$ we get underpredictions, most predictions are below the 45-degree gridlines. 

# <div class="span5 alert alert-info">
# <h3>Part 3 Checkup Exercise Set I</h3>
# 
# <p><b>Exercise:</b> Create a scatterplot between the predicted prices, available in `m.fittedvalues` (where `m` is the fitted model) and the original prices. How does the plot look? Do you notice anything interesting or weird in the plot? Comment on what you see.</p>
# </div>

# In[25]:


# your turn

plt.scatter(bos.PRICE, m.fittedvalues)
plt.title("Relationship between Original and Predicted Prices")
plt.xlabel("Original Price")
plt.ylabel("Predicted Price")


# The graph shows a positive linear relationship. There are two points of interest in the plot however. For one, none of the predicted prices at the original price of \\$50,000 fit well, and secondly, there is one case in which the original house price of approximately \$30,000 is predicted as having a price of around -\$10,000. 

# ### Fitting Linear Regression using `sklearn`
# 

# In[50]:


from sklearn.linear_model import LinearRegression
X = bos.drop('PRICE', axis = 1)

# This creates a LinearRegression object
lm = LinearRegression()
lm


# #### What can you do with a LinearRegression object? 
# ***
# Check out the scikit-learn [docs here](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html). We have listed the main functions here. Most machine learning models in scikit-learn follow this same API of fitting a model with `fit`, making predictions with `predict` and the appropriate scoring function `score` for each model.

# Main functions | Description
# --- | --- 
# `lm.fit()` | Fit a linear model
# `lm.predict()` | Predict Y using the linear model with estimated coefficients
# `lm.score()` | Returns the coefficient of determination (R^2). *A measure of how well observed outcomes are replicated by the model, as the proportion of total variation of outcomes explained by the model*

# #### What output can you get?

# In[30]:


# Look inside lm object
# lm.<tab>


# Output | Description
# --- | --- 
# `lm.coef_` | Estimated coefficients
# `lm.intercept_` | Estimated intercept 

# ### Fit a linear model
# ***
# 
# The `lm.fit()` function estimates the coefficients the linear regression using least squares. 

# In[31]:


# Use all 13 predictors to fit linear regression model
lm.fit(X, bos.PRICE)


# <div class="span5 alert alert-info">
# <h3>Part 3 Checkup Exercise Set II</h3>
# 
# <p><b>Exercise:</b> How would you change the model to not fit an intercept term? Would you recommend not having an intercept? Why or why not? For more information on why to include or exclude an intercept, look [here](https://stats.idre.ucla.edu/other/mult-pkg/faq/general/faq-what-is-regression-through-the-origin/).</p>
# 
# <p><b>Exercise:</b> One of the assumptions of the linear model is that the residuals must be i.i.d. (independently and identically distributed). To satisfy this, is it enough that the residuals are normally distributed? Explain your answer.</p>
# 
# <p><b>Exercise:</b> True or false. To use linear regression, $Y$ must be normally distributed. Explain your answer.</p>
# </div>
# 

# **How would you change the model to not fit an intercept term? Would you recommend not having an intercept? Why or why not?**
# 
# In general, the LinearRegression function has the fit_intercept argument which defaults to 'True', unless otherwise specified. To change the model to not fit an intercept term, this argument would have to be changed to 'False'.
# 
# I would not recommend removing the intercept however, because it represents the average of the target variable (y) when all independent variables are set to 0.

# **One of the assumptions of the linear model is that the residuals must be i.i.d. (independently and identically distributed). To satisfy this, is it enough that the residuals are normally distributed? Explain your answer.**
# 
# 

# The residuals do not have to be normally distributed to be independently and identically distributed. 

# **True or false. To use linear regression, Y must be normally distributed. Explain your answer**
# 
# False. The target variable does not have to be normally distributed to use linear regression. 

# ### Estimated intercept and coefficients
# 
# Let's look at the estimated coefficients from the linear model using `1m.intercept_` and `lm.coef_`.  
# 
# After we have fit our linear regression model using the least squares method, we want to see what are the estimates of our coefficients $\beta_0$, $\beta_1$, ..., $\beta_{13}$: 
# 
# $$ \hat{\beta}_0, \hat{\beta}_1, \ldots, \hat{\beta}_{13} $$
# 
# 

# In[32]:


print('Estimated intercept coefficient: {}'.format(lm.intercept_))


# In[33]:


print('Number of coefficients: {}'.format(len(lm.coef_)))


# In[31]:


# The coefficients
pd.DataFrame({'features': X.columns, 'estimatedCoefficients': lm.coef_})[['features', 'estimatedCoefficients']]


# ### Predict Prices 
# 
# We can calculate the predicted prices ($\hat{Y}_i$) using `lm.predict`. 
# 
# $$ \hat{Y}_i = \hat{\beta}_0 + \hat{\beta}_1 X_1 + \ldots \hat{\beta}_{13} X_{13} $$

# In[32]:


# first five predicted prices
lm.predict(X)[0:5]


# <div class="span5 alert alert-info">
# <h3>Part 3 Checkup Exercise Set III</h3>
# 
# <p><b>Exercise:</b> Histogram: Plot a histogram of all the predicted prices. Write a story about what you see. Describe the shape, center and spread of the distribution. Are there any outliers? What might be the reason for them? Should we do anything special with them?</p>
# 
# <p><b>Exercise:</b> Scatterplot: Let's plot the true prices compared to the predicted prices to see they disagree (we did this with `statsmodels` before).</p>
# 
# <p><b>Exercise:</b> We have looked at fitting a linear model in both `statsmodels` and `scikit-learn`. What are the advantages and disadvantages of each based on your exploration? Based on the information provided by both packages, what advantage does `statsmodels` provide?</p>
# </div>

# **Plot a histogram of all the predicted prices. Write a story about what you see. Describe the shape, center and spread of the distribution. Are there any outliers? What might be the reason for them? Should we do anything special with them?**

# In[33]:


np.median(lm.predict(X))


# In[34]:


np.std(lm.predict(X))


# In[35]:


plt.hist(lm.predict(X))
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.title("Predicted Prices")
plt.show()


# The predicted prices follow a normal distribution with a center of 22.1 and standard deviation (spread) of 7.90. The negatively predicted price definitely hints at the presence of an outlier, with collinearity being a reason for this occurrence. There are also thirteen predictor variables at play, which introduces the "Curse of Dimensionality". The best way to avoid this issue is to regularize the data.

# **Let's plot the true prices compared to the predicted prices to see if they disagree (we did this with statsmodels before)**

# In[36]:


plt.scatter(bos.PRICE, lm.predict(X))
plt.title("Relationship between Original and Predicted Prices")
plt.xlabel("Original Price")
plt.ylabel("Predicted Price")


# **We have looked at fitting a linear model in both statsmodels and scikit-learn. What are the advantages and disadvantages of each based on your exploration? Based on the information provided by both packages, what advantage does statsmodels provide?**
# 
# I would say the primary advantage of scikit-learn is its LinearRegression function, which provides a better calculation of predicted values. 

# ### Evaluating the Model: Sum-of-Squares
# 
# The partitioning of the sum-of-squares shows the variance in the predictions explained by the model and the variance that is attributed to error.
# 
# $$TSS = ESS + RSS$$
# 
# #### Residual Sum-of-Squares (aka $RSS$)
# 
# The residual sum-of-squares is one of the basic ways of quantifying how much error exists in the fitted model. We will revisit this in a bit.
# 
# $$ RSS = \sum_{i=1}^N r_i^2 = \sum_{i=1}^N \left(y_i - \left(\beta_0 + \beta_1 x_i\right)\right)^2 $$

# In[37]:


print(np.sum((bos.PRICE - lm.predict(X)) ** 2))


# #### Explained Sum-of-Squares (aka $ESS$)
# 
# The explained sum-of-squares measures the variance explained by the regression model.
# 
# $$ESS = \sum_{i=1}^N \left( \hat{y}_i - \bar{y} \right)^2 = \sum_{i=1}^N \left( \left( \hat{\beta}_0 + \hat{\beta}_1 x_i \right) - \bar{y} \right)^2$$

# In[38]:


print(np.sum(lm.predict(X) - np.mean(bos.PRICE)) ** 2)


# ### Evaluating the Model: The Coefficient of Determination ($R^2$)
# 
# The coefficient of determination, $R^2$, tells us the percentage of the variance in the response variable $Y$ that can be explained by the linear regression model.
# 
# $$ R^2 = \frac{ESS}{TSS} $$
# 
# The $R^2$ value is one of the most common metrics that people use in describing the quality of a model, but it is important to note that *$R^2$ increases artificially as a side-effect of increasing the number of independent variables.* While $R^2$ is reported in almost all statistical packages, another metric called the *adjusted $R^2$* is also provided as it takes into account the number of variables in the model, and can sometimes even be used for non-linear regression models!
# 
# $$R_{adj}^2 = 1 - \left( 1 - R^2 \right) \frac{N - 1}{N - K - 1} = R^2 - \left( 1 - R^2 \right) \frac{K}{N - K - 1} = 1 - \frac{\frac{RSS}{DF_R}}{\frac{TSS}{DF_T}}$$
# 
# where $N$ is the number of observations, $K$ is the number of variables, $DF_R = N - K - 1$ is the degrees of freedom associated with the residual error and $DF_T = N - 1$ is the degrees of the freedom of the total error.

# ### Evaluating the Model: Mean Squared Error and the $F$-Statistic
# ***
# The mean squared errors are just the *averages* of the sum-of-squares errors over their respective degrees of freedom.
# 
# $$MSE = \frac{ESS}{K}$$
# $$MSR = \frac{RSS}{N-K-1}$$
# 
# **Remember: ** Notation may vary across resources particularly the use of *R* and *E* in *RSS/ESS* and *MSR/MSE*. In some resources, E = explained and R = residual. In other resources, E = error and R = regression (explained). **This is a very important distinction that requires looking at the formula to determine which naming scheme is being used.**
# 
# Given the MSR and MSE, we can now determine whether or not the entire model we just fit is even statistically significant. We use an $F$-test for this. The null hypothesis is that all of the $\beta$ coefficients are zero, that is, none of them have any effect on $Y$. The alternative is that *at least one* $\beta$ coefficient is nonzero, but it doesn't tell us which one in a multiple regression:
# 
# $$H_0: \beta_i = 0, \mbox{for all $i$} \\
# H_A: \beta_i > 0, \mbox{for some $i$}$$ 
# 
# $$F = \frac{MSR}{MSE} = \left( \frac{R^2}{1 - R^2} \right) \left( \frac{N - K - 1}{K} \right)$$
#  
# Once we compute the $F$-statistic, we can use the $F$-distribution with $N-K$ and $K-1$ degrees of degrees of freedom to get a p-value.
# 
# **Warning!** The $F$-statistic mentioned in this section is NOT the same as the F1-measure or F1-value discussed in Unit 7.

# <div class="span5 alert alert-info">
# <h3>Part 3 Checkup Exercise Set IV</h3>
# 
# <p>Let's look at the relationship between `PTRATIO` and housing price.</p>
# 
# <p><b>Exercise:</b> Try fitting a linear regression model using only the 'PTRATIO' (pupil-teacher ratio by town) and interpret the intercept and the coefficients.</p>
# 
# <p><b>Exercise:</b> Calculate (or extract) the $R^2$ value. What does it tell you?</p>
# 
# <p><b>Exercise:</b> Compute the $F$-statistic. What does it tell you?</p>
# 
# <p><b>Exercise:</b> Take a close look at the $F$-statistic and the $t$-statistic for the regression coefficient. What relationship do you notice? Note that this relationship only applies in *simple* linear regression models.</p>
# </div>

# **Try fitting a linear regression model using only the 'PTRATIO' (pupil-teacher ratio by town) and interpret the intercept and the coefficients.**

# In[51]:


m_PTR = ols('PRICE ~ PTRATIO',bos).fit()
print(m_PTR.summary())


# The coefficient (-2.157) can be interpreted as follows: when there is a one-unit increase in PTRATIO, house pricing decreases by ~2.2. The intercept (63.345) represents the average house price when PTRATIO is 0.

# **Calculate (or extract) the R2 value. What does it tell you?**

# The R2 value = 0.258. In general, the R2 value is the percentage of variance in the response variable. In this case, R2 signifies that 25.8% of housing price can be explained by PTRATIO.

# **Compute the F-statistic. What does it tell you?**

# The F-statistic helps determine whether the fitted model has statistical significance. In this case, the F-statistic = 175.1 and p = 1.61e-34. As such, the null hypothesis is rejected for the alternate, which states that at least one β coefficient is nonzero, making the model statistically sound. 

# **Take a close look at the F-statistic and the t-statistic for the regression coefficient. What relationship do you notice? Note that this relationship only applies in simple linear regression models.**

# The t-statistic is the square root value of the F-statistic. 

# <div class="span5 alert alert-info">
# <h3>Part 3 Checkup Exercise Set V</h3>
# 
# <p>Fit a linear regression model using three independent variables</p>
# 
# <ol>
# <li> 'CRIM' (per capita crime rate by town)
# <li> 'RM' (average number of rooms per dwelling)
# <li> 'PTRATIO' (pupil-teacher ratio by town)
# </ol>
# 
# <p><b>Exercise:</b> Compute or extract the $F$-statistic. What does it tell you about the model?</p>
# 
# <p><b>Exercise:</b> Compute or extract the $R^2$ statistic. What does it tell you about the model?</p>
# 
# <p><b>Exercise:</b> Which variables in the model are significant in predicting house price? Write a story that interprets the coefficients.</p>
# </div>

# In[52]:


# your turn

m_three_vars = ols('PRICE ~ CRIM + RM + PTRATIO',bos).fit()
print(m_three_vars.summary())


# **Compute or extract the F-statistic. What does it tell you about the model?**
# 
# The F-statistic = 245.2 and p = 6.15e-98. The null hypothesis can rejected - the model is statistically significant.

# **Compute or extract the R2 statistic. What does it tell you about the model?**
# 
# The R2 value = 0.594. This R2 signifies that 59.4% of housing price can be explained by the three predictor variables - CRIM, RM, and PTRATIO.

# **Which variables in the model are significant in predicting house price? Write a story that interprets the coefficients.**
# 
# The p-values for the variables prove that each of the three variables is significant in predicting housing price. The coefficients can be interpreted as follows:
# 
# - When there is a 1% increase in crime rate per capita, house pricing decreases by ~0.21.
# - When there is a one-unit increase in the average number of rooms per dwelling, house pricing increases by ~7.38.
# - When there is a one-unit increase in PTRATIO, house pricing decreases by ~1.07.

# ## Part 4: Comparing Models

# During modeling, there will be times when we want to compare models to see which one is more predictive or fits the data better. There are many ways to compare models, but we will focus on two.

# ### The $F$-Statistic Revisited
# 
# The $F$-statistic can also be used to compare two *nested* models, that is, two models trained on the same dataset where one of the models contains a *subset* of the variables of the other model. The *full* model contains $K$ variables and the *reduced* model contains a subset of these $K$ variables. This allows us to add additional variables to a base model and then test if adding the variables helped the model fit.
# 
# $$F = \frac{\left( \frac{RSS_{reduced} - RSS_{full}}{DF_{reduced} - DF_{full}} \right)}{\left( \frac{RSS_{full}}{DF_{full}} \right)}$$
# 
# where $DF_x = N - K_x - 1$ where $K_x$ is the number of variables in model $x$.

# ### Akaike Information Criterion (AIC)
# 
# Another statistic for comparing two models is AIC, which is based on the likelihood function and takes into account the number of variables in the model.
# 
# $$AIC = 2 K - 2 \log_e{L}$$
# 
# where $L$ is the likelihood of the model. AIC is meaningless in the absolute sense, and is only meaningful when compared to AIC values from other models. Lower values of AIC indicate better fitting models.
# 
# `statsmodels` provides the AIC in its output.

# <div class="span5 alert alert-info">
# <h3>Part 4 Checkup Exercises</h3>
# 
# <p><b>Exercise:</b> Find another variable (or two) to add to the model we built in Part 3. Compute the $F$-test comparing the two models as well as the AIC. Which model is better?</p>
# </div>

# In[53]:


m_four_vars = ols('PRICE ~ CRIM + RM + PTRATIO + INDUS',bos).fit()
print(m_four_vars.summary())


# **Compute the F-test comparing the two models as well as the AIC. Which model is better?**

# This model is better due to a lower AIC value. 

# 
# ## Part 5: Evaluating the Model via Model Assumptions and Other Issues
# ***
# Linear regression makes several assumptions. It is always best to check that these assumptions are valid after fitting a linear regression model.
# 
# <div class="span5 alert alert-danger">
# <ul>
#   <li>**Linearity**. The dependent variable $Y$ is a linear combination of the regression coefficients and the independent variables $X$. This can be verified with a scatterplot of each $X$ vs. $Y$ and plotting correlations among $X$. Nonlinearity can sometimes be resolved by [transforming](https://onlinecourses.science.psu.edu/stat501/node/318) one or more independent variables, the dependent variable, or both. In other cases, a [generalized linear model](https://en.wikipedia.org/wiki/Generalized_linear_model) or a [nonlinear model](https://en.wikipedia.org/wiki/Nonlinear_regression) may be warranted.</li>
#   <li>**Constant standard deviation**. The SD of the dependent variable $Y$ should be constant for different values of X. We can check this by plotting each $X$ against $Y$ and verifying that there is no "funnel" shape showing data points fanning out as $X$ increases or decreases. Some techniques for dealing with non-constant variance include weighted least squares (WLS), [robust standard errors](https://en.wikipedia.org/wiki/Heteroscedasticity-consistent_standard_errors), or variance stabilizing transformations.
#     </li>
#   <li> **Normal distribution for errors**.  The $\epsilon$ term we discussed at the beginning are assumed to be normally distributed. This can be verified with a fitted values vs. residuals plot and verifying that there is no pattern, and with a quantile plot.
#   $$ \epsilon_i \sim N(0, \sigma^2)$$
# Sometimes the distributions of responses $Y$ may not be normally distributed at any given value of $X$.  e.g. skewed positively or negatively. </li>
# <li> **Independent errors**.  The observations are assumed to be obtained independently.
#     <ul>
#         <li>e.g. Observations across time may be correlated
#     </ul>
# </li>
# </ul>  
# 
# </div>
# 
# There are some other issues that are important to investigate with linear regression models.
# 
# <div class="span5 alert alert-danger">
# <ul>
#   <li>**Correlated Predictors:** Care should be taken to make sure that the independent variables in a regression model are not too highly correlated. Correlated predictors typically do not majorly affect prediction, but do inflate standard errors of coefficients making interpretation unreliable. Common solutions are dropping the least important variables involved in the correlations, using regularization, or, when many predictors are highly correlated, considering a dimension reduction technique such as principal component analysis (PCA).
#   <li>**Influential Points:** Data points that have undue influence on the regression model. These points can be high leverage points or outliers. Such points are typically removed and the regression model rerun.
# </ul>
# </div>
# 

# <div class="span5 alert alert-info">
# <h3>Part 5 Checkup Exercises</h3>
# 
# <p>Take the reduced model from Part 3 to answer the following exercises. Take a look at [this blog post](http://mpastell.com/2013/04/19/python_regression/) for more information on using statsmodels to construct these plots.</p>
#     
# <p><b>Exercise:</b> Construct a fitted values versus residuals plot. What does the plot tell you? Are there any violations of the model assumptions?</p>
# 
# <p><b>Exercise:</b> Construct a quantile plot of the residuals. What does the plot tell you?</p>
# 
# <p><b>Exercise:</b> What are some advantages and disadvantages of the fitted vs. residual and quantile plot compared to each other?</p>
# 
# <p><b>Exercise:</b> Identify any outliers (if any) in your model and write a story describing what these outliers might represent.</p>
# 
# <p><b>Exercise:</b> Construct a leverage plot and identify high leverage points in the model. Write a story explaining possible reasons for the high leverage points.</p>
# 
# <p><b>Exercise:</b> Remove the outliers and high leverage points from your model and run the regression again. How do the results change?</p>
# </div>

# **Construct a fitted values versus residuals plot. What does the plot tell you? Are there any violations of the model assumptions?**

# In[52]:


# your turn

plt.scatter(m_three_vars.fittedvalues, m_three_vars.resid)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Fitted Values vs. Residuals')

plt.show()


# A fitted values vs. residual plot determines whether the distribution of the random error ϵ is normal. If it is normally distributed, its mean is 0. If the plot shows no pattern, it can be concluded that ϵ follows a normal distribution, which is the case here. 

# **Construct a quantile plot of the residuals. What does the plot tell you?**

# In[54]:


# Normality Test with a QQ Plot

resid_norm = stats.probplot(m_three_vars.resid, dist='norm', plot=pylab)


# The QQ plot also checks for the normal distribution of errors. In this case, the data follows a relatively normal distribution, though a positive skew appears from the 20 range of 'Ordered Values' (i.e. residual values greater than 20). Such a positive skew indicates the presence of outliers. 

# **What are some advantages and disadvantages of the fitted vs. residual and quantile plot compared to each other?**

# When the two plots are compared, the clear advantage of the fitted vs. residual plot is that it displays the relationship between each fitted data point and its residual, whereas the quantile plot gives a general overview of the model's performance. 

# **Identify any outliers (if any) in your model and write a story describing what these outliers might represent.**

# As mentioned above, the outliers in the model are apparent above the range of 20 in 'Ordered Values'. In general, outliers represent erroneous data or a poorly fitting regression line. Here, the outliers can represent discrepancies between price and crime rate per capita, the number of rooms per dwelling, and/or pupil-to-teacher ratio by town.

# **Construct a leverage plot and identify high leverage points in the model. Write a story explaining possible reasons for the high leverage points.**

# In[60]:


from statsmodels.graphics.regressionplots import *
plot_leverage_resid2(m_three_vars)
influence_plot(m_three_vars)


# High leverage points lie far from the other data in the horizontal direction and have a significant impact on the slope of the regression line. From the leverage plots displayed above, high leverage points can clearly be seen for four distinct values: [380, 405, 418, and 440].

# **Remove the outliers and high leverage points from your model and run the regression again. How do the results change?**

# In[55]:


# Remove the outliers

bos_drop_outliers = bos.drop(bos[m_three_vars.resid>20].index, axis=0)
bos_updated = bos_drop_outliers

# Remove high leverage points

bos_drop_leverage = bos_updated.drop([380, 405, 418, 440], axis=0)


# In[56]:


# Re-run regression model

m_updated = ols('PRICE ~ PTRATIO + RM + CRIM', bos_drop_leverage).fit()
print(m_updated.summary())


# Removing the outliers and high leverage points change the viability of the model significantly . 
# This can be seen concluded due to the following:
# 
# - R2 increases from 0.594 to 0.715. 
# - The F-statistic increases from 245.2 to 411.6. 
# - AIC decreases from 3232 to 2941. 
# 
# All three of these factors prove that the model's performance improved with the removal of outliers and high leverage points. 
