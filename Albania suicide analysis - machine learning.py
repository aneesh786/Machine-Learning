import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#####input file######
df = pd.read_csv('C:\Aneesh\Kaggle\suicide-rates-overview-1985-to-2016\master.csv')
#####Input File for Albania Only#######
Albania = df[df.country == 'Albania']
###Remove the column 'HDI for year'####
Albania.drop("HDI for year", axis=1, inplace=True)
Albania.groupby('generation').size()
Albania.groupby('sex').size()
suicide = Albania['suicides_no']
Albania.sex.replace(['male','female'],['1','0'], inplace=True)


gdp_capita = Albania['gdp_per_capita ($)']
sns.countplot(df['age'])
sns.countplot(Albania['age'],hue=Albania['sex'])##Not much predictions
sns.countplot(suicide,hue=Albania['sex']) ##shows suicide is higher in Men compared to Women

Albania.info()
Albania.drop("country-year", axis=1, inplace=True)
x = Albania[['year','age','population','suicides/100k pop','gdp_per_capita ($)']]
y = Albania['suicides_no']
x.rename(columns={' gdp_for_year ($) ':'gdp_year'},inplace=True)
x.rename(columns={'gdp_per_capita ($)':'gdp_capita'},inplace=True)
#########Feature Scaling###########
x.population = x.population / 100000
x.gdp_capita = x.gdp_capita / x.gdp_capita.max()
x.age.replace(['5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years'],['0','1','2','3','4','5'],inplace=True)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 45)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
mse = np.mean((y_test - y_pred)**2)
print("MSE :", mse)
rmse = np.sqrt(mse)
print("RMSE :", rmse)
r2 = r2_score(y_test, y_pred)
print("r2_score :", r2)


from sklearn.ensemble import RandomForestRegressor
# creating the model
model = RandomForestRegressor()
# feeding the training data into the model
model.fit(x_train, y_train)
# predicting the test set results
y_pred = model.predict(x_test)
# calculating the mean squared error
mse = np.mean((y_test - y_pred)**2)
print("MSE :", mse)
# calculating the root mean squared error
rmse = np.sqrt(mse)
print("RMSE :", rmse)
#calculating the r2 score
r2 = r2_score(y_test, y_pred)
print("r2_score :", r2)


from sklearn.tree import DecisionTreeRegressor
# creating the model
model = DecisionTreeRegressor()
# feeding the training data into the model
model.fit(x_train, y_train)
# predicting the test set results
y_pred = model.predict(x_test)
# calculating the mean squared error
mse = np.mean((y_test - y_pred)**2)
print("MSE :", mse)
# calculating the root mean squared error
rmse = np.sqrt(mse)
print("RMSE :", rmse)
#calculating the r2 score
r2 = r2_score(y_test, y_pred)
print("r2_score :", r2)