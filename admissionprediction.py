import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

exam = pd.read_csv('C:\\Aneesh\\Machine Learning\\examdata.csv')
X = exam.iloc[:,:-1] ### all columns except last column(Input)
Y = exam.iloc[:, -1] ### Only last column (Output)

# filter out the applicants that got admitted
admitted = exam.loc[Y == 1]
# filter out the applicants that din't get admission
not_admitted = exam.loc[Y == 0]

##PLOTS##
plt.scatter(admitted.iloc[:, 0],admitted.iloc[:,1], label='Admitted')
plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], label='Not Admitted')
plt.legend()
plt.show()

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)
model = LogisticRegression()
model.fit(xTrain, yTrain)
predicted_classes = model.predict(xTest)



