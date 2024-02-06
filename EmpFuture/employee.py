import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score,precision_score,confusion_matrix, accuracy_score


data=pd.read_csv('empenv\Employee.csv')

ed = {'Bachelors': 1, 'Masters': 2, 'PHD': 3}
city = {'Bangalore': 1, 'Pune': 2, 'New Delhi': 3}
gender = {'Male': 0, 'Female': 1}
benched = {'No': 0, 'Yes': 1}

data['Education'] = data['Education'].map(ed)
data['City'] = data['City'].map(city)
data['Gender'] = data['Gender'].map(gender)
data['EverBenched'] = data['EverBenched'].map(benched)



x = data.iloc[:, 0:8]
y = data['LeaveOrNot'].values


x_train, x_test, y_train, y_test = train_test_split( x,y , test_size = 0.2, random_state = 0)

classifier = RandomForestClassifier()
grid_values = {'n_estimators':[50, 80,  100], 'max_depth':[3, 5, 7]}
classifier = GridSearchCV(classifier, param_grid = grid_values, scoring = 'roc_auc', cv=5)
classifier.fit(x_train, y_train)


classifier.fit(x_train, y_train)

pickle.dump(classifier, open('empenv\model.mkl','wb'))
