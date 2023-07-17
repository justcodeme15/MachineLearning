import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

#get csv file
df = pd.read_csv('flask-app/csv/AW_all_data.csv')

#all australian customer
all_australia = df.loc[df['CountryRegionName'] == 'Australia']

x = np.array(all_australia.iloc[:,18:21])
y = np.array(all_australia.iloc[:,-1])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0) 

classifier = RandomForestClassifier()
classifier.fit(x_train,y_train)


pickle.dump(classifier, open('../model.pkl',"wb"))