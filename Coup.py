import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

df = pd.read_csv("CoupCSV1.csv", sep=';')

df['Inflation'] = df.Inflation.str.replace(r'%', r'', regex=True)
df['Grow'] = df.Inflation.str.replace(r'%', r'', regex=True)
df['Literate'] = df.Inflation.str.replace(r'%', r'', regex=True)
df['Unemployment'] = df.Inflation.str.replace(r'%', r'', regex=True)
df = df.replace(np.nan, 0, regex=True)
print(df)

feature_columns = df.drop(['Country', 'Year', 'Coup'], axis=1)

df_target = df['Coup']

normalized = preprocessing.normalize(feature_columns)
print(normalized)


x_train, x_test, y_train, y_test = train_test_split(normalized, df_target, test_size=0.2, random_state=1000)

model = LogisticRegression(solver='lbfgs', max_iter=1000000)
model.fit(x_train, y_train)

prediction = model.predict(x_test)
accuracy = metrics.accuracy_score(y_test, prediction)
confusion = confusion_matrix(y_test, prediction)
print(confusion)
print("Accuracy:", accuracy)

svmModel = SVC(kernel='linear')
svmModel.fit(x_train, y_train)
svmPrediction = svmModel.predict(x_test)
svmAccuracy = metrics.accuracy_score(y_test, svmPrediction)
print("SVM Accuracy:", svmAccuracy)