import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

df = pd.read_csv("CoupDataset.xlsx - Sayfa1.csv")

df = df.drop(['Country'], axis=1)


"""
df = df.drop(['GDPGrowth'], axis=1)
df = df.drop(['InflationRate'], axis=1)
"""
#print(df['coup'])
#print(df['past'])

df = df.drop(['Population', 'Military Size', 'Export', 'Import', 'GDPPerCapita', 'RateOfPopOnMilitary',
              "MilitaryBudget", 'RateOfExportToImport'], axis=1)

df_target = df['Coup']
feature_columns = df.drop(['Coup'], axis=1)
#feature_columns = df
print(feature_columns)

normalized = preprocessing.normalize(feature_columns)


x_train = feature_columns[:90]
y_train = df_target[:90]
x_test = feature_columns[70:]
y_test = df_target[70:]


x_train, x_test, y_train, y_test = train_test_split(normalized, df_target, test_size=0.3, random_state=56)

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
confusion = confusion_matrix(y_test, svmPrediction)
print(confusion)
print("SVM Accuracy:", svmAccuracy)

kmeans = KMeans(n_clusters=2, random_state=56).fit(df)
print(kmeans.labels_)
print(df['Coup'].to_numpy())

print(df.columns)


ranfor = RandomForestClassifier()

voting = VotingClassifier(estimators=[('svc', svmModel), ('rf', ranfor), ('log', model)], voting='hard')

voting.fit(x_train, y_train)
pred = voting.predict(x_test)
confusion = confusion_matrix(y_test, pred)
print(confusion)
