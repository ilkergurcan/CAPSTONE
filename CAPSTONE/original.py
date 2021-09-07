import numpy as np
import pandas as pd

df = pd.read_csv("Cleaned-Data.csv")

del df["Country"]
del df["Gender_Female"]
del df["Gender_Male"]
del df["Gender_Transgender"]

del df["None_Sympton"]
del df["None_Experiencing"]

#del df["Age_0-9"]
#del df["Age_10-19"]
#del df["Age_20-24"]
#del df["Age_25-59"]
#del df["Age_60+"]
#df["Contact"] = np.nan

#df.loc[df.Contact_Yes == 1, "Contact"] = 1
#df.loc[df.Contact_No == 1, "Contact"] = 0
#df.loc[df["Contact_Dont-Know"] == 1, "Contact"] = 2

#del df["Contact_Yes"]
#del df["Contact_No"]
#del df["Contact_Dont-Know"]
#
df["Severity"] = np.nan

df.loc[df["Severity_Mild"] == 1, "Severity"] = 0
df.loc[df["Severity_Moderate"] == 1, "Severity"] = 1
df.loc[df["Severity_None"] == 1, "Severity"] = 1 #2
df.loc[df["Severity_Severe"] == 1, "Severity"] = 1 #3

del df["Severity_Mild"]
del df["Severity_Moderate"]
del df["Severity_None"]
del df["Severity_Severe"]

Y = df["Severity"]
del df["Severity"]
X = df

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=Y)

from imblearn.over_sampling import RandomOverSampler

oversample = RandomOverSampler(sampling_strategy = "minority")
X_train, y_train = oversample.fit_resample(X_train, y_train)

from sklearn.tree import DecisionTreeClassifier

dtree_model = DecisionTreeClassifier(max_depth = 4).fit(X_train, y_train)

dtree_predictions = dtree_model.predict(X_test)

dtree_predictions = pd.DataFrame(dtree_predictions)

from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, dtree_predictions))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, dtree_predictions)
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(dtree_model, X_test, y_test)  

recall = cm[1,1]/(cm[1,1] + cm[1,0])
print(recall)


from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3)#3

neigh.fit(X_train, y_train)

y_pred=neigh.predict(X_test)
y_pred = pd.DataFrame(y_pred)

from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix

cm2 = confusion_matrix(y_test, y_pred)

y_test.value_counts()

from sklearn.ensemble import RandomForestClassifier
#y_over = np.ravel(y_over)

RandomForest = RandomForestClassifier(n_estimators=20, max_depth=10,min_samples_split=2, random_state=0)

RandomForest=RandomForest.fit(X_train, y_train)
predRandomForest = RandomForest.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, predRandomForest))

cm3 = confusion_matrix(y_test, predRandomForest)

data = [[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0]]

p = neigh.predict(data)
print(p)

# çok fazla False Positive var


