#Amu Phalke - Decision Tree Task
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree

dataset = pd.read_csv("iris.csv")

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

dtree = DecisionTreeClassifier()
dtree.fit(x_train, y_train)

y_pred = dtree.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Accuracy: {accuracy:.2f}")

print("Classification Task:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

tree.plot_tree(dtree, filled=True)
plt.title("Iris Decision Tree Visualization")
plt.show()
#Amu Phalke - Decision Tree Task
