import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
confusion = confusion_matrix(y_test, predictions)

with open("metrics.txt", "w") as outfile:
    outfile.write("Accuracy: " + str(accuracy) + "
")
with open("metrics_2.txt", "w") as outfile:
    outfile.write("Confusion_matrix: " + str(confusion) + "
")

print("Accuracy:", accuracy)
print("Confusion matrix:")
print(confusion)

disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=clf.classes_)
plt.savefig("plot.png")
