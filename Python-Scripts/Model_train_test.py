import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pickle

data = pd.read_csv('/home/amitabh/PycharmProjects/PLANTS-DISEASE-IDENTIFICATIN-CASCADED-MODEL/Python-Scripts/model_0_features.csv')
X = data.drop('label', axis=1)  # Features
y = data['label']  # Labels
# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier()

#Random Forest Classifier
rf.fit(x_train.values,y_train)
pred_rf = rf.predict(x_test)

cm = confusion_matrix(y_test, pred_rf)
model_filename = '/home/amitabh/PycharmProjects/PLANTS-DISEASE-IDENTIFICATIN-CASCADED-MODEL/Level_0_model/level_0_model_no_0.pkl'
with open(model_filename, 'wb') as model_file:
    pickle.dump(rf, model_file)
print("Model saved as", model_filename)
# Plot confusion matrix using seaborn
fig=plt.figure(figsize=(6,3))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix(Random_Forest)")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()
fig.savefig('/home/amitabh/PycharmProjects/PLANTS-DISEASE-IDENTIFICATIN-CASCADED-MODEL/Python-Scripts/model_0-cm.svg',format='svg')

accuracy_rf = accuracy_score(y_test, pred_rf)
print("Accuracy is :",accuracy_rf)