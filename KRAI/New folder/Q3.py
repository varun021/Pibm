import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("customer_churn_dataset-training-master.csv")

print(df.head())

print(df.isnull().sum())

le = LabelEncoder()

df['Gender'] = le.fit_transform(df['Gender'])  # Encode Gender (Male, Female)
df['Subscription Type'] = le.fit_transform(df['Subscription Type'])  # Encode Subscription Type (Standard, Basic, Premium)
df['Contract Length'] = le.fit_transform(df['Contract Length'])  # Encode Contract Length (Annual, Monthly, Quarterly)

X = df.drop(['CustomerID', 'Churn'], axis=1)  # Drop irrelevant columns
y = df['Churn']  # Churn is the target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_test.mean(), inplace=True)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Accuracy Score: {accuracy_score(y_test, y_pred)}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
print(feature_importance_df.sort_values(by='Importance', ascending=False))
