import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix

# Load data
df = pd.read_csv("customer_churn_dataset-training-master.csv")

# Drop rows with missing Churn values
df = df.dropna(subset=['Churn'])

# Encode categorical columns
df['Gender'] = LabelEncoder().fit_transform(df['Gender'])
df['Subscription Type'] = LabelEncoder().fit_transform(df['Subscription Type'])
df['Contract Length'] = LabelEncoder().fit_transform(df['Contract Length'])

# Features and target
X = df.drop(['CustomerID', 'Churn'], axis=1)
y = df['Churn']

# Handle missing values in features
imp = SimpleImputer(strategy='mean')
X = imp.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(confusion_matrix(y_test, y_pred))
