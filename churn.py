import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# load dataset
df = pd.read_csv("/content/customer_churn (2).csv")

# display dataset
print(df.head())
print(df.info())

# remove customerID if exists
if 'customerID' in df.columns:
    df.drop('customerID', axis=1, inplace=True)

# convert TotalCharges to numeric
if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'],
        errors='coerce'
    )

# handle missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# encode target column
# df['Churn'] = df['Churn'].map({
#     'Yes': 1,
#     'No': 0
# })

# encode categorical columns
label_encoder = LabelEncoder()

for col in df.select_dtypes(include='object').columns:

    df[col] = label_encoder.fit_transform(df[col])

# split features and target
X = df.drop('Churn', axis=1)

y = df['Churn']

# feature scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# build logistic regression model
model = LogisticRegression(
    max_iter=1000
)

# train model
model.fit(X_train, y_train)

# predictions
y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

# evaluation metrics
accuracy = accuracy_score(y_test, y_pred)

roc_auc = roc_auc_score(y_test, y_prob)

print("\nAccuracy Score:", accuracy)

print("\nROC AUC Score:", roc_auc)

# classification report
print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()

# ROC curve
fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(6, 5))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {roc_auc:.2f}"
)

plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()

# feature importance
feature_importance = pd.Series(
    model.coef_[0],
    index=X.columns
)

feature_importance = feature_importance.sort_values(
    ascending=False
)

print("\nTop Features Affecting Churn:\n")

print(feature_importance.head(10))

# plot feature importance
plt.figure(figsize=(10, 6))

feature_importance.head(10).plot(
    kind='barh'
)

plt.title("Top Features Affecting Customer Churn")

plt.gca().invert_yaxis()

plt.show()
