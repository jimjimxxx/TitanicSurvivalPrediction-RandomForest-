import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

file_path = "C:\\Users\\USER\OneDrive - 國立成功大學 National Cheng Kung University\\2(二)\\Python程式設計與資料庫實務\\train_and_test2.csv"
titanic_data = pd.read_csv(file_path)

titanic_data_cleaned = titanic_data.drop(columns=['Passengerid']).dropna()

feature_columns = ['Age', 'Fare', 'Sex', 'sibsp', 'Pclass', 'Embarked']
target_column = '2urvived'

X = titanic_data_cleaned[feature_columns]
y = titanic_data_cleaned[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_rf_pred = rf_model.predict(X_test)

rf_train_accuracy = rf_model.score(X_train, y_train)
rf_test_accuracy = accuracy_score(y_test, y_rf_pred)
rf_conf_matrix = confusion_matrix(y_test, y_rf_pred)


print("隨機森林")
print("訓練資料準確率:",rf_train_accuracy)
print("測試資料準確率:",rf_test_accuracy)
print("混淆矩陣:\n",rf_conf_matrix)

