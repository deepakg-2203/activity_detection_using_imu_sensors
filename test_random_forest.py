import pandas as pd
from sklearn.metrics import accuracy_score
import joblib  # Import joblib for loading the trained model
from sklearn.impute import SimpleImputer

# Load your test dataset
test_dataset_path = 'E:/GEETHA/sit_test.csv'
test_data = pd.read_csv(test_dataset_path)

# Separate the test data into features (X_test)
X_test = test_data.drop('Label', axis=1)


# Create an imputer
imputer = SimpleImputer(strategy='mean')

# Fit and transform the test data
X_test_imputed = imputer.fit_transform(X_test)

# Now you can use X_test_imputed with the Random Forest Classifier model
X_test.dropna(inplace=True)

# Preprocess the test data (apply the same preprocessing steps as for training data)

# Load the trained model
model = joblib.load('your_trained_model.pkl')  # Replace 'your_trained_model.pkl' with the path to your trained model

# Make predictions on the test data using the loaded model
y_pred = model.predict(X_test)

print(y_pred)
prev= None

for i in y_pred:
    if i!=prev:
        prev=i
        if i==1:
            print("sitting")
        elif i==2:
            print("standing")
        elif i==3:
            print("walking")
# Load the actual labels/targets for the test data
y_true = test_data['Label']

# Evaluate model performance (use appropriate metrics for your problem)
accuracy = accuracy_score(y_true, y_pred)
print(f'Accuracy: {accuracy:.2f}')