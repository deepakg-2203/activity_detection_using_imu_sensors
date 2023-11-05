import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib  # Import joblib

# Load your dataset
dataset_path = 'E:/GEETHA/simple_final_combined_data.csv'
data = pd.read_csv(dataset_path)

# Split the data into features (X) and the target variable (y)
X = data.drop('Label', axis=1)  # Assuming 'label' is the column representing the target
y = data['Label']

# Split the data into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier model (you can choose a different model)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model to the training data
model.fit(X_train, y_train)

# Save (dump) the trained model to a file
model_filename = 'your_trained_model.pkl'  # You can choose the filename
joblib.dump(model, model_filename)

print(f"Model saved to {model_filename}")

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')