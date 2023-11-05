import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# Load your test dataset
test_dataset_path = 'E:/GEETHA/all_test2.csv'
test_data = pd.read_csv(test_dataset_path)

# Load the trained K-Means model using joblib
model_filename = 'kmeans_model.pkl'  # Ensure this matches the filename you used for saving the model
loaded_kmeans = joblib.load(model_filename)
from sklearn.impute import SimpleImputer

# Create an imputer
imputer = SimpleImputer(strategy='mean')

# Fit and transform the test data
test_data_imputed = imputer.fit_transform(test_data)

# Now you can use test_data_imputed with the K-Means model

test_data.dropna(inplace=True)

# Predict cluster labels for the test data
test_data['Cluster'] = loaded_kmeans.predict(test_data.drop(['Label'], axis=1))

# Assuming you have the true labels (activities) for the test data in a 'Label' column
y_true = test_data['Label']
y_pred = []

# Map cluster labels to activity labels
for cluster in test_data['Cluster'].unique():
    # Calculate the most common activity label in the cluster
    cluster_activity = test_data[test_data['Cluster'] == cluster]['Label'].mode().values[0]
    # Assign this activity label to all data points in the cluster
    y_pred.extend([cluster_activity] * len(test_data[test_data['Cluster'] == cluster]))

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Print the predicted cluster labels for each data point in the test dataset
#print("Predicted Cluster Labels:")
#print(test_data['Cluster'])

# Print the corresponding mapped activity labels
print("Mapped Activity Labels:")
print(y_pred)