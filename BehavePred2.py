import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, silhouette_score
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns
from interpret.blackbox import ShapKernel, PartialDependence
import os
from collections import Counter

# Load the filtered dataset
data_path = 'filtered_peptide_data.csv'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"File not found: {data_path}")

data = pd.read_csv(data_path)

# Ensure visit_month is numeric for modeling
data['visit_month'] = pd.to_numeric(data['visit_month'])

# Plotting peptide abundance distribution
plt.figure(figsize=(8, 5))
sns.histplot(data['peptide_abundance'], bins=30, kde=True, color='blue')
plt.title("Distribution of Peptide Abundance")
plt.xlabel("Peptide Abundance")
plt.ylabel("Frequency")
plt.show()

# Prepare features and target variable
X = data[['peptide_abundance']]
y = data['visit_month']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
y = imputer.fit_transform(y)

# Handle class imbalance using SMOTE
#smote = SMOTE()
#X_resampled, y_resampled = smote.fit_resample(X, y)




# Check class distribution
class_counts = Counter(y)
print(f"Class distribution before handling rare classes: {class_counts}")

# Remove rare classes
min_samples_threshold = 2
valid_classes = [cls for cls, count in class_counts.items() if count >= min_samples_threshold]
X = X[np.isin(y, valid_classes)]
y = y[np.isin(y, valid_classes)]

# Apply SMOTE
min_class_size = min(Counter(y).values())
smote_neighbors = min(min_class_size - 1, 5) if min_class_size > 1 else 1
smote = SMOTE(k_neighbors=smote_neighbors, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)




# Plotting resampled visit_month distribution
plt.figure(figsize=(8, 5))
sns.histplot(y_resampled, bins=20, kde=False, color='green')
plt.title("Resampled Visit Month Distribution")
plt.xlabel("Visit Month")
plt.ylabel("Frequency")
plt.show()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Regression Model to Predict Visit Frequency
regressor = RandomForestRegressor(random_state=42)
regressor.fit(X_train_scaled, y_train)
y_pred = regressor.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)

# Plotting predicted vs actual values
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.7, color='purple')
plt.title("Deep Learning Model: Predicted vs Actual")
plt.xlabel("Actual Visit Month")
plt.ylabel("Predicted Visit Month")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.show()

print(f"Random Forest Regression MSE: {mse:.2f}")

# 2. Clustering Analysis
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_train_scaled)
silhouette_avg = silhouette_score(X_train_scaled, clusters)

# Plot clustering results
plt.figure(figsize=(8, 5))
sns.scatterplot(x=X_train_scaled[:, 0], y=y_train, hue=clusters, palette='viridis')
plt.title("Clustering of Patients by Peptide Abundance")
plt.xlabel("Standardized Peptide Abundance")
plt.ylabel("Visit Month")
plt.show()

print(f"Clustering Silhouette Score: {silhouette_avg:.2f}")

# 3. Deep Learning Model
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(X_train_scaled, y_train, epochs=50, validation_split=0.2, batch_size=16, verbose=1)

# Plotting training and validation loss
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Deep Learning Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test_scaled, y_test)
print(f"Deep Learning Model Test MAE: {test_mae:.2f}")

# Insights Analysis: Checking Correlation between Peptide Abundance and Visit Frequency
correlation = np.corrcoef(data['peptide_abundance'], data['visit_month'])[0, 1]
print(f"Correlation between Peptide Abundance and Visit Month: {correlation:.2f}")

# Plotting correlation
plt.figure(figsize=(8, 5))
sns.regplot(x='peptide_abundance', y='visit_month', data=data, scatter_kws={"alpha":0.5})
plt.title("Correlation Between Peptide Abundance and Visit Month")
plt.xlabel("Peptide Abundance")
plt.ylabel("Visit Month")
plt.show()


#SHAP
# Initialize the ShapKernel explainer with correct arguments (including model)
shap_explainer = ShapKernel(model=model, data=X_train_scaled[:100])

# Generate local explanations for the test set
shap_local = shap_explainer.explain_local(X_test_scaled[:5], y_test[:5])

# Visualize local explanations for the first few instances (using matplotlib instead of show)
for i in range(5):  # First 5 instances
    explanation = shap_local.data(i)
    print(f"Explanation for instance {i + 1}: {explanation}")

    # Plot SHAP values for each instance
    plt.figure(figsize=(8, 4))
    plt.bar(explanation['names'], explanation['scores'].flatten(), color='blue')
    plt.title(f'Feature Importance {i + 1}')
    plt.ylabel('SHAP Value')
    plt.show()


feature_names = ['peptide_abundance']  # Replace this with the actual feature names
print(f"Feature index for 'peptide_abundance': {feature_names.index('peptide_abundance')}")




# Save the model for future use
model.save('/mnt/data/peptide_prediction_model.h5')
print("Model saved to /mnt/data/peptide_prediction_model.h5")

del model
del regressor
