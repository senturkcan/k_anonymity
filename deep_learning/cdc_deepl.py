import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 1. Load the data
df = pd.read_excel('new_dataset.xlsx')

# Display basic information about the dataset
print(f"Dataset shape: {df.shape}")
print("\nSample of the data:")
print(df.head())

# 2. Preprocess the data
# Inspect target variable distribution
print("\nTarget variable (GENHLTH) distribution:")
print(df['GENHLTH'].value_counts(normalize=True))

# Separate features and target
X = df.drop('GENHLTH', axis=1)
y = df['GENHLTH']

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"\nNumeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

# Apply preprocessing
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Get input dimensions for the model
input_dim = X_train_processed.shape[1]
print(f"\nProcessed feature dimension: {input_dim}")


# 3. Build the neural network model
def create_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dense(5, activation='softmax')  # 5 classes for GENHLTH (1-5)
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# Create model
model = create_model(input_dim)
model.summary()

# 4. Train the model
# Define callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Convert target to zero-based indexing for sparse_categorical_crossentropy
y_train_zero_indexed = y_train - 1
y_test_zero_indexed = y_test - 1

# Train the model
history = model.fit(
    X_train_processed, y_train_zero_indexed,
    epochs=30,
    batch_size=256,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# 5. Evaluate the model
# Plot training history
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout()
plt.savefig('training_history.png')
plt.close()

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(X_test_processed, y_test_zero_indexed)
print(f"\nTest accuracy: {test_accuracy:.4f}")

# Make predictions
y_pred_proba = model.predict(X_test_processed)
y_pred = np.argmax(y_pred_proba, axis=1) + 1  # Convert back to 1-5 scale
y_test_original = y_test_zero_indexed + 1  # Convert back to 1-5 scale

# Generate classification report
print("\nClassification Report:")
print(classification_report(y_test_original, y_pred))

# Generate confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test_original, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
plt.close()

# Feature importance analysis
# We can use a simpler model to get feature importance
from sklearn.ensemble import RandomForestClassifier

# Train a Random Forest for feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_processed, y_train_zero_indexed)

# Get feature names after preprocessing
feature_names = []
for name, transformer, features in preprocessor.transformers_:
    if name == 'num':
        feature_names.extend(features)
    elif name == 'cat':
        for feature in features:
            categories = transformer.categories_[features.index(feature)]
            for category in categories:
                feature_names.append(f"{feature}_{category}")

# Plot top 20 features
if hasattr(rf, 'feature_importances_'):
    importances = rf.feature_importances_
    indices = np.argsort(importances)[-20:]  # Top 20 features

    plt.figure(figsize=(12, 8))
    plt.title('Top 20 Feature Importances')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] if i < len(feature_names) else f"feature_{i}" for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

# 6. Save the model
model.save('health_prediction_model.h5')
print("\nModel saved as 'health_prediction_model.h5'")


# Function to make predictions on new data
def predict_health_status(new_data, model, preprocessor):
    """
    Make predictions on new data.

    Parameters:
    - new_data: DataFrame with the same features as the training data
    - model: Trained model
    - preprocessor: Fitted preprocessor

    Returns:
    - Predicted health status (1-5)
    """
    # Preprocess the new data
    processed_data = preprocessor.transform(new_data)

    # Make predictions
    predictions_proba = model.predict(processed_data)
    predictions = np.argmax(predictions_proba, axis=1) + 1  # Convert back to 1-5 scale

    return predictions


print("\nPrediction function is ready to use.")