import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

# Data Preprocessing 
dataset = pd.read_csv('your_dataset.csv')  # Update the file path

# Extracting features and labels
X = dataset[['N', 'P', 'K', 'temperature']].values  # Features: N.P.K and temperature
y = dataset['output_column'].values  # Output column

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Building the ANN
ann = tf.keras.models.Sequential()  # Initializing NN

# Input and first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Third hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compiling the ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the ANN
ann.fit(X_train, y_train, batch_size=32, epochs=500)

# Predicting for a new observation
new_observation = sc.transform([[N_value, P_value, K_value, temperature_value]])  # Replace values with actual values
prediction = ann.predict(new_observation)
print(prediction)

# Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
