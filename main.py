from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import math
import numpy as np

data_2022 = pd.read_csv('data_2022.csv')
data_2023 = pd.read_csv('data_2023.csv') 

def make_dataset(data, window_size = 4):
    x, y = [], []
    for i in range(len(data) - window_size):
        x.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return x, y

validation_mse = []
test_mse = []

for i in range(1, 101):
    window_size = i
    dataset_2022 = data_2022['Demand'].tolist()
    X, Y = make_dataset(dataset_2022, window_size)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, shuffle = False)

    model = DecisionTreeRegressor(random_state = 3)
    model.fit(X_train, Y_train)

    predictions_val = model.predict(X_val)
    mse = math.sqrt(mean_squared_error(Y_val, predictions_val))
    validation_mse.append(mse)

    dataset_2023 = data_2023['Demand'].tolist()
    X_test, Y_test = make_dataset(dataset_2023, window_size)
    predictions_test = model.predict(X_test)
    mse_test = math.sqrt(mean_squared_error(Y_test, predictions_test))
    test_mse.append(mse_test)

average_rmse = [(v + t) / 2 for v, t in zip(validation_mse, test_mse)]
min_avg_rmse = min(average_rmse)
min_avg_rmse_index = average_rmse.index(min_avg_rmse) + 1

print("Min average RMSE:", min_avg_rmse)
print("Optimal window size:", min_avg_rmse_index)


effective_window_size = min_avg_rmse_index

# Create the dataset
dataset_2022 = data_2022['Demand'].tolist()
X, Y = make_dataset(dataset_2022, effective_window_size)

# Split the dataset
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=3, shuffle = False)

# Initialize and fit the model
model = DecisionTreeRegressor(random_state=3)
model.fit(X_train, Y_train)


# Plotting the actual vs predicted demand for 2022
predictions_2022 = []
for i in range(len(dataset_2022) - effective_window_size):
    seq = dataset_2022[i:i+effective_window_size]
    predictions_2022.append(model.predict([seq])[0])

dataset_2023 = data_2023['Demand'].tolist()
X_2023, Y_2023 = make_dataset(dataset_2023, effective_window_size)

# Predicting the demand for 2023
predictions_2023 = []
for i in range(len(X_2023)):
    seq = X_2023[i]  # No need for the loop as we already have the sequences prepared
    predictions_2023.append(model.predict([seq])[0])

def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[indices], y[indices]

n_trees = 100
n_features = int(np.sqrt(np.array(X_train).shape[1]))*2 #Max features for each tree

ensemble = []
for i in range(n_trees):
    X_sample, y_sample = bootstrap_sample(np.array(X_train), np.array(Y_train))
    
    tree = DecisionTreeRegressor(max_depth=np.random.choice(range(3, 10)),
                                 min_samples_split=np.random.choice(range(2, 10)),
                                 min_samples_leaf=np.random.choice(range(1, 10)),
                                 max_features=n_features,
                                 random_state=np.random.randint(0, 1000))

    tree.fit(X_sample, y_sample)
    ensemble.append(tree)

def aggregate_predictions(predictions):
    predictions_array = np.array(predictions)
    mean_predictions = np.mean(predictions_array, axis=0)
    return mean_predictions


ensemble_predictions_2022 = []

for i in range(len(dataset_2022) - effective_window_size):
    seq = np.array(dataset_2022[i:i+effective_window_size]).reshape(1, -1)  # Reshape for a single sample
    # Get predictions from each tree and aggregate
    tree_predictions = [tree.predict(seq)[0] for tree in ensemble]
    ensemble_predictions_2022.append(np.mean(tree_predictions))

ensemble_predictions_2023 = []

for i in range(len(dataset_2023) - effective_window_size):
    seq = np.array(dataset_2023[i:i+effective_window_size]).reshape(1, -1)  # Reshape for a single sample
    # Get predictions from each tree and aggregate
    tree_predictions = [tree.predict(seq)[0] for tree in ensemble]
    ensemble_predictions_2023.append(np.mean(tree_predictions))

def predict_ensemble(ensemble, X):
    predictions = np.array([tree.predict(X) for tree in ensemble])
    return np.mean(predictions, axis=0)

ensemble_predictions = predict_ensemble(ensemble, np.array(X_val))

# Calculate the R^2 score for the ensemble
ensemble_r2 = r2_score(Y_val, ensemble_predictions)

print(f"Ensemble R^2 on validation set: {ensemble_r2}")

# Now plot the results
plt.figure(figsize=(15, 7))

plt.plot(range(1, len(dataset_2022) + 1), dataset_2022, color="red", label="Actual Demand 2022", dashes=[1, 1])
plt.plot(range(1, len(dataset_2023) + 1), dataset_2023, color="green", label="Actual Demand 2023", dashes=[1, 1])

plt.plot(range(effective_window_size + 1, len(ensemble_predictions_2022) + effective_window_size + 1), ensemble_predictions_2022, label='Ensemble Predicted Demand 2022', color='blue')
plt.plot(range(effective_window_size + 1, len(ensemble_predictions_2023) + effective_window_size + 1), ensemble_predictions_2023, label='Ensemble Predicted Demand 2023', color='orange')

#plt.plot(range(effective_window_size + 1, len(predictions_2022) + effective_window_size + 1), predictions_2022, color="purple", label="Predicted Demand 2022")
#plt.plot(range(effective_window_size + 1, len(predictions_2023) + effective_window_size + 1), predictions_2023, color="cyan", label="Predicted Demand 2023")
plt.title('Actual vs Ensemble Predicted Demand for 2022 and 2023')
plt.xlabel('Day of the Year')
plt.ylabel('Demand')
plt.legend()
#plt.show()