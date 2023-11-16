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

def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[indices], y[indices]

def aggregate_predictions(predictions):
    predictions_array = np.array(predictions)
    mean_predictions = np.mean(predictions_array, axis=0)
    return mean_predictions

def weighted_predict_ensemble(ensemble, X, weights):
    predictions = np.array([tree.predict(X) for tree in ensemble])
    weighted_predictions = np.average(predictions, axis=0, weights=weights)
    return weighted_predictions

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

n_trees = 100
n_features = int(np.sqrt(np.array(X_train).shape[1])) #Max features for each tree

ensemble = []
for i in range(n_trees):
    X_sample, y_sample = bootstrap_sample(np.array(X_train), np.array(Y_train))
    
    tree = DecisionTreeRegressor(max_depth=np.random.choice(range(3, 50)),
                                 min_samples_split=np.random.choice(range(2, 50)),
                                 min_samples_leaf=np.random.choice(range(1, 50)),
                                 max_features=n_features,
                                 random_state=np.random.randint(0, 10000))

    tree.fit(X_sample, y_sample)
    ensemble.append(tree)

class CustomRandomForest:
    def __init__(self, n_trees=100, max_depth_range=(3, 50), min_samples_split_range=(2, 50), min_samples_leaf_range=(1, 50), max_features='sqrt', random_state=None):
        self.n_trees = n_trees
        self.max_depth_range = max_depth_range
        self.min_samples_split_range = min_samples_split_range
        self.min_samples_leaf_range = min_samples_leaf_range
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []

    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        for _ in range(self.n_trees):
            X_sample, y_sample = self.bootstrap_sample(X, y)
            tree = DecisionTreeRegressor(
                max_depth=np.random.choice(range(*self.max_depth_range)),
                min_samples_split=np.random.choice(range(*self.min_samples_split_range)),
                min_samples_leaf=np.random.choice(range(*self.min_samples_leaf_range)),
                max_features=self.max_features,
                random_state=np.random.randint(0, 10000) if self.random_state is None else self.random_state
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_predictions = [tree.predict(X) for tree in self.trees]
        return np.mean(tree_predictions, axis=0)

    def weighted_predict(self, X, weights):
        tree_predictions = [tree.predict(X) for tree in self.trees]
        return np.average(tree_predictions, axis=0, weights=weights)

    def evaluate_tree_performance(self, X, y):
        performances = []
        for tree in self.trees:
            predictions = tree.predict(X)
            performances.append(r2_score(y, predictions))
        return performances

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

# Predictions for validation set (considering the effective window size)
val_predictions = model.predict(X_val)

tree_performances = []

for tree in ensemble:
    tree_predictions = tree.predict(X_val)
    tree_r2 = r2_score(Y_val, tree_predictions)
    tree_performances.append(tree_r2)

# Convert performance to positive weights (e.g., higher R2 score gets higher weight)
weights = np.array(tree_performances) - min(tree_performances)

# Normalize weights to sum up to 1
weights = weights / weights.sum()

weighted_ensemble_predictions = weighted_predict_ensemble(ensemble, np.array(X_val), weights)

# Predictions for validation set from the ensemble
ensemble_predictions_val = predict_ensemble(ensemble, np.array(X_val))

# Calculate R^2 score for the ensemble model
ensemble_r2 = r2_score(Y_val[-len(weighted_ensemble_predictions):], weighted_ensemble_predictions)

# Calculate R^2 score for the single tree model
r2_score_val = r2_score(Y_val[-len(val_predictions):], val_predictions)

from sklearn.ensemble import RandomForestRegressor

# Initialize RandomForestRegressor
# n_estimators is the number of trees in the forest
# max_features is the number of features to consider for each split - you might set this to 'sqrt' or 'log2'
rf_model = RandomForestRegressor(n_estimators=100, max_features='sqrt', random_state=3)

# Train the model on the training data
rf_model.fit(X_train, Y_train)

# Make predictions
rf_predictions_val = rf_model.predict(X_val)

# Calculate R^2 score
rf_r2_score_val = r2_score(Y_val, rf_predictions_val)
print(f"Random Forest R^2 on validation set: {rf_r2_score_val}")

print(f"Ensemble R^2 on validation set: {ensemble_r2}")
print(f"R^2 on validation set (single tree): {r2_score_val}")


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
plt.show()