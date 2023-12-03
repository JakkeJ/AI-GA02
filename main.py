from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.utils import resample
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime, timedelta

data_2022 = 'data_2022.csv'
data_2023 = 'data_2023.csv'

def plot_data(
        start_date: datetime,
        end_date: datetime,
        dataset,
        train_set_length,
        predictions,
        window_size: int,
        title="Title",
        ):
    start_num = mdates.date2num(start_date)
    end_num = mdates.date2num(end_date)

    dates = [start_date + timedelta(days=i) for i in range(len(dataset))]
    dates = mdates.date2num(dates)

    plt.figure(figsize=(15, 10))
    plt.xlim(start_num, end_num)
    plt.title(title)

    plt.plot(dates, dataset, label='Actual Demand', color='blue')

    prediction_start_date = start_date + timedelta(days=window_size)
    prediction_dates = [prediction_start_date + timedelta(days=i) for i in range(len(predictions))]
    prediction_dates = mdates.date2num(prediction_dates)

    plt.plot(prediction_dates, predictions, label=f'Predictions with Window Size {window_size}', color='red')

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
    plt.gcf().autofmt_xdate()

    plt.xlabel('Date')
    plt.ylabel('Demand')
    plt.legend()
    plt.show()

def make_dataset(
        file_names: list,
        target_column,
        window_size=4
        ):
    data = pd.concat([pd.read_csv(i) for i in file_names])
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data.iloc[i:i + window_size][target_column].values)
        y.append(data.iloc[i + window_size][target_column])
    return np.array(X), np.array(y), data[target_column].values

def find_best_window_size(data):
    test_rmse = []
    test_mape = []
    test_r2 = []
    test_mae = []
    for i in range(1, 101):
        window_size = i
        X, Y, _ = make_dataset(data, "Demand", window_size)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle = False)
        model = DecisionTreeRegressor(random_state = 89)
        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)
        test_rmse.append(math.sqrt(mean_squared_error(Y_test, predictions)))
        test_mape.append(mean_absolute_percentage_error(Y_test, predictions))
        test_r2.append(r2_score(Y_test, predictions))
        test_mae.append(mean_absolute_error(Y_test, predictions))
    min_rmse = min(test_rmse)
    min_rmse_index = test_rmse.index(min_rmse) + 1
    min_mape = min(test_mape)
    min_mape_index = test_mape.index(min_mape) + 1
    min_r2 = max(test_r2)
    min_r2_index = test_r2.index(min_r2) + 1
    min_mae = min(test_mae)
    min_mae_index = test_mae.index(min_mae) + 1

    return min_rmse, min_rmse_index, min_mape, min_mape_index, min_r2, min_r2_index, min_mae, min_mae_index

##############
### Task 2 ###
##############

min_rmse, min_rmse_index, min_mape, min_mape_index, min_r2, min_r2_index, min_mae, min_mae_index = find_best_window_size([data_2022])

optimal_window_size = int((min_mae_index + min_r2_index + min_mape_index + min_rmse_index)/4)

print("\n### Decision Tree Regressor ###")
print("Min RMSE:", min_rmse)
print("Min MAPE:", min_mape)
print("Max R2:", min_r2)
print("Min MAE:", min_mae)
print("Optimal window size:", optimal_window_size)

X_2022, Y_2022, dataset_2022 = make_dataset([data_2022], "Demand", optimal_window_size)
X_train_2022, X_test_2022, Y_train_2022, Y_test_2022 = train_test_split(X_2022, Y_2022, test_size=0.3, random_state=89, shuffle=False)

model = DecisionTreeRegressor(random_state=89)
model.fit(X_train_2022, Y_train_2022)

predictions_2022 = model.predict(X_2022)

mae_tree = mean_absolute_error(Y_2022, predictions_2022)
r2_tree = r2_score(Y_2022, predictions_2022)
print(f"MAE for Decision Tree Regressor: {mae_tree}")
print(f"R2 for Decision Tree Regressor: {r2_tree}")

start_date = datetime(2022, 1, 1)
end_date = datetime(2022, 12, 31)

plot_data(start_date, end_date, dataset_2022, len(X_train_2022), predictions_2022, optimal_window_size, 'Decision Tree Regressor: Actual vs Predicted Demand for 2022')

##############
### Task 3 ###
##############

class RandomForest:
    '''
    Custom Random Forest class for regression problems. This implementation supports
    bootstrapped samples, random splits, and random subspace method. The implementation
    supports a weighted prediction scheme for regression problems.
    
    Parameters:
    -----------
    n_trees: int
    The number of classification trees that are used.
    max_depth: int
    The maximum depth of the tree
    min_samples_split: int
    The minimum number of samples needed to make a split when building a tree
    min_samples_leaf: int
    The minimum number of samples needed to be at a leaf node.
    max_features: int or literal string "sqrt", "log2", "auto".
    The number of features to consider when looking for the best split
    random_state: int
    Random seed for sampling and subspace randomization
    
    Attributes:
    -----------
    forest: list
    A list of DecisionTree objects
    
    Methods:
    -----------
    fit(X, y)
    Given training data X and targets y, trains a random forest of decision trees
    predict(X)
    Given a new set of data X, predicts the target for each point in X
    weighted_predict(X, weights)
    Given a new set of data X, predicts the target for each point in X using a weighted prediction scheme
    evaluate_tree_performance(X, y)
    Given a dataset X and targets y, evaluates the performance of each tree in the random forest
    evaluate_performance(X, y)
    Given a dataset X and targets y, evaluates the performance of the random forest
    '''
    def __init__(self, n_trees=100, max_depth_range=(3, 50), min_samples_split_range=(2, 50), min_samples_leaf_range=(1, 50), max_features='sqrt', random_state=None):
        self.n_trees = n_trees
        self.max_depth_range = max_depth_range
        self.min_samples_split_range = min_samples_split_range
        self.min_samples_leaf_range = min_samples_leaf_range
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []

    def bootstrap_sample(self, X, y, block_size = 100):
        n_samples = X.shape[0]
        n_blocks = int(np.ceil(n_samples / block_size))
        block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)
        X_sample = np.concatenate([X[i * block_size:(i + 1) * block_size] for i in block_indices], axis=0)
        y_sample = np.concatenate([y[i * block_size:(i + 1) * block_size] for i in block_indices], axis=0)
        X_sample = X_sample[:n_samples]
        y_sample = y_sample[:n_samples]
        
        return X_sample, y_sample

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

    def evaluate_performance(self, X, y):
        predictions = self.predict(X)
        return r2_score(y, predictions), mean_absolute_error(y, predictions)

random_forest = RandomForest(
                n_trees = 10,
                max_depth_range = (3, 50),
                min_samples_split_range = (2, 50),
                min_samples_leaf_range = (1, 50),
                max_features = 'sqrt',
                random_state = 89
                )
random_forest.fit(X_train_2022, Y_train_2022)
rf_predictions_train = random_forest.predict(X_train_2022)
rf_predictions_2022 = random_forest.predict(X_test_2022)
all_rf_predictions = np.concatenate([rf_predictions_train, rf_predictions_2022])

rf_r2, rf_mae = random_forest.evaluate_performance(X_test_2022, Y_test_2022)
print("\n### Random Forest ###")
print("MAE for Random Forest:", rf_mae)
print("R2 for Random Forest:", rf_r2)

plot_data(start_date, end_date, dataset_2022, len(X_train_2022), all_rf_predictions, optimal_window_size, 'Random Forest: Actual vs Predicted Demand for 2022')

##############
### Task 4 ###
##############

def plot_mlp_predictions(
        actual_dates,
        actual_demand,
        predictions,
        title='MLP Predictions'
        ):
    assert len(actual_dates) == len(predictions), "The lengths of actual dates and predictions must match."

    plt.figure(figsize=(15, 10))
    plt.plot(actual_dates, actual_demand, label='Actual Demand', color='blue')
    plt.plot(actual_dates, predictions, label='MLP Predictions', color='red', linestyle='--')
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
    plt.title(title)
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Demand')
    plt.legend()
    plt.show()

def make_dataset_mlp(
        file_names,
        target_column,
        window_size=4
        ):
    data = pd.concat([pd.read_csv(file) for file in file_names])
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    data['DayOfWeek'] = data['Date'].dt.dayofweek

    for lag in range(1, window_size + 1):
        data[f'Demand_Lag{lag}'] = data[target_column].shift(lag)

    data.dropna(inplace=True)
    feature_cols = [f'Demand_Lag{lag}' for lag in range(1, window_size + 1)] + ['DayOfWeek']
    X = data[feature_cols].values
    y = data[target_column].values

    return X, y

def find_best_window_size_mlp(data):
    test_rmse = []
    test_mape = []
    test_r2 = []
    test_mae = []
    for i in range(1, 15):
        window_size = i
        X, Y = make_dataset_mlp([data_2022], "Demand", window_size)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle = False)
        model = MLPRegressor(random_state = 89, max_iter = 5000)
        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)
        test_rmse.append(math.sqrt(mean_squared_error(Y_test, predictions)))
        test_mape.append(mean_absolute_percentage_error(Y_test, predictions))
        test_r2.append(r2_score(Y_test, predictions))
        test_mae.append(mean_absolute_error(Y_test, predictions))

    min_rmse = min(test_rmse)
    min_rmse_index = test_rmse.index(min_rmse) + 1
    min_mape = min(test_mape)
    min_mape_index = test_mape.index(min_mape) + 1
    min_r2 = max(test_r2)
    min_r2_index = test_r2.index(min_r2) + 1
    min_mae = min(test_mae)
    min_mae_index = test_mae.index(min_mae) + 1

    return min_rmse, min_rmse_index, min_mape, min_mape_index, min_r2, min_r2_index, min_mae, min_mae_index

class MLPEnsemble:
    '''
    Custom MLP Ensemble class for regression problems. This implementation supports
    bootstrapped samples, random splits, and random subspace method. The implementation
    supports a weighted prediction scheme for regression problems.

    Parameters:
    -----------
    n_estimators: int
    The number of MLPs that are used.
    hidden_layer_sizes: tuple
    The number of hidden layers in each MLP
    max_iter: int
    The maximum number of iterations to train each MLP
    random_state: int
    Random seed for sampling and subspace randomization
    
    Attributes:
    -----------
    ensemble: list
    A list of MLPRegressor objects
    
    Methods:
    -----------
    fit(X, y)
    Given training data X and targets y, trains an ensemble of MLPs
    predict(X)
    Given a new set of data X, predicts the target for each point in X
    evaluate_performance(X, y)
    Given a dataset X and targets y, evaluates the performance of the ensemble
    '''
    def __init__(self, n_estimators=5, max_iter=1000, random_state = None, early_stopping = True):
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.random_state = random_state
        self.ensemble = []
        self.early_stopping = early_stopping
        np.random.seed(random_state)

    def fit(self, X, y):
        bootstrapped_samples = self.bootstrap_samples(X, y)
        for i in range(self.n_estimators):
            hidden_layer_sizes = (15, 15)

            mlp = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=self.max_iter,
                learning_rate = 'adaptive',
                learning_rate_init = 0.01,
                random_state=self.random_state,
                early_stopping=self.early_stopping
            )
            X_sample, y_sample = bootstrapped_samples[i]
            mlp.fit(X_sample, y_sample)
            self.ensemble.append(mlp)

    def bootstrap_samples(self, X, y):
        bootstrapped_samples = []
        for _ in range(self.n_estimators):
            X_sample, y_sample = resample(X, y, replace=True)
            bootstrapped_samples.append((X_sample, y_sample))
        return bootstrapped_samples

    def predict(self, X):
        predictions = np.array([mlp.predict(X) for mlp in self.ensemble])
        return np.mean(predictions, axis=0)

    def evaluate_performance(self, X, y):
        predictions = self.predict(X)
        return r2_score(y, predictions), mean_absolute_error(y, predictions)


dataset_2022_mlp = pd.read_csv(data_2022)
dataset_2022_mlp['Date'] = pd.to_datetime(dataset_2022_mlp['Date'], dayfirst = True)
dataset_2022_mlp['DayOfWeek'] = dataset_2022_mlp['Date'].dt.dayofweek

min_rmse_mlp, min_rmse_index_mlp, min_mape_mlp, min_mape_index_mlp, min_r2_mlp, min_r2_index_mlp, min_mae_mlp, min_mae_index_mlp = find_best_window_size_mlp([data_2022])

optimal_window_size_mlp = int((min_mae_index_mlp + min_r2_index_mlp + min_mape_index_mlp + min_rmse_index_mlp)/4) * 2

print("\n### MLP Ensemble 2022 ###")
print("Min RMSE:", min_rmse_mlp)
print("Min MAPE:", min_mape_mlp)
print("Max R2:", min_r2_mlp)
print("Min MAE:", min_mae_mlp)
print("Optimal window size:", optimal_window_size_mlp)

X, y = make_dataset_mlp([data_2022], "Demand", optimal_window_size_mlp)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

mlp_ensemble = MLPEnsemble(n_estimators = 10, max_iter = 10000, random_state = 89, early_stopping = True)
mlp_ensemble.fit(X_train, y_train)
mlp_predictions_train = mlp_ensemble.predict(X_train)
mlp_predictions_2022 = mlp_ensemble.predict(X_test)
all_mlp_predictions = np.concatenate([mlp_predictions_train, mlp_predictions_2022])

performanceR2, performanceMAE = mlp_ensemble.evaluate_performance(X_test, y_test)
print("MAE for MLP Ensemble:", performanceMAE)
print("R2 for MLP Ensemble:", performanceR2)

plot_mlp_predictions(
    dataset_2022_mlp['Date'][optimal_window_size_mlp:],
    dataset_2022_mlp['Demand'][optimal_window_size_mlp:],
    all_mlp_predictions,
    title='MLP Model: Actual vs Predicted Demand for 2022'
)

##############
### Task 5 ###
##############

dataset_2023_mlp = pd.read_csv(data_2023)
dataset_2023_mlp['Date'] = pd.to_datetime(dataset_2023_mlp['Date'], dayfirst = True)
dataset_2023_mlp['DayOfWeek'] = dataset_2023_mlp['Date'].dt.dayofweek

min_rmse_mlp_2023, min_rmse_index_mlp_2023, min_mape_mlp_2023, min_mape_index_mlp_2023, min_r2_mlp_2023, min_r2_index_mlp_2023, min_mae_mlp_2023, min_mae_index_mlp_2023 = find_best_window_size_mlp([data_2023])

optimal_window_size_mlp_2023 = int((min_mae_index_mlp_2023 + min_r2_index_mlp_2023 + min_mape_index_mlp_2023 + min_rmse_index_mlp_2023)/4)

optimal_window_size_mlp_2023 *= 2

print("\n### MLP Ensemble 2023 ###")
print("Min RMSE:", min_rmse_mlp_2023)
print("Min MAPE:", min_mape_mlp_2023)
print("Max R2:", min_r2_mlp_2023)
print("Min MAE:", min_mae_mlp_2023)
print("Optimal window size:", optimal_window_size_mlp_2023)

X_2023, y_2023 = make_dataset_mlp([data_2023], "Demand", optimal_window_size_mlp_2023)

X_train_2023, X_test_2023, y_train_2023, y_test_2023 = train_test_split(X_2023, y_2023, test_size=0.3, shuffle=False)

mlp_ensemble_2023 = MLPEnsemble(n_estimators = 5, max_iter = 1000, random_state = 89, early_stopping = True)
mlp_ensemble_2023.fit(X_train_2023, y_train_2023)
mlp_predictions_train_2023 = mlp_ensemble_2023.predict(X_train_2023)
mlp_predictions_test_2023 = mlp_ensemble_2023.predict(X_test_2023)
all_mlp_predictions_2023 = np.concatenate([mlp_predictions_train_2023, mlp_predictions_test_2023])

print("MAE for MLP Ensemble:", mean_absolute_error(y_test_2023, mlp_predictions_test_2023))

plot_mlp_predictions(
    dataset_2023_mlp['Date'][optimal_window_size_mlp_2023:],
    dataset_2023_mlp['Demand'][optimal_window_size_mlp_2023:],
    all_mlp_predictions_2023,
    title='MLP Model: Actual vs Predicted Demand for 2023'
)

initial_data = dataset_2023_mlp[['Demand', 'DayOfWeek']].tail(optimal_window_size_mlp_2023)
most_recent_data_list = initial_data.values.tolist()

start_date = datetime(2023, 8, 26)
end_date = datetime(2023, 12, 31)
current_date = start_date

predicted_demand = []
while current_date <= end_date:
    demand_features = [row[0] for row in most_recent_data_list][-optimal_window_size_mlp_2023:]
    day_of_week = current_date.weekday()

    features = demand_features + [day_of_week]
    current_prediction = mlp_ensemble_2023.predict([features])[0]
    predicted_demand.append((current_date, current_prediction))

    most_recent_data_list.append([current_prediction, day_of_week])
    if len(most_recent_data_list) > optimal_window_size_mlp_2023:
        most_recent_data_list.pop(0)
    current_date += timedelta(days=1)

predicted_demand_df = pd.DataFrame(predicted_demand, columns=['Date', 'Predicted Demand'])
predicted_demand_df.set_index('Date', inplace=True)

plt.figure(figsize=(15, 10))
plt.plot(predicted_demand_df.index, predicted_demand_df['Predicted Demand'], label='Predicted Demand', color='blue', linestyle='--')
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
plt.title('Predicted Demand for the Rest of 2023')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()