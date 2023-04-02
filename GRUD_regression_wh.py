import numpy as np
from sklearn.preprocessing import StandardScaler
from pypots.data import load_specific_dataset, mcar, masked_fill
from pypots.imputation import SAITS
from pypots.utils.metrics import cal_mae
from pypots.regression import GRUD_REGRESSOR
# Data preprocessing. Tedious, but PyPOTS can help. 🤓
# data = load_specific_dataset('physionet_2012')
import pandas as pd

data_path = '/netapp/datasets/NirBZ/Predictive_maintanence/mode_separation'
X = pd.read_csv(data_path + '/combined.csv')

window = 49
horizon = 1
seq_len = window + horizon

def generate_dataset_from_samples(X):
    stacked_sequences = []
    for idx in range(len(X)-seq_len):
        stacked_sequences.append(X[idx: idx + seq_len])
    X = np.stack(stacked_sequences)
    # X = X.reshape(int(len(X) / seq_len), seq_len, -1)
    y = np.zeros(len(X))
    X_intact, X, missing_mask, indicating_mask = mcar(X, 0.1) # hold out 10% observed values as ground truth
    X = masked_fill(X, 1 - missing_mask, np.nan)
    return X, y

def generate_train_test_set(X, seq_len, train_size=50000, test_size=50000):
    X = X.iloc[:int(len(X)/seq_len)*seq_len]
    X = X.drop('date', axis = 1)
    X = StandardScaler().fit_transform(X.to_numpy())

    train_set = X[:train_size + seq_len]
    test_set = X[train_size:train_size + test_size + seq_len]

    train_X, train_y = generate_dataset_from_samples(train_set)
    test_X, test_y = generate_dataset_from_samples(test_set)

    return train_X, train_y, test_X, test_y

# dataset_size = 50000
# X = X.iloc[:int(len(X) / seq_len) * seq_len]
#
# X = X.drop('date', axis=1)
# X = StandardScaler().fit_transform(X.to_numpy())
#
# X = X[:dataset_size + seq_len]
# stacked_sequences = []
# for idx in range(len(X) - seq_len):
#     stacked_sequences.append(X[idx: idx + seq_len])
# X = np.stack(stacked_sequences)
# # X = X.reshape(int(len(X) / seq_len), seq_len, -1)
# y = np.zeros(len(X))
# X_intact, X, missing_mask, indicating_mask = mcar(X, 0.1)  # hold out 10% observed values as ground truth
# X = masked_fill(X, 1 - missing_mask, np.nan)

train_X, train_y, test_X, test_y = generate_train_test_set(X, seq_len, train_size=50000, test_size=10000)

grud = GRUD_REGRESSOR(n_steps=50, n_features=6, rnn_hidden_size=32, n_classes=2, epochs=10)

# saits = SAITS(n_steps=48, n_features=37, n_layers=2, d_model=256, d_inner=128, n_head=4, d_k=64, d_v=64, dropout=0.1, epochs=2)
grud.fit(train_X, train_y, val_X=test_X, val_y=test_y)
prediction_result = grud.predict_on_train(train_X, train_y, val_X=test_X, val_y=test_y)

mean_train_loss, epoch_train_predictions_collector, epoch_train_gt_collector, \
                mean_val_loss, epoch_val_predictions_collector, epoch_val_gt_collector = prediction_result


# import plotly.express as px
# fig = px.line(epoch_train_gt_collector)
# fig.add_scatter(y=epoch_train_predictions_collector)
# fig.show()
#fig.write_html('train_50k_window50.html')

x=1
# imputation = saits.impute(X)  # impute the originally-missing values and artificially-missing values
# mae = cal_mae(imputation, X_intact, indicating_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)