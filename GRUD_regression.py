import numpy as np
from sklearn.preprocessing import StandardScaler
from pypots.data import load_specific_dataset, mcar, masked_fill
from pypots.imputation import SAITS
from pypots.utils.metrics import cal_mae
from pypots.regression import GRUD_REGRESSOR
# Data preprocessing. Tedious, but PyPOTS can help. 🤓
data = load_specific_dataset('physionet_2012')  # PyPOTS will automatically download and extract it.
X = data['X']
y = np.array(data['y'].sort_values('RecordID')['In-hospital_death'])

num_samples = len(X['RecordID'].unique())
X = X.drop('RecordID', axis = 1)
X = StandardScaler().fit_transform(X.to_numpy())
X = X.reshape(num_samples, 48, -1)
X_intact, X, missing_mask, indicating_mask = mcar(X, 0.1) # hold out 10% observed values as ground truth
X = masked_fill(X, 1 - missing_mask, np.nan)
# Model training. This is PyPOTS showtime. 💪

grud = GRUD_REGRESSOR(n_steps=48, n_features=37, rnn_hidden_size=128, n_classes=2, epochs=100)

# saits = SAITS(n_steps=48, n_features=37, n_layers=2, d_model=256, d_inner=128, n_head=4, d_k=64, d_v=64, dropout=0.1, epochs=2)
grud.fit(X)  # train the model. Here I use the whole dataset as the training set, because ground truth is not visible to the model.

x=1
# imputation = saits.impute(X)  # impute the originally-missing values and artificially-missing values
# mae = cal_mae(imputation, X_intact, indicating_mask)  # calculate mean absolute error on the ground truth (artificially-missing values)