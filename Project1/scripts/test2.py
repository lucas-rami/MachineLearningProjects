from implementations import *
from proj1_helpers import *
from manipulate_data import *
from best_poly import *
from multi_model_splitter import *
from cross_validation import *

print("Loading training data...")
y_tr,tx_tr,ids_tr = load_csv_data('../all/train.csv')
print("Done!")

print("Loading test data...")
_,tx_te,ids_te = load_csv_data('../all/test.csv')
print("Done!")

tx_tr_modified = np.copy(tx_tr)
tx_te_modified = np.copy(tx_te)

for i in range(0,tx_tr.shape[1]):
    feature_mean_tr = tx_tr[:,i][np.where(tx_tr[:,i] != -999)].mean()
    feature_mean_te = tx_tr[:,i][np.where(tx_tr[:,i] != -999)].mean()
    tx_tr_modified[:,i][tx_tr[:,i] == -999] = feature_mean_tr
    tx_te_modified[:,i][tx_te[:,i] == -999] = feature_mean_te

print("Making predictions...")
lambdas = np.logspace(-10, 0, 21)
k = 5
y_pred = multi_model_datasets_creation(y_tr,tx_tr_modified,tx_te_modified,22,build_poly,[6],lambdas, k)
print("Done!")

filename = 'results/logistic_reg_squared.csv'
create_csv_submission(ids_te,y_pred,filename)
