from implementations import *
from proj1_helpers import *
from manipulate_data import *
from best_poly import *

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

lambdas = np.logspace(-10, 0, 21)
k = 5
max_degree = 3
ws_best, best_degrees, best_te = find_best_poly(k, y_tr, tx_tr_modified, ridge_regression, max_degree, lambdas)
print("Best degrees found!, test error:", best_te)

print("Making predictions...")
tx_te_diff_poly = build_poly_diff_degrees(tx_te_modified, best_degrees)
y_pred = predict_labels(ws_best,tx_te_diff_poly)
print("Done!")

filename = 'results/best_poly_ridge_mean_outliers.csv'
create_csv_submission(ids_te,y_pred,filename)
