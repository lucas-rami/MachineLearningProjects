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

lambdas = np.logspace(-5, 0, 12)
k = 5
max_degree = 10
ws_best, best_degrees = find_best_poly(k, y_tr, tx_tr, ridge_regression, max_degree, lambdas)
print("Best degrees found!")

print("Making predictions...")
tx_te_diff_poly = build_poly_diff_degrees(tx_te, best_degrees)
y_pred = predict_labels(ws_best,tx_te_diff_poly)
print("Done!")

filename = 'results/best_poly_ridge.csv'
create_csv_submission(ids_te,y_pred,filename)
