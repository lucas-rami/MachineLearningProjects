from implementations import *
from proj1_helpers import *
from manipulate_data import *

print("Loading training data...")
y_tr,tx_tr,ids_tr = load_csv_data('../all/train.csv')
print("Done!")

print("Loading test data...")
_,tx_te,ids_te = load_csv_data('../all/test.csv')
print("Done!")

# Linear model, least squares solution

print("Solving least squares...")
weights,loss = least_squares(y_tr,tx_tr)
y_pred = predict_labels(weights,tx_te)
print("Done!")

filename = 'results/least_squares_implementation.csv'
create_csv_submission(ids_te,y_pred,filename)

print("CSV File saved as " + filename)

# Remove outliers

print("Removing outlier data from input data...")
tx_tr_noOut = tx_tr[np.invert(np.any(np.isin(tx_tr, -999), 1))]
y_tr_noOut = y_tr[np.invert(np.any(np.isin(tx_tr, -999), 1))]
print("Done!")

print("Solving least squares without outliers...")
weights_noOut,loss_noOut = least_squares(y_tr_noOut,tx_tr_noOut)
y_pred_noOut = predict_labels(weights_noOut,tx_te)
print("Done!")

filename = 'results/least_squares_implementation_no_outliers.csv'
create_csv_submission(ids_te,y_pred_noOut,filename)

print("CSV File saved as " + filename)

# Remove outlier features

print("Removing outlier features from input data...")
tx_tr_noOutFeatures = tx_tr[:, np.invert(np.any(np.isin(tx_tr, -999), 0))]
tx_te_noOutFeatures = tx_te[:, np.invert(np.any(np.isin(tx_tr, -999), 0))]
print("Done!")

print("Solving least squares without outlier features...")
weights_noOutFeatures,loss_noOutFeatures = least_squares(y_tr,tx_tr_noOutFeatures)
y_pred_noOutFeatures = predict_labels(weights_noOutFeatures,tx_te_noOutFeatures)
print("Done!")

filename = 'results/least_squares_implementation_no_outlier_features.csv'
create_csv_submission(ids_te,y_pred_noOut,filename)

print("CSV File saved as " + filename)

# Quadratic model, least squares solution

degree = 2

print("Generating degree 2 polynomial of data...")
tx_trSq = build_poly(tx_tr, degree)
tx_teSq = build_poly(tx_te, degree)
print("Done!")

print("Solving least squares for degree 2 polynomial of data...")
weightsSq,lossSq = least_squares(y_tr,tx_trSq)
y_pred = predict_labels(weightsSq,tx_teSq)
print("Done!")

filename = 'results/least_squares_implementation_Degree_2.csv'
create_csv_submission(ids_te,y_pred,filename)

print("CSV File saved as " + filename)

# Quadratic model, no outliers

print("Generating degree 2 polynomial of data...")
tx_trSq_noOut = build_poly(tx_tr_noOut, degree)
print("Done!")

print("Solving least squares for degree 2 polynomial of data...")
weightsSq_noOut,lossSq_noOut = least_squares(y_tr_noOut,tx_trSq_noOut)
y_predSq_noOut = predict_labels(weightsSq_noOut,tx_teSq)
print("Done!")

filename = 'results/least_squares_implementation_Degree_2_no_outliers.csv'
create_csv_submission(ids_te,y_predSq_noOut,filename)

print("CSV File saved as " + filename)