from implementations import *
from proj1_helpers import *

print("Loading training data...")
y_tr,tx_tr,ids_tr = load_csv_data('../all/train.csv')
print("Done!")

print("Loading test data...")
_,tx_te,ids_te = load_csv_data('../all/test.csv')
print("Done!")

print("Solving least squares...")
weights,loss = least_squares(y_tr,tx_tr)
y_pred = predict_labels(weights,tx_te)
print("Done!")


filename = 'least_squares_implementation.csv'
create_csv_submission(ids_te,y_pred,filename)

print("CSV File saved as " + filename)