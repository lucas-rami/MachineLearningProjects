# -*- coding: utf-8 -*-
"""A function to split the data and work with multiple models
    
    Example use: 
        y_pred = multi_model_datasets_creation(y_tr,tx_tr_modified,tx_te_modified,22,build_poly,[6],ridge_regression,[lambda_])
"""
import numpy as np
import proj1_helpers

def multi_model_datasets_creation(y_tr,tx_tr,tx_te,feature_column_index,preprocess_func,preprocess_func_args,solve_function,solve_function_args):
    '''Return the prediction given a data processing function (like build_poly)'''
    categorization_values, _ = np.unique(tx_tr[:,feature_column_index], return_counts=True)
    num_models = len(categorization_values)
    
    tx_tr_array = list()
    y_tr_array = list()
    
    tx_te_array = list()
    
    tx_tr_preprocess_array = list()
    tx_te_preprocess_array = list()
    
    weights_array = list()
    loss_array = list()
    y_pred_array = list()
    idx_tr_array = list()
    idx_te_array = list()
    
    y_pred = np.zeros(tx_te.shape[0])
    
    for i in range(0,num_models):
        tx_tr_array.append(tx_tr[tx_tr[:, feature_column_index] == categorization_values[i], :])
        idx_tr_array.append(np.where(tx_tr[:, feature_column_index] == categorization_values[i]))
        y_tr_array.append(y_tr[idx_tr_array[i]])
        
        tx_te_array.append(tx_te[tx_te[:, feature_column_index] == categorization_values[i], :])
        idx_te_array.append(np.where(tx_te[:, feature_column_index] == categorization_values[i]))
        
        tx_tr_preprocess_array.append(preprocess_func(tx_tr_array[i],*preprocess_func_args))
        tx_te_preprocess_array.append(preprocess_func(tx_te_array[i],*preprocess_func_args))
        
        if(solve_function_args != None):
            weight,loss = solve_function(y_tr_array[i],tx_tr_preprocess_array[i],*solve_function_args)
            weights_array.append(weight)
            loss_array.append(loss)
        else:
            weight,loss = solve_function(y_tr_array[i],tx_tr_preprocess_array[i],*solve_function_args)
            weights_array.append(weight)
            loss_array.append(loss)
        
        y_pred_array.append(proj1_helpers.predict_labels(weights_array[i],tx_te_preprocess_array[i]))
    
        # we get the prediction back in the correct order
        
        y_pred[idx_te_array[i]] = y_pred_array[i]
    
    
    return y_pred