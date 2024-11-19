import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def insurance_score(y_true, y_pred, weights=None):
    """
    Calculate a combined score for insurance prediction models incorporating:
    - Normalized Gini Coefficient
    - Zero prediction accuracy
    - RMSE on non-zero values
    - MAE on non-zero values
    
    Parameters:
    y_true: array-like of true values
    y_pred: array-like of predicted values
    weights: dict with weights for each metric. Default weights if None provided
    
    Returns:
    float: combined score (higher is better)
    dict: individual component scores
    """
    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Default weights if none provided
    if weights is None:
        weights = {
            'gini': 0.3,
            'zero_acc': 0.3,
            'rmse': 0.2,
            'mae': 0.2
        }
    
    # 1. Calculate Normalized Gini Coefficient
    def normalized_gini(y_true, y_pred):
        # Sort by predicted values
        n = len(y_true)
        indices = np.argsort(y_pred)
        y_true_sorted = y_true[indices]
        
        # Calculate cumulative sums
        cum_actual = np.cumsum(y_true_sorted) / np.sum(y_true_sorted)
        cum_random = np.linspace(0, 1, n)
        
        # Calculate Gini coefficient
        gini = np.sum(cum_actual - cum_random) / n
        return max(0, gini * 2)  # Scale to 0-1 range and ensure non-negative
    
    # 2. Calculate Zero Prediction Accuracy
    zero_mask_true = (y_true == 0)
    zero_mask_pred = (y_pred == 0)
    zero_accuracy = np.mean(zero_mask_true == zero_mask_pred)
    
    # 3. Calculate RMSE on non-zero values
    non_zero_mask = (y_true != 0)
    if np.any(non_zero_mask):
        rmse_non_zero = np.sqrt(mean_squared_error(
            y_true[non_zero_mask], 
            y_pred[non_zero_mask]
        ))
        # Convert RMSE to a 0-1 scale where higher is better
        rmse_score = 1 / (1 + rmse_non_zero)
    else:
        rmse_score = 1.0
    
    # 4. Calculate MAE on non-zero values
    if np.any(non_zero_mask):
        mae_non_zero = mean_absolute_error(
            y_true[non_zero_mask], 
            y_pred[non_zero_mask]
        )
        # Convert MAE to a 0-1 scale where higher is better
        mae_score = 1 / (1 + mae_non_zero)
    else:
        mae_score = 1.0
    
    # Calculate Gini coefficient
    gini_score = normalized_gini(y_true, y_pred)
    
    # Combine all scores using weights
    combined_score = (
        weights['gini'] * gini_score +
        weights['zero_acc'] * zero_accuracy +
        weights['rmse'] * rmse_score +
        weights['mae'] * mae_score
    )
    
    # Create dictionary with all scores
    scores_dict = {
        'combined_score': combined_score,
        'gini': gini_score,
        'zero_accuracy': zero_accuracy,
        'rmse_score': rmse_score,
        'mae_score': mae_score,
        'raw_rmse_non_zero': 1/rmse_score - 1 if rmse_score < 1 else 0,
        'raw_mae_non_zero': 1/mae_score - 1 if mae_score < 1 else 0
    }
    
    return combined_score
