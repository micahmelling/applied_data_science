import numpy as np
import pandas as pd

from sklearn.metrics import brier_score_loss, log_loss


def calculate_log_loss_brier_score_mix(y_true, y_pred):
    """
    Calculates a weighted average of brier score and log loss. Log loss is applied for predictions off by greater
    than 0.5; otherwise, the brier score is applied.

    :param y_true: true labels
    :param y_pred: predicted positive probabilities
    """
    y_true = y_true.to_frame()
    y_true = y_true.reset_index(drop=True)
    y_true.columns = ['y_true']

    if isinstance(y_pred, np.ndarray):
        y_pred = pd.DataFrame(y_pred, columns=['y_pred'])
    else:
        y_pred = y_pred.to_frame()
        y_pred.columns = ['y_pred']
    y_pred = y_pred.reset_index(drop=True)
    temp_df = pd.concat([y_true, y_pred], axis=1)

    temp_df['abs_diff'] = abs(temp_df['y_true'] - temp_df['y_pred'])
    below_cutoff_df = temp_df.loc[temp_df['abs_diff'] <= 0.5]
    above_cutoff_df = temp_df.loc[temp_df['abs_diff'] > 0.5]
    brier_score = brier_score_loss(below_cutoff_df['y_true'], below_cutoff_df['y_pred'])
    log_loss_score = log_loss(above_cutoff_df['y_true'], above_cutoff_df['y_pred'], labels=[0, 1])
    score = (brier_score * (len(below_cutoff_df) / len(temp_df))) + \
            (log_loss_score * (len(above_cutoff_df) / len(temp_df)))
    return score
