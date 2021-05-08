import numpy as np
import sklearn.metrics as metrics
import warnings

def get_auc(labels, y):
        try:
            return metrics.roc_auc_score(labels.cpu().numpy(), y.detach().cpu().numpy(), average='weighted')
        except ValueError:
            return np.nan

def get_prc(labels, y):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        return metrics.average_precision_score(labels.cpu().numpy(), y.detach().cpu().numpy(), average='weighted')