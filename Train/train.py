
# train.py

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np

def train_models(X_train, y_train):
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), weights))
    scale_pos_weight = weights[1] / weights[0]

    log_model = LogisticRegression(max_iter=1000, class_weight=class_weight_dict)
    log_model.fit(X_train, y_train)

    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight)
    xgb_model.fit(X_train, y_train)

    return log_model, xgb_model
