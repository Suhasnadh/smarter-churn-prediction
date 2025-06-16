
# test.py

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import shap
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test, model_name):
    pred = model.predict(X_test)
    print(f"\n===== {model_name} =====")
    print("Accuracy:", accuracy_score(y_test, pred))
    print("F1 Score:", f1_score(y_test, pred))
    print("Classification Report:\n", classification_report(y_test, pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
    return pred

def explain_xgboost(model, X_train, X_test, feature_names):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, feature_names=feature_names)
