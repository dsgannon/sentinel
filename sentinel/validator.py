import numpy as np
from sentinel.core.metrics.classification import roc_auc, gini, ks_statistic, pr_auc
from sentinel.core.metrics.calibration import brier_score, expected_calibration_error
from sentinel.core.metrics.regression import mae, rmse
from sentinel.core.metrics.drift import psi
from sentinel.core.reporting.html_report import generate_report

class Validator:
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y

    def run(self):
        y_pred = self.model.predict_proba(self.X)[:, 1]
        metrics = {
            "auc": roc_auc(self.y, y_pred),
            "gini": gini(self.y, y_pred),
            "ks": ks_statistic(self.y, y_pred)[0],
            "brier_score": brier_score(self.y, y_pred),
            "ece": expected_calibration_error(self.y, y_pred),
            "pr_auc": pr_auc(self.y, y_pred)
        }
        return ValidationReport(metrics)

class ValidationReport:
    def __init__(self, metrics):
        self.metrics = metrics

    def save(self, path, model_name = "Model", validation_date = "", dataset_info = None):
        if dataset_info is None:
            dataset_info = {}
        generate_report(
            metrics=self.metrics,
            model_name=model_name,
            validation_date=validation_date,
            dataset_info=dataset_info,
            output_path=path
        )
