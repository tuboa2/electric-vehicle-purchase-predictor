import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from typing import Any


class AdversarialValidator:
    """
    Diagnoses covariate shift between train and test distributions.
    Calculates ROC-AUC and ranks top drifting features.
    """

    @staticmethod
    def evaluate(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        drop_cols: list[str],
        seed: int = 42,
    ) -> dict[str, Any]:
        # Filter dropped columns
        cols_to_drop = set(drop_cols)
        features = [c for c in train_df.columns if c in test_df.columns and c not in cols_to_drop]

        if not features:
            raise ValueError("No common features found between train and test after exclusions")

        tr_sub = train_df[features].copy()
        te_sub = test_df[features].copy()

        tr_sub["__is_test__"] = 0
        te_sub["__is_test__"] = 1

        combined = pd.concat([tr_sub, te_sub], axis=0).reset_index(drop=True)
        X = combined[features]
        y = combined["__is_test__"].values

        # Handle categoricals
        X = X.copy()
        cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        for c in cat_cols:
            X[c] = X[c].astype("category")

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        oof_preds = np.zeros(len(combined))
        feature_importances = np.zeros(len(features))

        for tr_idx, va_idx in skf.split(X, y):
            X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
            X_va, y_va = X.iloc[va_idx], y[va_idx]

            clf = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.08,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                verbose=-1,
                n_jobs=-1,
            )
            clf.fit(
                X_tr,
                y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)],
            )

            oof_preds[va_idx] = clf.predict_proba(X_va)[:, 1]
            feature_importances += clf.feature_importances_ / skf.n_splits

        auc = float(roc_auc_score(y, oof_preds))

        drift_rankings = sorted(
            [{"feature": f, "importance": float(imp)} for f, imp in zip(features, feature_importances)],
            key=lambda x: x["importance"],
            reverse=True,
        )

        severity = (
            "CRITICAL"
            if auc >= 0.85
            else ("SUBSTANTIAL" if auc >= 0.70 else ("MILD" if auc >= 0.60 else "NEGLIGIBLE"))
        )

        return {
            "adversarial_auc": round(auc, 4),
            "severity": severity,
            "top_drift_features": drift_rankings[:10],
            "total_features_evaluated": len(features),
        }
