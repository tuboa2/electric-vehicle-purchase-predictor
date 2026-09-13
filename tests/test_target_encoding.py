import numpy as np
import pandas as pd
from features.target_encoding import OutOfFoldTargetEncoder


def test_target_encoding_leak_free():
    train_df = pd.DataFrame({
        "cat_a": ["A", "B", "A", "B", "C", "A"],
        "target": [1, 0, 1, 0, 1, 0]
    })
    val_df = pd.DataFrame({
        "cat_a": ["A", "B", "C", "D"],  # D is unseen
    })
    y_tr = train_df["target"].to_numpy()

    encoder = OutOfFoldTargetEncoder(target_col="target", m_smoothing=10.0)
    tr_enc, va_enc, cols = encoder.fit_transform_fold(train_df, val_df, ["cat_a"], y_tr)

    assert "te_cat_a" in cols
    assert "te_cat_a" in tr_enc.columns
    assert "te_cat_a" in va_enc.columns
    assert not tr_enc["te_cat_a"].isnull().any()
    assert not va_enc["te_cat_a"].isnull().any()

    # Verify unseen category falls back to global prior
    global_prior = float(np.mean(y_tr))
    assert np.isclose(va_enc.loc[val_df["cat_a"] == "D", "te_cat_a"].values[0], global_prior)
