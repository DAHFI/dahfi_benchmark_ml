"""
Clinical features (known before delivery). OPTIONAL: they are not used in the benchmark of
the paper, which relies only on the FHR signal.

To avoid data leakage, this module returns the clinical data RAW, with their missing
values. Imputation and scaling depend on the whole set of records, so they cannot be done
here: if they were fitted on all records, the test records would influence the training
values [1, 2].

Preprocessing is done inside the model Pipeline with make_clinical_preprocessor(), so that
the median, the mode and the scaling are fitted only on the training data of each
cross-validation fold.

Example:
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression

    names, X_clin = get_clinical_features(ids, "clinical.csv", ["Age", "Gest. weeks"], ["Parity"],
                                          get_names=True)
    model = Pipeline([
        ("prep", make_clinical_preprocessor(["Age", "Gest. weeks"], ["Parity"])),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

References
[1] Kaufman S, Rosset S, Perlich C, Stitelman O. Leakage in data mining: formulation,
    detection, and avoidance. ACM Trans Knowl Discov Data. 2012;6(4):15.
[2] Hastie T, Tibshirani R, Friedman J. The Elements of Statistical Learning. 2nd ed.
    Springer; 2009. Section 7.10.2 ("The wrong and right way to do cross-validation").
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_clinical_features(ids, csv_path, numeric_cols, binary_cols, get_names=False):
    """
    Returns the raw clinical features of `ids`, in the order numeric_cols + binary_cols.

    csv_path : CSV (optionally gzip-compressed) whose first column is the record id.
    Nothing is imputed: missing values, ids that are not in the CSV and columns that do not
    exist are left as NaN (with a warning).
    """
    data = pd.read_csv(csv_path, index_col=0, compression="infer")
    columns = list(numeric_cols) + list(binary_cols)

    missing_cols = [c for c in columns if c not in data.columns]
    if missing_cols:
        print(f"WARNING clinical: columns not in {csv_path} (left as NaN): {missing_cols}")

    # Ids are compared as text, so it does not matter whether they are numbers or strings
    data.index = data.index.astype(str)
    ids_str = [str(i) for i in ids]
    missing_ids = [i for i in ids_str if i not in data.index]
    if missing_ids:
        print(f"WARNING clinical: {len(missing_ids)} ids without clinical data (left as NaN): "
              f"{missing_ids[:20]}" + (" ..." if len(missing_ids) > 20 else ""))

    X = (data.reindex(index=ids_str, columns=columns)
         .apply(pd.to_numeric, errors="coerce")
         .to_numpy(dtype=float))
    return (columns, X) if get_names else X


def make_clinical_preprocessor(numeric_cols, binary_cols, remainder="passthrough"):
    """
    UNFITTED preprocessor to be placed inside the model Pipeline.

        numeric: median imputation (robust to extreme values) + standardisation
        binary : mode imputation

    It works by column name, so the input DataFrame can also contain the signal features;
    with remainder="passthrough" those columns pass unchanged (add their own imputer if
    they have NaN).
    """
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    binary = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])
    return ColumnTransformer(
        transformers=[("numeric", numeric, list(numeric_cols)), ("binary", binary, list(binary_cols))],
        remainder=remainder,
    )
