"""Testable functions embedded verbatim into standalone notebooks.

The app copies these functions as source text; it never executes a notebook.
"""


def preprocess_dataframe(raw, column_kinds, missing_markers, numeric_rules, drop_duplicates=True):
    """Normalize values without learning an imputation value from the dataset."""
    import numpy as np
    import pandas as pd

    df = raw.copy(deep=True)
    changes = []
    markers = {str(value).strip().casefold() for value in missing_markers}
    for column in df.columns:
        # object + np.nan works with sklearn; pandas StringDtype + pd.NA does not.
        text = df[column].astype("string").str.replace("\u200b", "", regex=False).str.strip()
        missing = text.isna() | text.str.casefold().isin(markers)
        values = text.astype(object).where(~missing, np.nan)
        before = int(missing.sum())
        invalid_type = invalid_range = 0
        kind = column_kinds.get(column, "text")
        if kind == "numeric":
            # Accept unambiguous thousands grouping (1,234.50), not decimal-comma guessing.
            grouped = text.str.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", na=False)
            values = values.where(~grouped, text.str.replace(",", "", regex=False))
            numbers = pd.to_numeric(values, errors="coerce").astype("float64")
            numbers = numbers.where(np.isfinite(numbers), np.nan)
            invalid_type = int((~missing & numbers.isna()).sum())
            rule = numeric_rules.get(column, {})
            bad = pd.Series(False, index=df.index)
            if "min" in rule:
                bad |= numbers.lt(rule["min"])
            if "max" in rule:
                bad |= numbers.gt(rule["max"])
            if rule.get("integer"):
                bad |= numbers.notna() & numbers.mod(1).ne(0)
            invalid_range = int(bad.sum())
            df[column] = numbers.mask(bad, np.nan)
        elif kind == "date":
            dates = pd.to_datetime(values, format="mixed", errors="coerce", utc=True)
            invalid_type = int((~missing & dates.isna()).sum())
            df[column] = dates
        else:
            # Keep leading zeros, real category spellings, and identifiers intact.
            df[column] = values
        changes.append({"column": column, "kind": kind, "missing_markers": before,
                        "invalid_type": invalid_type, "invalid_range": invalid_range,
                        "missing_after": int(df[column].isna().sum())})
    empty_rows = int(df.isna().all(axis=1).sum())
    df = df.loc[~df.isna().all(axis=1)].copy()
    duplicates = int(df.duplicated().sum())
    if drop_duplicates:
        df = df.drop_duplicates(keep="first")
    print(f"Rows: {len(raw)} → {len(df)}; empty rows removed: {empty_rows}; "
          f"duplicate rows {'removed' if drop_duplicates else 'retained'}: {duplicates}")
    return df.reset_index(drop=True), pd.DataFrame(changes)


def train_baseline(df, target, features, column_kinds, task="regression", use_xgboost=False):
    """Train only on usable targets; learn imputations/encodings from X_train."""
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder

    def skip(reason):
        print("Training skipped:", reason)
        return {"status": "skipped", "reason": reason}

    if not target or target not in df.columns:
        return skip("Choose an existing TARGET column in the configuration cell.")
    y = df[target].copy()
    if task == "regression":
        y = pd.to_numeric(y, errors="coerce").astype("float64")
        y = y.where(np.isfinite(y), np.nan)
    usable = y.notna()
    print(f"Excluded {int((~usable).sum())} missing/invalid target rows. Targets are never imputed.")
    y = y.loc[usable]
    if task == "classification":
        y = y.astype(str)
    feature_names = [name for name in features if name in df.columns and name != target
                     and column_kinds.get(name) != "identifier"]
    X = df.loc[usable, feature_names].copy()
    if len(X) < 4:
        return skip("At least four usable rows are needed for a train/test split.")
    if y.nunique() < 2:
        return skip("The target has fewer than two distinct usable values.")
    if not feature_names:
        return skip("No non-target, non-identifier features remain.")

    # Dates become finite numeric features, not mixed Timestamp/string objects.
    numeric_features, categorical_features = [], []
    for column in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[column]):
            dates = X.pop(column)
            for part in ("year", "month", "day"):
                name = f"{column}__{part}"
                while name in X.columns:
                    name += "_date"
                X[name] = getattr(dates.dt, part).astype("float64")
                numeric_features.append(name)
        elif pd.api.types.is_numeric_dtype(X[column]):
            X[column] = X[column].astype("float64").replace([np.inf, -np.inf], np.nan)
            numeric_features.append(column)
        else:
            text = X[column].astype("string")
            X[column] = text.astype(object).where(text.notna(), np.nan)
            categorical_features.append(column)

    test_size = max(2, int(np.ceil(len(X) * 0.2)))
    stratify = None
    if task == "classification":
        counts = y.value_counts()
        test_size = max(test_size, len(counts))
        if counts.min() < 2 or len(X) - test_size < len(counts):
            return skip("Each class needs at least two rows and representation in both splits.")
        stratify = y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify)
    # Select empty columns using training data only. Never peek at test-set values.
    usable_features = X_train.columns[X_train.notna().any()].tolist()
    numeric_features = [name for name in numeric_features if name in usable_features]
    categorical_features = [name for name in categorical_features if name in usable_features]
    if not usable_features:
        return skip("All features are empty in the training split.")
    transformers = []
    if numeric_features:
        transformers.append(("numeric", SimpleImputer(strategy="median"), numeric_features))
    if categorical_features:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="<missing>")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=100)),
        ]), categorical_features))
    preprocess = ColumnTransformer(transformers, remainder="drop")
    label_encoder = None
    if task == "classification":
        label_encoder = LabelEncoder().fit(y_train)
        y_train = label_encoder.transform(y_train)
        y_test = label_encoder.transform(y_test)
    if use_xgboost:
        from xgboost import XGBClassifier, XGBRegressor
        model_class = XGBRegressor if task == "regression" else XGBClassifier
        estimator = model_class(n_estimators=150, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=2)
    else:
        model_class = RandomForestRegressor if task == "regression" else RandomForestClassifier
        estimator = model_class(n_estimators=150, random_state=42, n_jobs=2)
    model = Pipeline([("preprocess", preprocess), ("model", estimator)])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    if task == "regression":
        metrics = {"MAE": float(mean_absolute_error(y_test, predictions)),
                   "R2": float(r2_score(y_test, predictions))}
    else:
        metrics = {"accuracy": float(accuracy_score(y_test, predictions))}
        print(classification_report(y_test, predictions, zero_division=0))
        print("Label mapping:", dict(enumerate(label_encoder.classes_)))
    print("Held-out metrics:", metrics)
    return {"status": "trained", "model": model, "metrics": metrics,
            "train_rows": len(X_train), "test_rows": len(X_test), "label_encoder": label_encoder}


def plot_data(df, specs):
    """Plot only cleaned values; skip empty/constant/non-numeric selections."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    for spec in specs:
        kind, x, y = spec["kind"], spec.get("x"), spec.get("y")
        numeric = df.select_dtypes(include="number").dropna(axis=1, how="all")
        numeric = numeric.loc[:, numeric.nunique() > 1].iloc[:, :12]
        if kind == "heatmap":
            if numeric.shape[1] < 2:
                print("Heatmap skipped: need two non-constant numeric columns.")
                continue
            plt.figure(figsize=(10, 7))
            sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="vlag", center=0)
        elif kind == "histogram":
            if x not in numeric.columns:
                continue
            plt.figure(figsize=(8, 4))
            df[x].dropna().hist(bins=25)
            plt.title(x)
        elif kind == "bar":
            if x not in df.columns or not len(df):
                continue
            counts = df[x].fillna("<missing>").astype(str).value_counts().head(15)
            counts.sort_values().plot.barh(title=x, figsize=(8, 4))
        elif kind == "scatter":
            if x not in numeric.columns or y not in numeric.columns:
                continue
            pairs = df[[x, y]].dropna()
            if pairs.empty:
                continue
            plt.figure(figsize=(8, 4))
            sns.scatterplot(data=pairs, x=x, y=y)
        plt.tight_layout()
        plt.show()
        plt.close("all")
