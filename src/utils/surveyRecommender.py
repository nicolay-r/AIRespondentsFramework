from catboost import CatBoostClassifier
import numpy as np
import pandas as pd


class SurveyRecommender:
    def __init__(self, **catboost_params):
        self.catboost_params = {
            "loss_function": "MultiClass",
            "verbose": False,
            **catboost_params,
        }
        self.models = {}
        self.feature_columns = {}

    def fit(
        self,
        X: pd.DataFrame,
        *,
        feature_columns: list[str] | None = None,
        target_columns: list[str] | None = None,
    ):
        """
        Parameters
        ----------
        X : DataFrame
            Rows = respondents
            Columns = survey questions
            Values = survey answers
        feature_columns : list[str], optional
            Predictor question ids. Defaults to all non-target columns in X.
        target_columns : list[str], optional
            Target question ids to train. Defaults to all columns not in
            feature_columns.
        """

        self.models.clear()
        self.feature_columns.clear()

        if feature_columns is None and target_columns is None:
            for target in X.columns:
                features = [column for column in X.columns if column != target]
                self._fit_target(X, target, features)
            return self

        if target_columns is None:
            feature_set = set(feature_columns or [])
            target_columns = [
                column for column in X.columns if column not in feature_set
            ]
        if feature_columns is None:
            target_set = set(target_columns)
            feature_columns = [
                column for column in X.columns if column not in target_set
            ]

        for target in target_columns:
            if target not in X.columns:
                continue
            self._fit_target(X, target, list(feature_columns))

        return self

    def _fit_target(
        self,
        X: pd.DataFrame,
        target: str,
        features: list[str],
    ) -> None:
        y = X[target]
        if y.nunique(dropna=True) < 2:
            return

        X_train = X[features].copy().fillna("__MISSING__")
        cat_features = list(range(len(features)))
        mask = y.notna()

        model = CatBoostClassifier(**self.catboost_params)
        model.fit(
            X_train.loc[mask],
            y.loc[mask],
            cat_features=cat_features,
        )

        self.models[target] = model
        self.feature_columns[target] = features

    def top_features(
        self,
        target: str,
        *,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Return the most relevant question ids for a trained target."""
        model = self.models[target]
        features = self.feature_columns[target]
        importances = model.get_feature_importance()
        ranked = sorted(
            zip(features, importances),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[:top_k]

    def _feature_frame(self, X: pd.DataFrame, target: str) -> pd.DataFrame:
        features = self.feature_columns[target]
        missing = [column for column in features if column not in X.columns]
        if missing:
            raise KeyError(
                f"input is missing feature columns for {target!r}: {missing}"
            )
        return X[features].fillna("__MISSING__")

    def predict(
        self,
        X: pd.DataFrame | pd.Series,
        targets: list[str] | None = None,
    ) -> pd.DataFrame:
        """Predict target answers from respondents' other question answers."""
        if isinstance(X, pd.Series):
            X = X.to_frame().T

        if targets is None:
            targets = list(self.models.keys())

        predictions: dict[str, pd.Series] = {}
        for target in targets:
            if target not in self.models:
                raise KeyError(f"no model for target {target!r}")
            preds = np.asarray(self.models[target].predict(self._feature_frame(X, target)))
            predictions[target] = pd.Series(preds.reshape(-1), index=X.index)

        return pd.DataFrame(predictions, index=X.index)

    def predict_proba(
        self,
        X: pd.DataFrame | pd.Series,
        target: str,
    ) -> pd.DataFrame:
        """Return class probabilities for one trained target."""
        if isinstance(X, pd.Series):
            X = X.to_frame().T
        if target not in self.models:
            raise KeyError(f"no model for target {target!r}")

        model = self.models[target]
        probabilities = model.predict_proba(self._feature_frame(X, target))
        return pd.DataFrame(probabilities, index=X.index, columns=model.classes_)
