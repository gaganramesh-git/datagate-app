"""AnalysisContext: everything one analysis run shares.

Analyzers receive a context instead of raw inputs so the expensive pieces —
the sampled feature matrices, the candidate/reference joint space, the
out-of-fold proxy predictions — are computed once and reused. Every sample is
seeded, so a run is reproducible end to end.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from datagate.features import ColumnPlan, Featurizer, embed, plan_for
from datagate.ingest.loader import KIND_CATEGORICAL, KIND_NUMERIC, Dataset
from datagate.scoring import sample_indices
from datagate.types import ProjectSpec, TaskType

MAX_ROWS = 20_000            # candidate rows used by model-based checks
MAX_REFERENCE_ROWS = 5_000   # reference rows used by joint checks
MIN_CLASS_SIZE = 6           # a class needs this many rows to be cross-validated


@dataclass
class View:
    """A seeded sample of one dataset, featurized."""

    idx: np.ndarray             # row positions in the source dataframe
    df: pl.DataFrame            # the sampled rows
    X: sp.csr_matrix
    featurizer: Featurizer
    plan: ColumnPlan
    text_cols: list[str]
    y: Optional[np.ndarray]     # labels (str) or targets (float); None without a target
    labeled: np.ndarray         # mask of rows whose y is usable


@dataclass
class Joint:
    """Candidate and reference samples in one shared feature space."""

    cand_idx: np.ndarray
    ref_idx: np.ndarray
    Xc: sp.csr_matrix
    Xr: sp.csr_matrix
    yc: Optional[np.ndarray]
    yr: Optional[np.ndarray]
    labeled_c: np.ndarray
    labeled_r: np.ndarray
    featurizer: Featurizer
    emb_c: np.ndarray
    emb_r: np.ndarray

    @property
    def has_labels(self) -> bool:
        return self.yc is not None and self.yr is not None


@dataclass
class OOF:
    """Out-of-fold class probabilities from the proxy classifier."""

    pos: np.ndarray        # positions within the view
    y: np.ndarray          # given labels for those rows
    classes: np.ndarray
    proba: np.ndarray
    pred: np.ndarray
    accuracy: float
    balanced_accuracy: float
    chance: float


class AnalysisContext:
    def __init__(
        self,
        dataset: Dataset,
        spec: ProjectSpec,
        reference: Optional[Dataset] = None,
        seed: int = 7,
        max_rows: int = MAX_ROWS,
        max_reference_rows: int = MAX_REFERENCE_ROWS,
    ) -> None:
        self.dataset = dataset
        self.spec = spec
        self.reference = reference
        self.seed = seed
        self.max_rows = max_rows
        self.max_reference_rows = max_reference_rows
        self._cache: dict[str, Any] = {}
        self._lock = threading.RLock()

    # -- task -----------------------------------------------------------------

    @property
    def target(self) -> Optional[str]:
        t = self.spec.target_column
        return t if t and t in self.dataset.df.columns else None

    @property
    def task_kind(self) -> Optional[str]:
        """'classification' | 'regression' | None, reconciling spec and data."""
        target = self.target
        if target is None:
            return None
        series = self.dataset.df[target]
        is_numeric = series.dtype.is_numeric()
        if self.spec.task_type is TaskType.REGRESSION and is_numeric:
            return "regression"
        if self.spec.task_type is TaskType.CLASSIFICATION:
            return "classification"
        kind = self.dataset.kinds().get(target)
        if kind == KIND_CATEGORICAL or (is_numeric and series.n_unique() <= 20):
            return "classification"
        if kind == KIND_NUMERIC:
            return "regression"
        return None

    # -- shared, cached computations ---------------------------------------------

    def _once(self, key: str, build: Callable[[], Any]) -> Any:
        with self._lock:
            if key not in self._cache:
                self._cache[key] = build()
            return self._cache[key]

    def view(self) -> View:
        return self._once("view", self._build_view)

    def joint(self) -> Optional[Joint]:
        if self.reference is None:
            return None
        return self._once("joint", self._build_joint)

    def oof(self) -> Optional[OOF]:
        return self._once("oof", self._build_oof)

    def embeddings(self) -> np.ndarray:
        """Dense embeddings of the candidate view rows."""
        def build() -> np.ndarray:
            v = self.view()
            return embed([v.X], normalize=v.featurizer.has_text, seed=self.seed)[0]
        return self._once("emb", build)

    # -- builders -----------------------------------------------------------------

    def _targets(self, df: pl.DataFrame, target: Optional[str]
                 ) -> tuple[Optional[np.ndarray], np.ndarray]:
        if target is None or target not in df.columns:
            return None, np.zeros(df.height, dtype=bool)
        kind = self.task_kind
        if kind == "regression":
            y = df[target].cast(pl.Float64, strict=False).to_numpy().astype(float)
            return y, np.isfinite(y)
        raw = df[target].cast(pl.String, strict=False).to_list()
        y = np.array(["" if v is None else v for v in raw], dtype=object)
        return y, np.array([v is not None for v in raw], dtype=bool)

    def _build_view(self) -> View:
        ds, target = self.dataset, self.target
        idx = sample_indices(ds.n_rows, self.max_rows, self.seed)
        df = ds.df[idx] if len(idx) < ds.n_rows else ds.df
        plan = plan_for(ds, exclude=[target] if target else [])
        featurizer = Featurizer(plan.numeric, plan.categorical).fit([(df, plan.text)])
        X = featurizer.transform(df, plan.text)
        y, labeled = self._targets(df, target)
        return View(idx=idx, df=df, X=X, featurizer=featurizer, plan=plan,
                    text_cols=plan.text, y=y, labeled=labeled)

    def _build_joint(self) -> Joint:
        cand, ref, target = self.dataset, self.reference, self.target
        assert ref is not None
        view = self.view()
        ref_idx = sample_indices(ref.n_rows, self.max_reference_rows, self.seed + 1)
        ref_df = ref.df[ref_idx] if len(ref_idx) < ref.n_rows else ref.df

        exclude = [target] if target else []
        cand_plan = plan_for(cand, exclude)
        ref_text = ref.text_columns(exclude)
        featurizer = Featurizer(cand_plan.numeric, cand_plan.categorical).fit(
            [(view.df, cand_plan.text), (ref_df, ref_text)])
        Xc = featurizer.transform(view.df, cand_plan.text)
        Xr = featurizer.transform(ref_df, ref_text)
        yr, labeled_r = self._targets(ref_df, target)
        emb_c, emb_r = embed([Xc, Xr], normalize=featurizer.has_text, seed=self.seed)
        return Joint(cand_idx=view.idx, ref_idx=ref_idx, Xc=Xc, Xr=Xr,
                     yc=view.y if yr is not None else None, yr=yr,
                     labeled_c=view.labeled, labeled_r=labeled_r,
                     featurizer=featurizer, emb_c=emb_c, emb_r=emb_r)

    def model_matrix(self, view: Optional[View] = None):
        """The matrix the proxy model trains on: sparse for text, dense for tabular."""
        v = view or self.view()
        if v.featurizer.has_text:
            return v.X
        return v.featurizer.transform_dense(v.df)

    def classifier(self, has_text: bool):
        if has_text:
            return LogisticRegression(C=5.0, max_iter=300)
        return HistGradientBoostingClassifier(max_iter=120, early_stopping=False,
                                              random_state=self.seed)

    def regressor(self, has_text: bool):
        if has_text:
            return Ridge(alpha=1.0)
        return HistGradientBoostingRegressor(max_iter=120, early_stopping=False,
                                             random_state=self.seed)

    def _build_oof(self) -> Optional[OOF]:
        if self.task_kind != "classification":
            return None
        v = self.view()
        if v.y is None or v.X.shape[1] == 0:
            return None
        labels, counts = np.unique(v.y[v.labeled], return_counts=True)
        keep = set(labels[counts >= MIN_CLASS_SIZE])
        if len(keep) < 2 or len(keep) > 200:
            return None
        mask = v.labeled & np.isin(v.y, list(keep))
        pos = np.where(mask)[0]
        if len(pos) < 60:
            return None

        y = v.y[pos].astype(str)
        X = self.model_matrix(v)[pos]
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.seed)
        proba = cross_val_predict(self.classifier(v.featurizer.has_text), X, y,
                                  cv=cv, method="predict_proba")
        classes = np.unique(y)
        pred = classes[proba.argmax(axis=1)]
        accuracy = float((pred == y).mean())
        per_class = [float((pred[y == c] == c).mean()) for c in classes]
        return OOF(pos=pos, y=y, classes=classes, proba=proba, pred=pred,
                   accuracy=accuracy, balanced_accuracy=float(np.mean(per_class)),
                   chance=1.0 / len(classes))
