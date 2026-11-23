"""
Classical models and hyperparameter grids for the nested cross-validation in train_models.py.

Design choices
--------------
1. Class imbalance: every model that supports it (SVM, LASSO, RIDGE, RF, DT, ANN) searches
   class_weight in {None, "balanced"} in the inner CV. "balanced" weights each class by
   n / (2 * n_class), so both classes contribute equally to the loss [1, 2]. KNN does not
   support class weights.
2. SVM: the grid is split by kernel, so gamma is only searched with the RBF kernel [3].
3. RF: n_estimators is fixed to 500 instead of being searched. More trees do not hurt the
   performance of a random forest, they only reduce the variance of its estimate [4].
   min_samples_leaf reaches 10, which matters with few positive cases. n_jobs=1 inside the
   forest: GridSearchCV already parallelises, and nesting both oversubscribes the CPU.
4. DT: min_samples_leaf reaches 20.
5. KNN: large k values (up to 31). With few positives, larger neighbourhoods give smoother
   probability estimates, which helps the AUC.
6. ANN: multilayer perceptron with dropout (MLPDropoutClassifier, below), trained under the
   same conditions as the other models (same pipeline: imputer + StandardScaler, same nested
   CV, partitions and seeds). Three architectures are searched as a hyperparameter,
   together with dropout [5], L2 regularisation and class_weight. Sizes are kept small
   because there are only ~550 records:
       small  : 1 hidden layer of 16 units
       medium : 2 hidden layers, 32 -> 16
       large  : 2 hidden layers, 64 -> 32
   (with ~30 input features the large network has ~4,100 weights). Early stopping on an
   internal validation split taken ONLY from the training data of each fit [6].

Grid sizes (with class_weight): SVM 62, LASSO 12, RIDGE 12, RF 64, DT 80, KNN 32, ANN 54.

References
[1] He H, Garcia EA. Learning from imbalanced data. IEEE Trans Knowl Data Eng.
    2009;21(9):1263-1284.
[2] King G, Zeng L. Logistic regression in rare events data. Polit Anal. 2001;9(2):137-163.
[3] Hsu CW, Chang CC, Lin CJ. A practical guide to support vector classification.
    Technical report, National Taiwan University; 2003 (updated 2016).
[4] Probst P, Boulesteix AL. To tune or not to tune the number of trees in random forest.
    J Mach Learn Res. 2018;18(181):1-18.
[5] Srivastava N, Hinton G, Krizhevsky A, Sutskever I, Salakhutdinov R. Dropout: a simple way
    to prevent neural networks from overfitting. J Mach Learn Res. 2014;15:1929-1958.
[6] Prechelt L. Early stopping - but when? In: Neural Networks: Tricks of the Trade.
    Springer; 1998. p. 55-69.
[7] Kingma DP, Ba J. Adam: a method for stochastic optimization. ICLR; 2015.
[8] He K, Zhang X, Ren S, Sun J. Delving deep into rectifiers. ICCV; 2015. (He initialisation)
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

CLASS_WEIGHT = [None, "balanced"]

# =============================================================================
# Neural network: multilayer perceptron with dropout (NumPy, scikit-learn compatible)
# =============================================================================
# scikit-learn's MLPClassifier has no dropout and no class weights, so this small network is
# implemented here with NumPy. Being a scikit-learn estimator, it goes through exactly the same
# Pipeline, GridSearchCV and nested CV as the other models.
ANN_SIZES = {
    "small": (16,),
    "medium": (32, 16),
    "large": (64, 32),
}


class MLPDropoutClassifier(ClassifierMixin, BaseEstimator):
    """Binary MLP: [Linear -> ReLU -> Dropout] x hidden layers -> Linear -> sigmoid.

    Training: mini-batch Adam [7] on the (optionally class-weighted) binary cross-entropy,
    with an L2 penalty on the weights (weight_decay) and inverted dropout after each hidden
    layer [5]. Early stopping [6]: a stratified validation_fraction of the TRAINING data
    passed to fit() is held out; training stops after `patience` epochs without improving
    the validation loss, and the weights of the best epoch are restored. Dropout is only
    active during training.

    Parameters
    ----------
    size : "small" | "medium" | "large"  (see ANN_SIZES)
    dropout : probability of dropping each hidden unit during training (0 = no dropout)
    weight_decay : L2 penalty coefficient on the weights (not on the biases)
    class_weight : None or "balanced" (each class weighted by n / (2 * n_class))
    """

    def __init__(
        self,
        size="medium",
        dropout=0.0,
        weight_decay=1e-4,
        class_weight=None,
        learning_rate=3e-3,
        batch_size=64,
        max_epochs=500,
        patience=50,
        validation_fraction=0.2,
        random_state=None,
    ):
        self.size = size
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.class_weight = class_weight
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state

    # ---- helpers -------------------------------------------------------------
    def _init_params(self, n_in, rng):
        sizes = [n_in, *ANN_SIZES[self.size], 1]
        params = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            W = rng.normal(0.0, np.sqrt(2.0 / a), size=(a, b))  # He initialisation [8]
            params += [W, np.zeros(b)]
        return params

    def _forward(self, X, params, rng=None):
        """Returns the logits and the cache for backprop. rng != None -> training (dropout on)."""
        h, cache = X, []
        n_layers = len(params) // 2
        for i in range(n_layers - 1):
            W, b = params[2 * i], params[2 * i + 1]
            z = h @ W + b
            a = np.maximum(z, 0.0)
            mask = None
            if rng is not None and self.dropout > 0:
                mask = (rng.random(a.shape) >= self.dropout) / (
                    1.0 - self.dropout
                )  # inverted dropout
                a = a * mask
            cache.append((h, z, mask))
            h = a
        logits = (h @ params[-2] + params[-1]).ravel()
        cache.append((h, None, None))
        return logits, cache

    @staticmethod
    def _loss(logits, y, w):
        # BCE with logits, numerically stable: log(1 + e^z) - y z
        bce = np.logaddexp(0.0, logits) - y * logits
        return float(np.sum(w * bce) / np.sum(w))

    def _sample_weights(self, y):
        if self.class_weight == "balanced":
            n, n_pos = len(y), y.sum()
            w_pos, w_neg = n / (2.0 * max(n_pos, 1)), n / (2.0 * max(n - n_pos, 1))
            return np.where(y == 1, w_pos, w_neg)
        if isinstance(self.class_weight, dict):
            return np.array(
                [self.class_weight.get(c, 1.0) for c in self.classes_[y]], float
            )
        return np.ones(len(y))

    # ---- scikit-learn API ---------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.classes_, y = np.unique(y, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError("MLPDropoutClassifier is for binary problems.")
        y = y.astype(float)
        rng = np.random.default_rng(self.random_state)
        self.n_features_in_ = X.shape[1]

        # Early-stopping split, taken only from the data given to fit (no leakage)
        idx = np.arange(len(y))
        stratify = y if np.min(np.bincount(y.astype(int))) >= 2 else None
        tr, va = train_test_split(
            idx,
            test_size=self.validation_fraction,
            stratify=stratify,
            random_state=int(rng.integers(2**31 - 1)),
        )
        w_all = self._sample_weights(y.astype(int))
        Xtr, ytr, wtr = X[tr], y[tr], w_all[tr]
        Xva, yva, wva = X[va], y[va], w_all[va]

        params = self._init_params(X.shape[1], rng)
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
        best_loss, best_params, best_epoch, wait = (
            np.inf,
            [p.copy() for p in params],
            0,
            0,
        )
        self.loss_curve_, self.validation_curve_ = [], []

        for epoch in range(self.max_epochs):
            order = rng.permutation(len(ytr))
            for start in range(0, len(order), self.batch_size):
                bi = order[start : start + self.batch_size]
                xb, yb, wb = Xtr[bi], ytr[bi], wtr[bi]
                logits, cache = self._forward(xb, params, rng)
                # Gradient of the weighted mean BCE w.r.t. the logits
                g = (wb * (1.0 / (1.0 + np.exp(-logits)) - yb) / np.sum(wb))[:, None]
                grads = [None] * len(params)
                h_last = cache[-1][0]
                grads[-2] = h_last.T @ g
                grads[-1] = g.sum(axis=0)
                delta = g @ params[-2].T
                for i in range(len(params) // 2 - 2, -1, -1):
                    h_in, z, mask = cache[i]
                    if mask is not None:
                        delta = delta * mask
                    delta = delta * (z > 0)
                    grads[2 * i] = h_in.T @ delta
                    grads[2 * i + 1] = delta.sum(axis=0)
                    if i > 0:
                        delta = delta @ params[2 * i].T
                # L2 penalty on the weights (not on the biases)
                for i in range(0, len(params), 2):
                    grads[i] = grads[i] + self.weight_decay * params[i]
                # Adam update
                t += 1
                for k, (p, gk) in enumerate(zip(params, grads)):
                    m[k] = b1 * m[k] + (1 - b1) * gk
                    v[k] = b2 * v[k] + (1 - b2) * gk * gk
                    mh, vh = m[k] / (1 - b1**t), v[k] / (1 - b2**t)
                    params[k] = p - self.learning_rate * mh / (np.sqrt(vh) + eps)

            tr_loss = self._loss(self._forward(Xtr, params)[0], ytr, wtr)
            va_loss = self._loss(self._forward(Xva, params)[0], yva, wva)
            self.loss_curve_.append(tr_loss)
            self.validation_curve_.append(va_loss)
            if va_loss < best_loss - 1e-6:
                best_loss, best_params, best_epoch, wait = (
                    va_loss,
                    [p.copy() for p in params],
                    epoch,
                    0,
                )
            else:
                wait += 1
                if wait >= self.patience:
                    break

        self.params_ = best_params
        self.best_epoch_ = best_epoch + 1
        self.n_iter_ = epoch + 1
        return self

    def decision_function(self, X):
        return self._forward(np.asarray(X, dtype=float), self.params_)[0]

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-self.decision_function(X)))
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


def get_classical_models(random_state=42, use_scaler=True):

    def make_pipeline(model):
        """StandardScaler (fitted inside each CV split) + model."""
        steps = [("scaler", StandardScaler())] if use_scaler else []
        steps.append(("model", model))
        return Pipeline(steps)

    return {
        # probability=True: predict_proba (Platt scaling) is used for the test AUC and Brier
        "SVM": make_pipeline(SVC(probability=True, random_state=random_state)),
        "LASSO": make_pipeline(
            LogisticRegression(
                penalty="l1",
                solver="liblinear",
                max_iter=5000,
                random_state=random_state,
            )
        ),
        "RIDGE": make_pipeline(
            LogisticRegression(
                penalty="l2",
                solver="liblinear",
                max_iter=5000,
                random_state=random_state,
            )
        ),
        # Trees are invariant to monotone transformations: no scaling
        "RF": RandomForestClassifier(
            n_estimators=500, n_jobs=1, random_state=random_state
        ),
        "DT": DecisionTreeClassifier(random_state=random_state),
        "KNN": make_pipeline(KNeighborsClassifier()),
        "ANN": make_pipeline(MLPDropoutClassifier(random_state=random_state)),
    }


def get_classical_param_grids():
    """
    Hyperparameter grids for the INNER cross-validation (best mean validation AUC).

    Parameters of models inside a Pipeline use the prefix "model__"; RF and DT, which are
    not wrapped in a Pipeline, do not. A grid can be a dict or a list of dicts (sub-grids,
    as in the SVM, where gamma only makes sense for the RBF kernel).
    """
    return {
        "SVM": [
            {  # Linear kernel: only C
                "model__kernel": ["linear"],
                "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
                "model__class_weight": CLASS_WEIGHT,
            },
            {  # RBF kernel: C and gamma ("scale" = 1 / n_features with standardised data)
                "model__kernel": ["rbf"],
                "model__C": [0.01, 0.1, 1, 10, 100],
                "model__gamma": ["scale", 0.001, 0.01, 0.1, 1],
                "model__class_weight": CLASS_WEIGHT,
            },
        ],
        # C = inverse of the regularisation strength (smaller C -> more coefficients = 0 in L1)
        "LASSO": {
            "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
            "model__class_weight": CLASS_WEIGHT,
        },
        "RIDGE": {
            "model__C": [0.001, 0.01, 0.1, 1, 10, 100],
            "model__class_weight": CLASS_WEIGHT,
        },
        "RF": {
            "max_depth": [None, 3, 5, 10],
            "min_samples_leaf": [1, 3, 5, 10],
            "max_features": ["sqrt", 0.5],
            "class_weight": CLASS_WEIGHT,
        },
        "DT": {
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 3, 5, 10],
            "min_samples_leaf": [1, 3, 5, 10, 20],
            "class_weight": CLASS_WEIGHT,
        },
        # KNN has no class_weight
        "KNN": {
            "model__n_neighbors": [3, 5, 7, 9, 11, 15, 21, 31],
            "model__weights": ["uniform", "distance"],
            "model__metric": ["euclidean", "manhattan"],
        },
        # Neural network: architecture, dropout and L2 regularisation as hyperparameters
        "ANN": {
            "model__size": ["small", "medium", "large"],
            "model__dropout": [0.0, 0.25, 0.5],
            "model__weight_decay": [1e-3, 1e-2, 1e-1],
            "model__class_weight": CLASS_WEIGHT,
        },
    }


def get_classical_model(model_name, random_state=42, use_scaler=True):
    name = model_name.upper()
    models = get_classical_models(random_state=random_state, use_scaler=use_scaler)
    grids = get_classical_param_grids()
    if name not in models:
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {', '.join(models)}."
        )
    return models[name], grids[name]
