(ns tensorflow.contrib.keras.api.keras.wrappers.scikit-learn.KerasRegressor
  "Implementation of the scikit-learn regressor API for Keras.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce scikit-learn (import-module "tensorflow.contrib.keras.api.keras.wrappers.scikit_learn"))

(defn KerasRegressor 
  "Implementation of the scikit-learn regressor API for Keras.
  "
  [ build_fn ]
  (py/call-attr scikit-learn "KerasRegressor"  build_fn ))

(defn check-params 
  "Checks for user typos in `params`.

    Arguments:
        params: dictionary; the parameters to be checked

    Raises:
        ValueError: if any member of `params` is not a valid argument.
    "
  [ self params ]
  (py/call-attr self "check_params"  self params ))

(defn filter-sk-params 
  "Filters `sk_params` and returns those in `fn`'s arguments.

    Arguments:
        fn : arbitrary function
        override: dictionary, values to override `sk_params`

    Returns:
        res : dictionary containing variables
            in both `sk_params` and `fn`'s arguments.
    "
  [ self fn override ]
  (py/call-attr self "filter_sk_params"  self fn override ))

(defn fit 
  "Constructs a new model with `build_fn` & fit the model to `(x, y)`.

    Arguments:
        x : array-like, shape `(n_samples, n_features)`
            Training samples where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
            True labels for `x`.
        **kwargs: dictionary arguments
            Legal arguments are the arguments of `Sequential.fit`

    Returns:
        history : object
            details about the training history at each epoch.
    "
  [ self x y ]
  (py/call-attr self "fit"  self x y ))

(defn get-params 
  "Gets parameters for this estimator.

    Arguments:
        **params: ignored (exists for API compatibility).

    Returns:
        Dictionary of parameter names mapped to their values.
    "
  [ self  ]
  (py/call-attr self "get_params"  self  ))

(defn predict 
  "Returns predictions for the given test data.

    Arguments:
        x: array-like, shape `(n_samples, n_features)`
            Test samples where `n_samples` is the number of samples
            and `n_features` is the number of features.
        **kwargs: dictionary arguments
            Legal arguments are the arguments of `Sequential.predict`.

    Returns:
        preds: array-like, shape `(n_samples,)`
            Predictions.
    "
  [ self x ]
  (py/call-attr self "predict"  self x ))

(defn score 
  "Returns the mean loss on the given test data and labels.

    Arguments:
        x: array-like, shape `(n_samples, n_features)`
            Test samples where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y: array-like, shape `(n_samples,)`
            True labels for `x`.
        **kwargs: dictionary arguments
            Legal arguments are the arguments of `Sequential.evaluate`.

    Returns:
        score: float
            Mean accuracy of predictions on `x` wrt. `y`.
    "
  [ self x y ]
  (py/call-attr self "score"  self x y ))

(defn set-params 
  "Sets the parameters of this estimator.

    Arguments:
        **params: Dictionary of parameter names mapped to their values.

    Returns:
        self
    "
  [ self  ]
  (py/call-attr self "set_params"  self  ))
