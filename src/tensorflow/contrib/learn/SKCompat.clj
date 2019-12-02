(ns tensorflow.contrib.learn.SKCompat
  "Scikit learn wrapper for TensorFlow Learn Estimator.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce learn (import-module "tensorflow.contrib.learn"))

(defn SKCompat 
  "Scikit learn wrapper for TensorFlow Learn Estimator.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.
  "
  [ estimator ]
  (py/call-attr learn "SKCompat"  estimator ))

(defn fit 
  ""
  [self x y & {:keys [batch_size steps max_steps monitors]
                       :or {steps None max_steps None monitors None}} ]
    (py/call-attr-kw self "fit" [x y] {:batch_size batch_size :steps steps :max_steps max_steps :monitors monitors }))
(defn get-params 
  "Get parameters for this estimator.

    Args:
      deep: boolean, optional

        If `True`, will return the parameters for this estimator and
        contained subobjects that are estimators.

    Returns:
      params : mapping of string to any
      Parameter names mapped to their values.
    "
  [self   & {:keys [deep]} ]
    (py/call-attr-kw self "get_params" [] {:deep deep }))

(defn predict 
  ""
  [self x & {:keys [batch_size outputs]
                       :or {outputs None}} ]
    (py/call-attr-kw self "predict" [x] {:batch_size batch_size :outputs outputs }))

(defn score 
  ""
  [self x y & {:keys [batch_size steps metrics name]
                       :or {steps None metrics None name None}} ]
    (py/call-attr-kw self "score" [x y] {:batch_size batch_size :steps steps :metrics metrics :name name }))

(defn set-params 
  "Set the parameters of this estimator.

    The method works on simple estimators as well as on nested objects
    (such as pipelines). The former have parameters of the form
    ``<component>__<parameter>`` so that it's possible to update each
    component of a nested object.

    Args:
      **params: Parameters.

    Returns:
      self

    Raises:
      ValueError: If params contain invalid names.
    "
  [ self  ]
  (py/call-attr self "set_params"  self  ))
