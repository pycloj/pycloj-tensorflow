(ns tensorflow.contrib.learn.ProblemType
  "Enum-like values for the type of problem that the model solves.

  THIS CLASS IS DEPRECATED.

  These values are used when exporting the model to produce the appropriate
  signature function for serving.

  The following values are supported:
    UNSPECIFIED: Produces a predict signature_fn.
    CLASSIFICATION: Produces a classify signature_fn.
    LINEAR_REGRESSION: Produces a regression signature_fn.
    LOGISTIC_REGRESSION: Produces a classify signature_fn.
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

(defn ProblemType 
  "Enum-like values for the type of problem that the model solves.

  THIS CLASS IS DEPRECATED.

  These values are used when exporting the model to produce the appropriate
  signature function for serving.

  The following values are supported:
    UNSPECIFIED: Produces a predict signature_fn.
    CLASSIFICATION: Produces a classify signature_fn.
    LINEAR_REGRESSION: Produces a regression signature_fn.
    LOGISTIC_REGRESSION: Produces a classify signature_fn.
  "
  [  ]
  (py/call-attr learn "ProblemType"  ))
