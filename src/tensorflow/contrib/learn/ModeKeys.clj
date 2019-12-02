(ns tensorflow.contrib.learn.ModeKeys
  "Standard names for model modes (deprecated).

  THIS CLASS IS DEPRECATED.

  The following standard keys are defined:

  * `TRAIN`: training mode.
  * `EVAL`: evaluation mode.
  * `INFER`: inference mode.
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

(defn ModeKeys 
  "Standard names for model modes (deprecated).

  THIS CLASS IS DEPRECATED.

  The following standard keys are defined:

  * `TRAIN`: training mode.
  * `EVAL`: evaluation mode.
  * `INFER`: inference mode.
  "
  [  ]
  (py/call-attr learn "ModeKeys"  ))
