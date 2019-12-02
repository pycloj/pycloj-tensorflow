(ns tensorflow.-api.v1.train.experimental.FixedLossScale
  "Loss scale with a fixed value.

  The loss scale is not updated for the lifetime of instances of this class.
  A given instance of this class always returns the same number when called.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce experimental (import-module "tensorflow._api.v1.train.experimental"))

(defn FixedLossScale 
  "Loss scale with a fixed value.

  The loss scale is not updated for the lifetime of instances of this class.
  A given instance of this class always returns the same number when called.
  "
  [ loss_scale_value ]
  (py/call-attr experimental "FixedLossScale"  loss_scale_value ))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))

(defn update 
  ""
  [ self grads ]
  (py/call-attr self "update"  self grads ))
