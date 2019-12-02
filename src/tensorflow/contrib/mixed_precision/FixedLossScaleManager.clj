(ns tensorflow.contrib.mixed-precision.FixedLossScaleManager
  "Loss scale manager with a fixed loss scale.

  The loss scale is not updated for the lifetime of the class.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce mixed-precision (import-module "tensorflow.contrib.mixed_precision"))

(defn FixedLossScaleManager 
  "Loss scale manager with a fixed loss scale.

  The loss scale is not updated for the lifetime of the class.
  "
  [ loss_scale ]
  (py/call-attr mixed-precision "FixedLossScaleManager"  loss_scale ))

(defn get-loss-scale 
  ""
  [ self  ]
  (py/call-attr self "get_loss_scale"  self  ))

(defn update-loss-scale 
  ""
  [ self finite_grads ]
  (py/call-attr self "update_loss_scale"  self finite_grads ))
