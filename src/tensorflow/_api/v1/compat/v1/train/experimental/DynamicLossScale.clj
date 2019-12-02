(ns tensorflow.-api.v1.compat.v1.train.experimental.DynamicLossScale
  "Loss scale that dynamically adjusts itself.

  Dynamic loss scaling works by adjusting the loss scale as training progresses.
  The goal is to keep the loss scale as high as possible without overflowing the
  gradients. As long as the gradients do not overflow, raising the loss scale
  never hurts.

  The algorithm starts by setting the loss scale to an initial value. Every N
  steps that the gradients are finite, the loss scale is increased by some
  factor. However, if a NaN or Inf gradient is found, the gradients for that
  step are not applied, and the loss scale is decreased by the factor. This
  process tends to keep the loss scale as high as possible without gradients
  overflowing.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce experimental (import-module "tensorflow._api.v1.compat.v1.train.experimental"))

(defn DynamicLossScale 
  "Loss scale that dynamically adjusts itself.

  Dynamic loss scaling works by adjusting the loss scale as training progresses.
  The goal is to keep the loss scale as high as possible without overflowing the
  gradients. As long as the gradients do not overflow, raising the loss scale
  never hurts.

  The algorithm starts by setting the loss scale to an initial value. Every N
  steps that the gradients are finite, the loss scale is increased by some
  factor. However, if a NaN or Inf gradient is found, the gradients for that
  step are not applied, and the loss scale is decreased by the factor. This
  process tends to keep the loss scale as high as possible without gradients
  overflowing.
  "
  [ & {:keys [initial_loss_scale increment_period multiplier]} ]
   (py/call-attr-kw experimental "DynamicLossScale" [] {:initial_loss_scale initial_loss_scale :increment_period increment_period :multiplier multiplier }))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))

(defn increment-period 
  ""
  [ self ]
    (py/call-attr self "increment_period"))

(defn initial-loss-scale 
  ""
  [ self ]
    (py/call-attr self "initial_loss_scale"))

(defn multiplier 
  ""
  [ self ]
    (py/call-attr self "multiplier"))

(defn update 
  "Updates loss scale based on if gradients are finite in current step."
  [ self grads ]
  (py/call-attr self "update"  self grads ))
