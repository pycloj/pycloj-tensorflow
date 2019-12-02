(ns tensorflow.contrib.mixed-precision.ExponentialUpdateLossScaleManager
  "Loss scale manager uses an exponential update strategy.

  In general, the strategy increases loss scale by a greater-than-one factor
  after encountering a consecutive series of steps with finite gradients;
  Similarly, it decreases the loss scale by a factor when the accumulated number
  of steps with non-finite (nan or inf) gradients are met. An update is not
  applied if its result is less than 1 or overflows the float32 dynamic range.

  The number of finite and non-finite steps are cleared every time the loss
  scale is changed. The condition to decrease the loss scale is looser than to
  increase it since the former does not require the steps to be consecutive.
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
(defn ExponentialUpdateLossScaleManager 
  "Loss scale manager uses an exponential update strategy.

  In general, the strategy increases loss scale by a greater-than-one factor
  after encountering a consecutive series of steps with finite gradients;
  Similarly, it decreases the loss scale by a factor when the accumulated number
  of steps with non-finite (nan or inf) gradients are met. An update is not
  applied if its result is less than 1 or overflows the float32 dynamic range.

  The number of finite and non-finite steps are cleared every time the loss
  scale is changed. The condition to decrease the loss scale is looser than to
  increase it since the former does not require the steps to be consecutive.
  "
  [init_loss_scale incr_every_n_steps  & {:keys [decr_every_n_nan_or_inf incr_ratio decr_ratio]} ]
    (py/call-attr-kw mixed-precision "ExponentialUpdateLossScaleManager" [init_loss_scale incr_every_n_steps] {:decr_every_n_nan_or_inf decr_every_n_nan_or_inf :incr_ratio incr_ratio :decr_ratio decr_ratio }))

(defn get-loss-scale 
  "Returns the loss scale."
  [ self  ]
  (py/call-attr self "get_loss_scale"  self  ))

(defn update-loss-scale 
  "Updates loss scale based on if gradients are finite in current step."
  [ self finite_grads ]
  (py/call-attr self "update_loss_scale"  self finite_grads ))
