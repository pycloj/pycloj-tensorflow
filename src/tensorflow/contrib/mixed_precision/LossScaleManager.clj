(ns tensorflow.contrib.mixed-precision.LossScaleManager
  "Abstract loss scale manager class.

  Loss scale managers with a different strategy should subclass this class.
  Loss scaling is a process that:

  1) Applies a multiplier on the loss before computing gradients, and
  2) Applies the reciprocal of the multiplier on the gradients before they are
     applied on variables.

  This class is used together with
  `tf.contrib.mixed_precision.LossScaleOptimizer` for mixed precision training
  (float32 variables and float16 ops) on Nvidia GPUs in order to achieve the
  same model quality as single precision training, with the benefits of
  potential higher throughput.

  See `tf.contrib.mixed_precision.LossScaleOptimizer` for more details.
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

(defn LossScaleManager 
  "Abstract loss scale manager class.

  Loss scale managers with a different strategy should subclass this class.
  Loss scaling is a process that:

  1) Applies a multiplier on the loss before computing gradients, and
  2) Applies the reciprocal of the multiplier on the gradients before they are
     applied on variables.

  This class is used together with
  `tf.contrib.mixed_precision.LossScaleOptimizer` for mixed precision training
  (float32 variables and float16 ops) on Nvidia GPUs in order to achieve the
  same model quality as single precision training, with the benefits of
  potential higher throughput.

  See `tf.contrib.mixed_precision.LossScaleOptimizer` for more details.
  "
  [  ]
  (py/call-attr mixed-precision "LossScaleManager"  ))

(defn get-loss-scale 
  "Returns the loss scale as a scalar `float32` tensor."
  [ self  ]
  (py/call-attr self "get_loss_scale"  self  ))

(defn update-loss-scale 
  "Updates loss scale based on if gradients are finite in current step.

    Args:
      finite_grads: bool scalar tensor indicating if all gradients are
        finite (i.e., not inf or nan).

    Returns:
      An op, when executed updates the loss scale. If eager execution is
      enabled, does not return anything.
    "
  [ self finite_grads ]
  (py/call-attr self "update_loss_scale"  self finite_grads ))
