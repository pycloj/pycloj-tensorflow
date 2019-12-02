(ns tensorflow.-api.v1.compat.v1.train.experimental.LossScale
  "Loss scale base class.

  Loss scaling is a process that multiplies the loss by a multiplier called the
  loss scale, and divides each gradient by the same multiplier. The pseudocode
  for this process is:

  ```
  loss = ...
  loss *= loss_scale
  grads = gradients(loss, vars)
  grads /= loss_scale
  ```

  Mathematically, loss scaling has no effect, but can help avoid numerical
  underflow in intermediate gradients when float16 tensors are used for mixed
  precision training. By multiplying the loss, each intermediate gradient will
  have the same multiplier applied.

  Instances of this class represent a loss scale. Calling instances of this
  class returns the loss scale as a scalar float32 tensor, while method
  `update()` updates the loss scale depending on the values of the gradients.
  Optimizers use instances of this class to scale loss and gradients.
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

(defn LossScale 
  "Loss scale base class.

  Loss scaling is a process that multiplies the loss by a multiplier called the
  loss scale, and divides each gradient by the same multiplier. The pseudocode
  for this process is:

  ```
  loss = ...
  loss *= loss_scale
  grads = gradients(loss, vars)
  grads /= loss_scale
  ```

  Mathematically, loss scaling has no effect, but can help avoid numerical
  underflow in intermediate gradients when float16 tensors are used for mixed
  precision training. By multiplying the loss, each intermediate gradient will
  have the same multiplier applied.

  Instances of this class represent a loss scale. Calling instances of this
  class returns the loss scale as a scalar float32 tensor, while method
  `update()` updates the loss scale depending on the values of the gradients.
  Optimizers use instances of this class to scale loss and gradients.
  "
  [  ]
  (py/call-attr experimental "LossScale"  ))

(defn get-config 
  "Returns the config of this loss scale."
  [ self  ]
  (py/call-attr self "get_config"  self  ))

(defn update 
  "Updates the value of the loss scale.

    The loss scale will be potentially updated, based on the value of `grads`.
    The tensor returned by calling this class is only updated when this function
    is evaluated.

    In eager mode, this directly updates the loss scale, so that calling
    `__call__` will return the newly updated loss scale. In graph mode,
    this returns an op that, when evaluated, updates the loss scale.

    This function also returns a `should_apply_gradients` bool. If False,
    gradients should not be applied to the variables that step, as nonfinite
    gradients were found, and the loss scale has been be updated to reduce the
    chance of finding nonfinite gradients in the next step. Some loss scale
    classes will always return True, as they cannot adjust themselves in
    response to nonfinite gradients.

    When a DistributionStrategy is used, this function may only be called in a
    cross-replica context.

    Args:
      grads: A nested structure of unscaled gradients, each which is the
        gradient of the loss with respect to a weight. The gradients should have
        already been divided by the loss scale being before passed to this
        function. 'None' gradients are accepted, and are ignored.

    Returns:
      update_op: In eager mode, None. In graph mode, an op to update the loss
        scale.
      should_apply_gradients: Either a bool or a scalar boolean tensor. If
        False, the caller should skip applying `grads` to the variables this
        step.
    "
  [ self grads ]
  (py/call-attr self "update"  self grads ))
