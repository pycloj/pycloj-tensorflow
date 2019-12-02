(ns tensorflow.contrib.opt.DecoupledWeightDecayExtension
  "This class allows to extend optimizers with decoupled weight decay.

  It implements the decoupled weight decay described by Loshchilov & Hutter
  (https://arxiv.org/pdf/1711.05101.pdf), in which the weight decay is
  decoupled from the optimization steps w.r.t. to the loss function.
  For SGD variants, this simplifies hyperparameter search since it decouples
  the settings of weight decay and learning rate.
  For adaptive gradient algorithms, it regularizes variables with large
  gradients more than L2 regularization would, which was shown to yield better
  training loss and generalization error in the paper above.

  This class alone is not an optimizer but rather extends existing
  optimizers with decoupled weight decay. We explicitly define the two examples
  used in the above paper (SGDW and AdamW), but in general this can extend
  any OptimizerX by using
  `extend_with_weight_decay(OptimizerX, weight_decay=weight_decay)`.
  In order for it to work, it must be the first class the Optimizer with
  weight decay inherits from, e.g.

  ```python
  class AdamWOptimizer(DecoupledWeightDecayExtension, adam.AdamOptimizer):
    def __init__(self, weight_decay, *args, **kwargs):
      super(AdamWOptimizer, self).__init__(weight_decay, *args, **kwargs).
  ```

  Note that this extension decays weights BEFORE applying the update based
  on the gradient, i.e. this extension only has the desired behaviour for
  optimizers which do not depend on the value of'var' in the update step!

  Note: when applying a decay to the learning rate, be sure to manually apply
  the decay to the `weight_decay` as well. For example:

  ```python
    schedule =
    tf.compat.v1.train.piecewise_constant(tf.compat.v1.train.get_global_step(),
                                           [10000, 15000], [1e-0, 1e-1, 1e-2])
    lr = 1e-1 * schedule()
    wd = lambda: 1e-4 * schedule()

    # ...

    optimizer = tf.contrib.opt.MomentumWOptimizer(learning_rate=lr,
                                                  weight_decay=wd,
                                                  momentum=0.9,
                                                  use_nesterov=True)
  ```
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce opt (import-module "tensorflow.contrib.opt"))

(defn DecoupledWeightDecayExtension 
  "This class allows to extend optimizers with decoupled weight decay.

  It implements the decoupled weight decay described by Loshchilov & Hutter
  (https://arxiv.org/pdf/1711.05101.pdf), in which the weight decay is
  decoupled from the optimization steps w.r.t. to the loss function.
  For SGD variants, this simplifies hyperparameter search since it decouples
  the settings of weight decay and learning rate.
  For adaptive gradient algorithms, it regularizes variables with large
  gradients more than L2 regularization would, which was shown to yield better
  training loss and generalization error in the paper above.

  This class alone is not an optimizer but rather extends existing
  optimizers with decoupled weight decay. We explicitly define the two examples
  used in the above paper (SGDW and AdamW), but in general this can extend
  any OptimizerX by using
  `extend_with_weight_decay(OptimizerX, weight_decay=weight_decay)`.
  In order for it to work, it must be the first class the Optimizer with
  weight decay inherits from, e.g.

  ```python
  class AdamWOptimizer(DecoupledWeightDecayExtension, adam.AdamOptimizer):
    def __init__(self, weight_decay, *args, **kwargs):
      super(AdamWOptimizer, self).__init__(weight_decay, *args, **kwargs).
  ```

  Note that this extension decays weights BEFORE applying the update based
  on the gradient, i.e. this extension only has the desired behaviour for
  optimizers which do not depend on the value of'var' in the update step!

  Note: when applying a decay to the learning rate, be sure to manually apply
  the decay to the `weight_decay` as well. For example:

  ```python
    schedule =
    tf.compat.v1.train.piecewise_constant(tf.compat.v1.train.get_global_step(),
                                           [10000, 15000], [1e-0, 1e-1, 1e-2])
    lr = 1e-1 * schedule()
    wd = lambda: 1e-4 * schedule()

    # ...

    optimizer = tf.contrib.opt.MomentumWOptimizer(learning_rate=lr,
                                                  weight_decay=wd,
                                                  momentum=0.9,
                                                  use_nesterov=True)
  ```
  "
  [ weight_decay ]
  (py/call-attr opt "DecoupledWeightDecayExtension"  weight_decay ))

(defn apply-gradients 
  "Apply gradients to variables and decay the variables.

    This function is the same as Optimizer.apply_gradients except that it
    allows to specify the variables that should be decayed using
    decay_var_list. If decay_var_list is None, all variables in var_list
    are decayed.

    For more information see the documentation of Optimizer.apply_gradients.

    Args:
      grads_and_vars: List of (gradient, variable) pairs as returned by
        `compute_gradients()`.
      global_step: Optional `Variable` to increment by one after the variables
        have been updated.
      name: Optional name for the returned operation.  Default to the name
        passed to the `Optimizer` constructor.
      decay_var_list: Optional list of decay variables.

    Returns:
      An `Operation` that applies the specified gradients. If `global_step`
      was not None, that operation also increments `global_step`.
    "
  [ self grads_and_vars global_step name decay_var_list ]
  (py/call-attr self "apply_gradients"  self grads_and_vars global_step name decay_var_list ))

(defn minimize 
  "Add operations to minimize `loss` by updating `var_list` with decay.

    This function is the same as Optimizer.minimize except that it allows to
    specify the variables that should be decayed using decay_var_list.
    If decay_var_list is None, all variables in var_list are decayed.

    For more information see the documentation of Optimizer.minimize.

    Args:
      loss: A `Tensor` containing the value to minimize.
      global_step: Optional `Variable` to increment by one after the variables
        have been updated.
      var_list: Optional list or tuple of `Variable` objects to update to
        minimize `loss`.  Defaults to the list of variables collected in the
        graph under the key `GraphKeys.TRAINABLE_VARIABLES`.
      gate_gradients: How to gate the computation of gradients.  Can be
        `GATE_NONE`, `GATE_OP`, or  `GATE_GRAPH`.
      aggregation_method: Specifies the method used to combine gradient terms.
        Valid values are defined in the class `AggregationMethod`.
      colocate_gradients_with_ops: If True, try colocating gradients with the
        corresponding op.
      name: Optional name for the returned operation.
      grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.
      decay_var_list: Optional list of decay variables.

    Returns:
      An Operation that updates the variables in `var_list`.  If `global_step`
      was not `None`, that operation also increments `global_step`.

    "
  [self loss global_step var_list & {:keys [gate_gradients aggregation_method colocate_gradients_with_ops name grad_loss decay_var_list]
                       :or {aggregation_method None name None grad_loss None decay_var_list None}} ]
    (py/call-attr-kw self "minimize" [loss global_step var_list] {:gate_gradients gate_gradients :aggregation_method aggregation_method :colocate_gradients_with_ops colocate_gradients_with_ops :name name :grad_loss grad_loss :decay_var_list decay_var_list }))
