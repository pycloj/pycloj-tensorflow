(ns tensorflow.contrib.constrained-optimization.ConstrainedOptimizer
  "Base class representing a constrained optimizer.

  A ConstrainedOptimizer wraps a tf.compat.v1.train.Optimizer (or more than
  one), and applies it to a ConstrainedMinimizationProblem. Unlike a
  tf.compat.v1.train.Optimizer, which takes a tensor to minimize as a parameter
  to its minimize() method, a constrained optimizer instead takes a
  ConstrainedMinimizationProblem.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce constrained-optimization (import-module "tensorflow.contrib.constrained_optimization"))

(defn ConstrainedOptimizer 
  "Base class representing a constrained optimizer.

  A ConstrainedOptimizer wraps a tf.compat.v1.train.Optimizer (or more than
  one), and applies it to a ConstrainedMinimizationProblem. Unlike a
  tf.compat.v1.train.Optimizer, which takes a tensor to minimize as a parameter
  to its minimize() method, a constrained optimizer instead takes a
  ConstrainedMinimizationProblem.
  "
  [ optimizer ]
  (py/call-attr constrained-optimization "ConstrainedOptimizer"  optimizer ))

(defn minimize 
  "Returns an `Operation` for minimizing the constrained problem.

    This method combines the functionality of `minimize_unconstrained` and
    `minimize_constrained`. If global_step < unconstrained_steps, it will
    perform an unconstrained update, and if global_step >= unconstrained_steps,
    it will perform a constrained update.

    The reason for this functionality is that it may be best to initialize the
    constrained optimizer with an approximate optimum of the unconstrained
    problem.

    Args:
      minimization_problem: ConstrainedMinimizationProblem, the problem to
        optimize.
      unconstrained_steps: int, number of steps for which we should perform
        unconstrained updates, before transitioning to constrained updates.
      global_step: as in `tf.compat.v1.train.Optimizer`'s `minimize` method.
      var_list: as in `tf.compat.v1.train.Optimizer`'s `minimize` method.
      gate_gradients: as in `tf.compat.v1.train.Optimizer`'s `minimize` method.
      aggregation_method: as in `tf.compat.v1.train.Optimizer`'s `minimize`
        method.
      colocate_gradients_with_ops: as in `tf.compat.v1.train.Optimizer`'s
        `minimize` method.
      name: as in `tf.compat.v1.train.Optimizer`'s `minimize` method.
      grad_loss: as in `tf.compat.v1.train.Optimizer`'s `minimize` method.

    Returns:
      `Operation`, the train_op.

    Raises:
      ValueError: If unconstrained_steps is provided, but global_step is not.
    "
  [self minimization_problem unconstrained_steps global_step var_list & {:keys [gate_gradients aggregation_method colocate_gradients_with_ops name grad_loss]
                       :or {aggregation_method None name None grad_loss None}} ]
    (py/call-attr-kw self "minimize" [minimization_problem unconstrained_steps global_step var_list] {:gate_gradients gate_gradients :aggregation_method aggregation_method :colocate_gradients_with_ops colocate_gradients_with_ops :name name :grad_loss grad_loss }))

(defn minimize-constrained 
  "Returns an `Operation` for minimizing the constrained problem.

    Unlike `minimize_unconstrained`, this function attempts to find a solution
    that minimizes the `objective` portion of the minimization problem while
    satisfying the `constraints` portion.

    Args:
      minimization_problem: ConstrainedMinimizationProblem, the problem to
        optimize.
      global_step: as in `tf.compat.v1.train.Optimizer`'s `minimize` method.
      var_list: as in `tf.compat.v1.train.Optimizer`'s `minimize` method.
      gate_gradients: as in `tf.compat.v1.train.Optimizer`'s `minimize` method.
      aggregation_method: as in `tf.compat.v1.train.Optimizer`'s `minimize`
        method.
      colocate_gradients_with_ops: as in `tf.compat.v1.train.Optimizer`'s
        `minimize` method.
      name: as in `tf.compat.v1.train.Optimizer`'s `minimize` method.
      grad_loss: as in `tf.compat.v1.train.Optimizer`'s `minimize` method.

    Returns:
      `Operation`, the train_op.
    "
  [self minimization_problem global_step var_list & {:keys [gate_gradients aggregation_method colocate_gradients_with_ops name grad_loss]
                       :or {aggregation_method None name None grad_loss None}} ]
    (py/call-attr-kw self "minimize_constrained" [minimization_problem global_step var_list] {:gate_gradients gate_gradients :aggregation_method aggregation_method :colocate_gradients_with_ops colocate_gradients_with_ops :name name :grad_loss grad_loss }))

(defn minimize-unconstrained 
  "Returns an `Operation` for minimizing the unconstrained problem.

    Unlike `minimize_constrained`, this function ignores the `constraints` (and
    `proxy_constraints`) portion of the minimization problem entirely, and only
    minimizes `objective`.

    Args:
      minimization_problem: ConstrainedMinimizationProblem, the problem to
        optimize.
      global_step: as in `tf.compat.v1.train.Optimizer`'s `minimize` method.
      var_list: as in `tf.compat.v1.train.Optimizer`'s `minimize` method.
      gate_gradients: as in `tf.compat.v1.train.Optimizer`'s `minimize` method.
      aggregation_method: as in `tf.compat.v1.train.Optimizer`'s `minimize`
        method.
      colocate_gradients_with_ops: as in `tf.compat.v1.train.Optimizer`'s
        `minimize` method.
      name: as in `tf.compat.v1.train.Optimizer`'s `minimize` method.
      grad_loss: as in `tf.compat.v1.train.Optimizer`'s `minimize` method.

    Returns:
      `Operation`, the train_op.
    "
  [self minimization_problem global_step var_list & {:keys [gate_gradients aggregation_method colocate_gradients_with_ops name grad_loss]
                       :or {aggregation_method None name None grad_loss None}} ]
    (py/call-attr-kw self "minimize_unconstrained" [minimization_problem global_step var_list] {:gate_gradients gate_gradients :aggregation_method aggregation_method :colocate_gradients_with_ops colocate_gradients_with_ops :name name :grad_loss grad_loss }))

(defn optimizer 
  "Returns the `tf.compat.v1.train.Optimizer` used for optimization."
  [ self ]
    (py/call-attr self "optimizer"))
