(ns tensorflow.contrib.constrained-optimization.AdditiveExternalRegretOptimizer
  "A `ConstrainedOptimizer` based on external-regret minimization.

  This `ConstrainedOptimizer` uses the given `tf.compat.v1.train.Optimizer`s to
  jointly minimize over the model parameters, and maximize over Lagrange
  multipliers, with the latter maximization using additive updates and an
  algorithm that minimizes external regret.

  For more specifics, please refer to:

  > Cotter, Jiang and Sridharan. \"Two-Player Games for Efficient Non-Convex
  > Constrained Optimization\".
  > [https://arxiv.org/abs/1804.06500](https://arxiv.org/abs/1804.06500)

  The formulation used by this optimizer--which is simply the usual Lagrangian
  formulation--can be found in Definition 1, and is discussed in Section 3. It
  is most similar to Algorithm 3 in Appendix C.3, with the two differences being
  that it uses proxy constraints (if they're provided) in the update of the
  model parameters, and uses `tf.compat.v1.train.Optimizer`s, instead of SGD,
  for the \"inner\" updates.
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

(defn AdditiveExternalRegretOptimizer 
  "A `ConstrainedOptimizer` based on external-regret minimization.

  This `ConstrainedOptimizer` uses the given `tf.compat.v1.train.Optimizer`s to
  jointly minimize over the model parameters, and maximize over Lagrange
  multipliers, with the latter maximization using additive updates and an
  algorithm that minimizes external regret.

  For more specifics, please refer to:

  > Cotter, Jiang and Sridharan. \"Two-Player Games for Efficient Non-Convex
  > Constrained Optimization\".
  > [https://arxiv.org/abs/1804.06500](https://arxiv.org/abs/1804.06500)

  The formulation used by this optimizer--which is simply the usual Lagrangian
  formulation--can be found in Definition 1, and is discussed in Section 3. It
  is most similar to Algorithm 3 in Appendix C.3, with the two differences being
  that it uses proxy constraints (if they're provided) in the update of the
  model parameters, and uses `tf.compat.v1.train.Optimizer`s, instead of SGD,
  for the \"inner\" updates.
  "
  [ optimizer constraint_optimizer maximum_multiplier_radius ]
  (py/call-attr constrained-optimization "AdditiveExternalRegretOptimizer"  optimizer constraint_optimizer maximum_multiplier_radius ))

(defn constraint-optimizer 
  "Returns the `tf.compat.v1.train.Optimizer` used for the Lagrange multipliers."
  [ self ]
    (py/call-attr self "constraint_optimizer"))

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
