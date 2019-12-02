(ns tensorflow.contrib.constrained-optimization.ConstrainedMinimizationProblem
  "Abstract class representing a `ConstrainedMinimizationProblem`.

  A ConstrainedMinimizationProblem consists of an objective function to
  minimize, and a set of constraint functions that are constrained to be
  nonpositive.

  In addition to the constraint functions, there may (optionally) be proxy
  constraint functions: a ConstrainedOptimizer will attempt to penalize these
  proxy constraint functions so as to satisfy the (non-proxy) constraints. Proxy
  constraints could be used if the constraints functions are difficult or
  impossible to optimize (e.g. if they're piecewise constant), in which case the
  proxy constraints should be some approximation of the original constraints
  that is well-enough behaved to permit successful optimization.
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

(defn ConstrainedMinimizationProblem 
  "Abstract class representing a `ConstrainedMinimizationProblem`.

  A ConstrainedMinimizationProblem consists of an objective function to
  minimize, and a set of constraint functions that are constrained to be
  nonpositive.

  In addition to the constraint functions, there may (optionally) be proxy
  constraint functions: a ConstrainedOptimizer will attempt to penalize these
  proxy constraint functions so as to satisfy the (non-proxy) constraints. Proxy
  constraints could be used if the constraints functions are difficult or
  impossible to optimize (e.g. if they're piecewise constant), in which case the
  proxy constraints should be some approximation of the original constraints
  that is well-enough behaved to permit successful optimization.
  "
  [  ]
  (py/call-attr constrained-optimization "ConstrainedMinimizationProblem"  ))

(defn constraints 
  "Returns the vector of constraint functions.

    Letting g_i be the ith element of the constraints vector, the ith constraint
    will be g_i <= 0.

    Returns:
      A tensor of constraint functions.
    "
  [ self ]
    (py/call-attr self "constraints"))

(defn num-constraints 
  "Returns the number of constraints.

    Returns:
      An int containing the number of constraints.

    Raises:
      ValueError: If the constraints (or proxy_constraints, if present) do not
        have fully-known shapes, OR if proxy_constraints are present, and the
        shapes of constraints and proxy_constraints are fully-known, but they're
        different.
    "
  [ self ]
    (py/call-attr self "num_constraints"))

(defn objective 
  "Returns the objective function.

    Returns:
      A 0d tensor that should be minimized.
    "
  [ self ]
    (py/call-attr self "objective"))

(defn pre-train-ops 
  "Returns a list of `Operation`s to run before the train_op.

    When a `ConstrainedOptimizer` creates a train_op (in `minimize`
    `minimize_unconstrained`, or `minimize_constrained`), it will include these
    ops before the main training step.

    Returns:
      A list of `Operation`s.
    "
  [ self ]
    (py/call-attr self "pre_train_ops"))

(defn proxy-constraints 
  "Returns the optional vector of proxy constraint functions.

    The difference between `constraints` and `proxy_constraints` is that, when
    proxy constraints are present, the `constraints` are merely EVALUATED during
    optimization, whereas the `proxy_constraints` are DIFFERENTIATED. If there
    are no proxy constraints, then the `constraints` are both evaluated and
    differentiated.

    For example, if we want to impose constraints on step functions, then we
    could use these functions for `constraints`. However, because a step
    function has zero gradient almost everywhere, we can't differentiate these
    functions, so we would take `proxy_constraints` to be some differentiable
    approximation of `constraints`.

    Returns:
      A tensor of proxy constraint functions.
    "
  [ self ]
    (py/call-attr self "proxy_constraints"))
