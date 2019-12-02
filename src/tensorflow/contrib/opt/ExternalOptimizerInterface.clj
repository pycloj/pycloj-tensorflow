(ns tensorflow.contrib.opt.ExternalOptimizerInterface
  "Base class for interfaces with external optimization algorithms.

  Subclass this and implement `_minimize` in order to wrap a new optimization
  algorithm.

  `ExternalOptimizerInterface` should not be instantiated directly; instead use
  e.g. `ScipyOptimizerInterface`.

  @@__init__

  @@minimize
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

(defn ExternalOptimizerInterface 
  "Base class for interfaces with external optimization algorithms.

  Subclass this and implement `_minimize` in order to wrap a new optimization
  algorithm.

  `ExternalOptimizerInterface` should not be instantiated directly; instead use
  e.g. `ScipyOptimizerInterface`.

  @@__init__

  @@minimize
  "
  [ loss var_list equalities inequalities var_to_bounds ]
  (py/call-attr opt "ExternalOptimizerInterface"  loss var_list equalities inequalities var_to_bounds ))

(defn minimize 
  "Minimize a scalar `Tensor`.

    Variables subject to optimization are updated in-place at the end of
    optimization.

    Note that this method does *not* just return a minimization `Op`, unlike
    `Optimizer.minimize()`; instead it actually performs minimization by
    executing commands to control a `Session`.

    Args:
      session: A `Session` instance.
      feed_dict: A feed dict to be passed to calls to `session.run`.
      fetches: A list of `Tensor`s to fetch and supply to `loss_callback`
        as positional arguments.
      step_callback: A function to be called at each optimization step;
        arguments are the current values of all optimization variables
        flattened into a single vector.
      loss_callback: A function to be called every time the loss and gradients
        are computed, with evaluated fetches supplied as positional arguments.
      **run_kwargs: kwargs to pass to `session.run`.
    "
  [ self session feed_dict fetches step_callback loss_callback ]
  (py/call-attr self "minimize"  self session feed_dict fetches step_callback loss_callback ))
