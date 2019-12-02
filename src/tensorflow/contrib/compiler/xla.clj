(ns tensorflow.contrib.compiler.xla
  "xla is an experimental library that provides XLA support APIs."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce xla (import-module "tensorflow.contrib.compiler.xla"))

(defn check-function-argument-count 
  "Validate the number of input arguments to an XLA function.

  Args:
    func: the Python function that will be called to generate the body of an XLA
      computation graph.
    input_arity: the number of explicit arguments supplied by the caller.
    infeed_queue: if not None, the infeed queue that will supply
      additional arguments to the function.

  Returns:
    None if function can be called with the supplied number of
      arguments, or an error string if it cannot.
  "
  [ func input_arity infeed_queue ]
  (py/call-attr xla "check_function_argument_count"  func input_arity infeed_queue ))

(defn compile 
  "Builds an operator that compiles and runs `computation` with XLA.

  NOTE: In eager mode, `computation` will have `@tf.function` semantics.

  Args:
    computation: A Python function that builds a computation to apply to the
      input. If the function takes n inputs, 'inputs' should be a list of n
      tensors.

      `computation` may return a list of operations and tensors.  Tensors must
      come before operations in the returned list.  The return value of
      `compile` is a list of tensors corresponding to the tensors from the
      output of `computation`.

      All `Operation`s returned from `computation` will be executed when
      evaluating any of the returned output tensors.
    inputs: A list of inputs or `None` (equivalent to an empty list). Each input
      can be a nested structure containing values that are convertible to
      tensors. Note that passing an N-dimension list of compatible values will
      result in a N-dimension list of scalar tensors rather than a single Rank-N
      tensors. If you need different behavior, convert part of inputs to tensors
      with `tf.convert_to_tensor`.

  Returns:
    Same data structure as if computation(*inputs) is called directly with some
    exceptions for correctness. Exceptions include:
      1) None output: a NoOp would be returned which control-depends on
         computation.
      2) Single value output: A tuple containing the value would be returned.
      3) Operation-only outputs: a NoOp would be returned which
         control-depends on computation.
      TODO(b/121383831): Investigate into removing these special cases.

  Raises:
    RuntimeError: if called when eager execution is enabled.
  "
  [ computation inputs ]
  (py/call-attr xla "compile"  computation inputs ))

(defn estimator-model-fn 
  "estimator_model_fn decorates a model_fn to be compiled for execution.

  Currently it only works with `TPUEstimator`. If you need to use it with base
  `Estimator`, please add `tf.compat.v1.enable_resource_variables()` at the
  beginning of your program.

  Example 1, decorating model_fn:
  ```
  @xla.estimator_model_fn()
  def model_fn(features, labels, mode, params):
    ...
    return EstimatorSpec(...)


  est = Estimator(model_fn=model_fn, ...)
  est.train(...)

  ```

  Example 2, decorator as function:
  ```
  def model_fn(features, labels, mode, params):
    ...
    return EstimatorSpec(...)

  est = Estimator(model_fn=xla.estimator_model_fn(model_fn), ...)
  est.train(...)
  ```

  Args:
    target_model_fn: model_fn to be decorated. This is only needed when
      decorator is used in function call form (example 2).

  Returns:
    Decorated target_model_fn.
  "
  [ target_model_fn ]
  (py/call-attr xla "estimator_model_fn"  target_model_fn ))
