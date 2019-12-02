(ns tensorflow.-api.v1.compat.v2.xla.experimental
  "Public API for tf.xla.experimental namespace.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce experimental (import-module "tensorflow._api.v1.compat.v2.xla.experimental"))

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
  (py/call-attr experimental "compile"  computation inputs ))

(defn jit-scope 
  "Enable or disable JIT compilation of operators within the scope.

  NOTE: This is an experimental feature.

  The compilation is a hint and only supported on a best-effort basis.

  Example usage:
    with tf.xla.experimental.jit_scope():
      c = tf.matmul(a, b)  # compiled
    with tf.xla.experimental.jit_scope(compile_ops=False):
      d = tf.matmul(a, c)  # not compiled
    with tf.xla.experimental.jit_scope(
        compile_ops=lambda node_def: 'matmul' in node_def.op.lower()):
      e = tf.matmul(a, b) + d  # matmul is compiled, the addition is not.

  Example of separate_compiled_gradients:
    # In the example below, the computations for f, g and h will all be compiled
    # in separate scopes.
    with tf.xla.experimental.jit_scope(
        separate_compiled_gradients=True):
      f = tf.matmul(a, b)
    g = tf.gradients([f], [a, b], name='mygrads1')
    h = tf.gradients([f], [a, b], name='mygrads2')

  Args:
    compile_ops: Whether to enable or disable compilation in the scope.
      Either a Python bool, or a callable that accepts the parameter
      `node_def` and returns a python bool.
    separate_compiled_gradients: If true put each gradient subgraph into a
      separate compilation scope. This gives fine-grained control over which
      portions of the graph will be compiled as a single unit. Compiling
      gradients separately may yield better performance for some graphs.
      The scope is named based on the scope of the forward computation as well
      as the name of the gradients. As a result, the gradients will be compiled
      in a scope that is separate from both the forward computation, and from
      other gradients.
  Raises:
    RuntimeError: if called when eager execution is enabled.
  Yields:
    The current scope, enabling or disabling compilation.
  "
  [ & {:keys [compile_ops separate_compiled_gradients]} ]
   (py/call-attr-kw experimental "jit_scope" [] {:compile_ops compile_ops :separate_compiled_gradients separate_compiled_gradients }))
