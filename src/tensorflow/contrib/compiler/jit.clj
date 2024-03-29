(ns tensorflow.contrib.compiler.jit
  "Library for controlling the Tensorflow/XLA JIT compiler."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce jit (import-module "tensorflow.contrib.compiler.jit"))

(defn experimental-jit-scope 
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
   (py/call-attr-kw jit "experimental_jit_scope" [] {:compile_ops compile_ops :separate_compiled_gradients separate_compiled_gradients }))
