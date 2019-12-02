(ns tensorflow.autograph
  "Conversion of plain Python into TensorFlow graph code.

NOTE: In TensorFlow 2.0, AutoGraph is automatically applied when using
`tf.function`. This module contains lower-level APIs for advanced use.

For more information, see the
[AutoGraph guide](https://www.tensorflow.org/guide/autograph).

By equivalent graph code we mean code that generates a TensorFlow graph when
run. The generated graph has the same effects as the original code when executed
(for example with `tf.function` or `tf.compat.v1.Session.run`). In other words,
using AutoGraph can be thought of as running Python in TensorFlow.

"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce autograph (import-module "tensorflow.autograph"))
(defn set-verbosity 
  "Sets the AutoGraph verbosity level.

  _Debug logging in AutoGraph_

  More verbose logging is useful to enable when filing bug reports or doing
  more in-depth debugging.

  There are two means to control the logging verbosity:

   * The `set_verbosity` function

   * The `AUTOGRAPH_VERBOSITY` environment variable

  `set_verbosity` takes precedence over the environment variable.

  For example:

  ```python
  import os
  import tensorflow as tf

  os.environ['AUTOGRAPH_VERBOSITY'] = 5
  # Verbosity is now 5

  tf.autograph.set_verbosity(0)
  # Verbosity is now 0

  os.environ['AUTOGRAPH_VERBOSITY'] = 1
  # No effect, because set_verbosity was already called.
  ```

  Logs entries are output to [absl](https://abseil.io)'s 
  [default output](https://abseil.io/docs/python/guides/logging),
  with `INFO` level.
  Logs can be mirrored to stdout by using the `alsologtostdout` argument.
  Mirroring is enabled by default when Python runs in interactive mode.

  Args:
    level: int, the verbosity level; larger values specify increased verbosity;
      0 means no logging. When reporting bugs, it is recommended to set this
      value to a larger number, like 10.
    alsologtostdout: bool, whether to also output log messages to `sys.stdout`.
  "
  [level  & {:keys [alsologtostdout]} ]
    (py/call-attr-kw autograph "set_verbosity" [level] {:alsologtostdout alsologtostdout }))

(defn to-code 
  "Similar to `to_graph`, but returns Python source code as a string.

  Also see: `tf.autograph.to_graph`.

  `to_graph` returns the Python source code that can be used to generate a
  TensorFlow graph that is functionally identical to the input Python code.

  Args:
    entity: Python callable or class to convert.
    recursive: Whether to recursively convert any functions that the converted
      function may call.
    arg_values: Deprecated.
    arg_types: Deprecated.
    indentation: Deprecated.
    experimental_optional_features: `None`, a tuple of, or a single
      `tf.autograph.experimental.Feature` value. Controls the use of optional
      features in the conversion process.

  Returns:
    The converted code as string.
  "
  [entity & {:keys [recursive arg_values arg_types indentation experimental_optional_features]
                       :or {arg_values None arg_types None experimental_optional_features None}} ]
    (py/call-attr-kw autograph "to_code" [entity] {:recursive recursive :arg_values arg_values :arg_types arg_types :indentation indentation :experimental_optional_features experimental_optional_features }))

(defn to-graph 
  "Converts a Python entity into a TensorFlow graph.

  Also see: `tf.autograph.to_code`, `tf.function`.

  Unlike `tf.function`, `to_graph` is a low-level transpiler that converts
  Python code to TensorFlow graph code. It does not implement any caching,
  variable management or create any actual ops, and is best used where greater
  control over the generated TensorFlow graph is desired. Another difference
  from `tf.function` is that `to_graph` will not wrap the graph into a
  TensorFlow function or a Python callable. Internally, `tf.function` uses
  `to_graph`.

  _Example Usage_

  ```python
    def foo(x):
      if x > 0:
        y = x * x
      else:
        y = -x
      return y

    converted_foo = to_graph(foo)

    x = tf.constant(1)
    y = converted_foo(x)  # converted_foo is a TensorFlow Op-like.
    assert is_tensor(y)
  ```

  Supported Python entities include:
    * functions
    * classes
    * object methods

  Functions are converted into new functions with converted code.

  Classes are converted by generating a new class whose methods use converted
  code.

  Methods are converted into unbound function that have an additional first
  argument called `self`.

  Args:
    entity: Python callable or class to convert.
    recursive: Whether to recursively convert any functions that the converted
      function may call.
    arg_values: Deprecated.
    arg_types: Deprecated.
    experimental_optional_features: `None`, a tuple of, or a single
      `tf.autograph.experimental.Feature` value. Controls the use of optional
      features in the conversion process.

  Returns:
    Same as `entity`, the converted Python function or class.

  Raises:
    ValueError: If the entity could not be converted.
  "
  [entity & {:keys [recursive arg_values arg_types experimental_optional_features]
                       :or {arg_values None arg_types None experimental_optional_features None}} ]
    (py/call-attr-kw autograph "to_graph" [entity] {:recursive recursive :arg_values arg_values :arg_types arg_types :experimental_optional_features experimental_optional_features }))

(defn trace 
  "Traces argument information at compilation time.

  `trace` is useful when debugging, and it always executes during the tracing
  phase, that is, when the TF graph is constructed.

  _Example usage_

  ```python
  import tensorflow as tf

  for i in tf.range(10):
    tf.autograph.trace(i)
  # Output: <Tensor ...>
  ```

  Args:
    *args: Arguments to print to `sys.stdout`.
  "
  [  ]
  (py/call-attr autograph "trace"  ))
