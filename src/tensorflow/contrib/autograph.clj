(ns tensorflow-core.contrib.autograph
  "This is the legacy module for AutoGraph, kept for backward compatibility.

New users should instead use `tensorflow.python.autograph`.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce autograph (import-module "tensorflow_core.contrib.autograph"))

(defn convert 
  "Decorator that compiles a function to use TensorFlow ops.

  The decorator is dynamic - it recompiles the target whenever the decorated
  function is called. This means the parameter values are known at conversion.
  It also means that repeated calls with different types of parameters will be
  correctly processed.

  Args:
    recursive: bool, whether to recursively convert any functions or classes
      that the converted function may use.
    optional_features: converted.Feature, allows toggling optional or
      experimental features. When set to None, only the core features are
      enabled.
    user_requested: bool, whether to ignore the conversion whitelist. See
      ConversionOptions.user_requested.

  Returns:
    Callable, a decorator that converts the given function into an equivalent
    function that uses TensorFlow ops.
  "
  [ & {:keys [recursive optional_features user_requested]
       :or {optional_features None}} ]
  
   (py/call-attr-kw autograph "convert" [] {:recursive recursive :optional_features optional_features :user_requested user_requested }))

(defn converted-call 
  "Compiles a function call inline.

  For internal use only.

  Args:
    f: The function to convert.
    options: converter.ConversionOptions
    args: Tuple, the original positional arguments of f
    kwargs: Dict, the original keyword arguments of f
    caller_fn_scope: Optional[function_wrappers.FunctionScope], the function
      scope of the converted function in which this call was originally made.

  Returns:
    Any, the result of executing a possibly-converted `f` with the given
      arguments.
  "
  [ f options args kwargs caller_fn_scope ]
  (py/call-attr autograph "converted_call"  f options args kwargs caller_fn_scope ))

(defn do-not-convert 
  "Decorator that suppresses the conversion of a function.

  Args:
    func: function to decorate.

  Returns:
    If `func` is not None, returns a `Callable` which is equivalent to
    `func`, but is not converted by AutoGraph.
    If `func` is None, returns a decorator that, when invoked with a
    single `func` argument, returns a `Callable` equivalent to the
    above case.
  "
  [ func ]
  (py/call-attr autograph "do_not_convert"  func ))
(defn set-element-type 
  "Indicates that the entity is expected hold items of specified type/shape.

  The staged TensorFlow ops will reflect and assert this data type. Ignored
  otherwise.

  Args:
    entity: The entity to annotate.
    dtype: TensorFlow dtype value to assert for entity.
    shape: Optional shape to assert for entity.
  "
  [entity dtype  & {:keys [shape]} ]
    (py/call-attr-kw autograph "set_element_type" [entity dtype] {:shape shape }))

(defn set-loop-options 
  "Specifies additional arguments to be passed to the enclosing while_loop.

  The parameters apply to and only to the immediately enclosing loop. It only
  has effect if the loop is staged as a TF while_loop; otherwise the parameters
  have no effect.

  Usage example:

    @tf.function(autograph=True)
    def dynamic_rnn(..., parallel_iterations=32):
      num_steps = ...
      for t in tf.range(num_steps):
        tf.autograph.experimental.set_loop_options(
            parallel_iterations=parallel_iterations)
        ...

  Args:
    parallel_iterations: See tf.while_loop.
    back_prop: See tf.while_loop.
    swap_memory: See tf.while_loop.
    maximum_iterations: See tf.while_loop.
  "
  [ & {:keys [parallel_iterations back_prop swap_memory maximum_iterations]} ]
   (py/call-attr-kw autograph "set_loop_options" [] {:parallel_iterations parallel_iterations :back_prop back_prop :swap_memory swap_memory :maximum_iterations maximum_iterations }))
(defn stack 
  "Stacks the input, if it admits the notion of stacking.

  For example, a list of tensors can be stacked into a larger tensor. This
  function is similar to tf.stack, but it accepts non-lists and lists of
  non-tensors as arguments. In the latter case, the function does nothing.

  Args:
    list_or_tensor: Any
    element_dtype: tf.DType, optional dtypedtype for the elements in the list.
        Required if the input is stackable, and the list is untyped.
    strict: bool, if True an error is raised if the input is not stackable.
        Otherwise the function is a no-op.

  Returns:
    Any, if the input is stackable, the result will be a tf.Tensor. Otherwise,
    if strict=False, the result will be list_or_tensor.

  Raises:
    ValueError: if strict=True and the input is not stackable.
  "
  [list_or_tensor element_dtype  & {:keys [strict]} ]
    (py/call-attr-kw autograph "stack" [list_or_tensor element_dtype] {:strict strict }))

(defn to-code 
  "Similar to `to_graph`, but returns Python source code as a string.

  Also see: `tf.autograph.to_graph`.

  `to_graph` returns the Python source code that can be used to generate a
  TensorFlow graph that is functionally identical to the input Python code.

  Args:
    entity: Python callable or class to convert.
    recursive: Whether to recursively convert any functions that the converted
      function may call.
    experimental_optional_features: `None`, a tuple of, or a single
      `tf.autograph.experimental.Feature` value. Controls the use of optional
      features in the conversion process.

  Returns:
    The converted code as string.
  "
  [entity & {:keys [recursive experimental_optional_features]
                       :or {experimental_optional_features None}} ]
    (py/call-attr-kw autograph "to_code" [entity] {:recursive recursive :experimental_optional_features experimental_optional_features }))

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
    experimental_optional_features: `None`, a tuple of, or a single
      `tf.autograph.experimental.Feature` value. Controls the use of optional
      features in the conversion process.

  Returns:
    Same as `entity`, the converted Python function or class.

  Raises:
    ValueError: If the entity could not be converted.
  "
  [entity & {:keys [recursive experimental_optional_features]
                       :or {experimental_optional_features None}} ]
    (py/call-attr-kw autograph "to_graph" [entity] {:recursive recursive :experimental_optional_features experimental_optional_features }))
