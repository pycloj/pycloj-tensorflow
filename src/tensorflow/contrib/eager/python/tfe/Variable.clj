(ns tensorflow.contrib.eager.python.tfe.Variable
  "Variable based on resource handles.

  See the [Variables How To](https://tensorflow.org/guide/variables)
  for a high level overview.

  A `ResourceVariable` allows you to maintain state across subsequent calls to
  session.run.

  The `ResourceVariable` constructor requires an initial value for the variable,
  which can be a `Tensor` of any type and shape. The initial value defines the
  type and shape of the variable. After construction, the type and shape of
  the variable are fixed. The value can be changed using one of the assign
  methods.

  Just like any `Tensor`, variables created with
  `tf.Variable(use_resource=True)` can be used as inputs for other Ops in the
  graph. Additionally, all the operators overloaded for the `Tensor` class are
  carried over to variables, so you can also add nodes to the graph by just
  doing arithmetic on variables.

  Unlike ref-based variable, a ResourceVariable has well-defined semantics. Each
  usage of a ResourceVariable in a TensorFlow graph adds a read_value operation
  to the graph. The Tensors returned by a read_value operation are guaranteed to
  see all modifications to the value of the variable which happen in any
  operation on which the read_value depends on (either directly, indirectly, or
  via a control dependency) and guaranteed to not see any modification to the
  value of the variable from operations that depend on the read_value operation.
  Updates from operations that have no dependency relationship to the read_value
  operation might or might not be visible to read_value.

  For example, if there is more than one assignment to a ResourceVariable in
  a single session.run call there is a well-defined value for each operation
  which uses the variable's value if the assignments and the read are connected
  by edges in the graph. Consider the following example, in which two writes
  can cause tf.Variable and tf.ResourceVariable to behave differently:

  ```python
  a = tf.Variable(1.0, use_resource=True)
  a.initializer.run()

  assign = a.assign(2.0)
  with tf.control_dependencies([assign]):
    b = a.read_value()
  with tf.control_dependencies([b]):
    other_assign = a.assign(3.0)
  with tf.control_dependencies([other_assign]):
    # Will print 2.0 because the value was read before other_assign ran. If
    # `a` was a tf.Variable instead, 2.0 or 3.0 could be printed.
    tf.compat.v1.Print(b, [b]).eval()
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
(defonce tfe (import-module "tensorflow.contrib.eager.python.tfe"))

(defn Variable 
  "Variable based on resource handles.

  See the [Variables How To](https://tensorflow.org/guide/variables)
  for a high level overview.

  A `ResourceVariable` allows you to maintain state across subsequent calls to
  session.run.

  The `ResourceVariable` constructor requires an initial value for the variable,
  which can be a `Tensor` of any type and shape. The initial value defines the
  type and shape of the variable. After construction, the type and shape of
  the variable are fixed. The value can be changed using one of the assign
  methods.

  Just like any `Tensor`, variables created with
  `tf.Variable(use_resource=True)` can be used as inputs for other Ops in the
  graph. Additionally, all the operators overloaded for the `Tensor` class are
  carried over to variables, so you can also add nodes to the graph by just
  doing arithmetic on variables.

  Unlike ref-based variable, a ResourceVariable has well-defined semantics. Each
  usage of a ResourceVariable in a TensorFlow graph adds a read_value operation
  to the graph. The Tensors returned by a read_value operation are guaranteed to
  see all modifications to the value of the variable which happen in any
  operation on which the read_value depends on (either directly, indirectly, or
  via a control dependency) and guaranteed to not see any modification to the
  value of the variable from operations that depend on the read_value operation.
  Updates from operations that have no dependency relationship to the read_value
  operation might or might not be visible to read_value.

  For example, if there is more than one assignment to a ResourceVariable in
  a single session.run call there is a well-defined value for each operation
  which uses the variable's value if the assignments and the read are connected
  by edges in the graph. Consider the following example, in which two writes
  can cause tf.Variable and tf.ResourceVariable to behave differently:

  ```python
  a = tf.Variable(1.0, use_resource=True)
  a.initializer.run()

  assign = a.assign(2.0)
  with tf.control_dependencies([assign]):
    b = a.read_value()
  with tf.control_dependencies([b]):
    other_assign = a.assign(3.0)
  with tf.control_dependencies([other_assign]):
    # Will print 2.0 because the value was read before other_assign ran. If
    # `a` was a tf.Variable instead, 2.0 or 3.0 could be printed.
    tf.compat.v1.Print(b, [b]).eval()
  ```
  "
  [  ]
  (py/call-attr eager "Variable"  ))

(defn aggregation 
  ""
  [ self ]
    (py/call-attr self "aggregation"))
(defn assign 
  "Assigns a new value to this variable.

    Args:
      value: A `Tensor`. The new value for this variable.
      use_locking: If `True`, use locking during the assignment.
      name: The name to use for the assignment.
      read_value: A `bool`. Whether to read and return the new value of the
          variable or not.

    Returns:
      If `read_value` is `True`, this method will return the new value of the
      variable after the assignment has completed. Otherwise, when in graph mode
      it will return the `Operation` that does the assignment, and when in eager
      mode it will return `None`.
    "
  [self value use_locking name  & {:keys [read_value]} ]
    (py/call-attr-kw self "assign" [value use_locking name] {:read_value read_value }))
(defn assign-add 
  "Adds a value to this variable.

    Args:
      delta: A `Tensor`. The value to add to this variable.
      use_locking: If `True`, use locking during the operation.
      name: The name to use for the operation.
      read_value: A `bool`. Whether to read and return the new value of the
          variable or not.

    Returns:
      If `read_value` is `True`, this method will return the new value of the
      variable after the assignment has completed. Otherwise, when in graph mode
      it will return the `Operation` that does the assignment, and when in eager
      mode it will return `None`.
    "
  [self delta use_locking name  & {:keys [read_value]} ]
    (py/call-attr-kw self "assign_add" [delta use_locking name] {:read_value read_value }))
(defn assign-sub 
  "Subtracts a value from this variable.

    Args:
      delta: A `Tensor`. The value to subtract from this variable.
      use_locking: If `True`, use locking during the operation.
      name: The name to use for the operation.
      read_value: A `bool`. Whether to read and return the new value of the
          variable or not.

    Returns:
      If `read_value` is `True`, this method will return the new value of the
      variable after the assignment has completed. Otherwise, when in graph mode
      it will return the `Operation` that does the assignment, and when in eager
      mode it will return `None`.
    "
  [self delta use_locking name  & {:keys [read_value]} ]
    (py/call-attr-kw self "assign_sub" [delta use_locking name] {:read_value read_value }))

(defn batch-scatter-update 
  "Assigns `tf.IndexedSlices` to this variable batch-wise.

    Analogous to `batch_gather`. This assumes that this variable and the
    sparse_delta IndexedSlices have a series of leading dimensions that are the
    same for all of them, and the updates are performed on the last dimension of
    indices. In other words, the dimensions should be the following:

    `num_prefix_dims = sparse_delta.indices.ndims - 1`
    `batch_dim = num_prefix_dims + 1`
    `sparse_delta.updates.shape = sparse_delta.indices.shape + var.shape[
         batch_dim:]`

    where

    `sparse_delta.updates.shape[:num_prefix_dims]`
    `== sparse_delta.indices.shape[:num_prefix_dims]`
    `== var.shape[:num_prefix_dims]`

    And the operation performed can be expressed as:

    `var[i_1, ..., i_n,
         sparse_delta.indices[i_1, ..., i_n, j]] = sparse_delta.updates[
            i_1, ..., i_n, j]`

    When sparse_delta.indices is a 1D tensor, this operation is equivalent to
    `scatter_update`.

    To avoid this operation one can looping over the first `ndims` of the
    variable and using `scatter_update` on the subtensors that result of slicing
    the first dimension. This is a valid option for `ndims = 1`, but less
    efficient than this implementation.

    Args:
      sparse_delta: `tf.IndexedSlices` to be assigned to this variable.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.

    Returns:
      A `Tensor` that will hold the new value of this variable after
      the scattered subtraction has completed.

    Raises:
      TypeError: if `sparse_delta` is not an `IndexedSlices`.
    "
  [self sparse_delta & {:keys [use_locking name]
                       :or {name None}} ]
    (py/call-attr-kw self "batch_scatter_update" [sparse_delta] {:use_locking use_locking :name name }))

(defn constraint 
  "Returns the constraint function associated with this variable.

    Returns:
      The constraint function that was passed to the variable constructor.
      Can be `None` if no constraint was passed.
    "
  [ self ]
    (py/call-attr self "constraint"))

(defn count-up-to 
  "Increments this variable until it reaches `limit`. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Prefer Dataset.range instead.

When that Op is run it tries to increment the variable by `1`. If
incrementing the variable would bring it above `limit` then the Op raises
the exception `OutOfRangeError`.

If no error is raised, the Op outputs the value of the variable before
the increment.

This is essentially a shortcut for `count_up_to(self, limit)`.

Args:
  limit: value at which incrementing the variable raises an error.

Returns:
  A `Tensor` that will hold the variable value before the increment. If no
  other Op modifies this variable, the values produced will all be
  distinct."
  [ self limit ]
  (py/call-attr self "count_up_to"  self limit ))

(defn create 
  "The op responsible for initializing this variable."
  [ self ]
    (py/call-attr self "create"))

(defn device 
  "The device this variable is on."
  [ self ]
    (py/call-attr self "device"))

(defn dtype 
  "The dtype of this variable."
  [ self ]
    (py/call-attr self "dtype"))

(defn eval 
  "Evaluates and returns the value of this variable."
  [ self session ]
  (py/call-attr self "eval"  self session ))

(defn experimental-ref 
  "Returns a hashable reference object to this Variable.

    Warning: Experimental API that could be changed or removed.

    The primary usecase for this API is to put variables in a set/dictionary.
    We can't put variables in a set/dictionary as `variable.__hash__()` is no
    longer available starting Tensorflow 2.0.

    ```python
    import tensorflow as tf

    x = tf.Variable(5)
    y = tf.Variable(10)
    z = tf.Variable(10)

    # The followings will raise an exception starting 2.0
    # TypeError: Variable is unhashable if Variable equality is enabled.
    variable_set = {x, y, z}
    variable_dict = {x: 'five', y: 'ten'}
    ```

    Instead, we can use `variable.experimental_ref()`.

    ```python
    variable_set = {x.experimental_ref(),
                    y.experimental_ref(),
                    z.experimental_ref()}

    print(x.experimental_ref() in variable_set)
    ==> True

    variable_dict = {x.experimental_ref(): 'five',
                     y.experimental_ref(): 'ten',
                     z.experimental_ref(): 'ten'}

    print(variable_dict[y.experimental_ref()])
    ==> ten
    ```

    Also, the reference object provides `.deref()` function that returns the
    original Variable.

    ```python
    x = tf.Variable(5)
    print(x.experimental_ref().deref())
    ==> <tf.Variable 'Variable:0' shape=() dtype=int32, numpy=5>
    ```
    "
  [ self  ]
  (py/call-attr self "experimental_ref"  self  ))

(defn from-proto 
  ""
  [ self variable_def import_scope ]
  (py/call-attr self "from_proto"  self variable_def import_scope ))

(defn gather-nd 
  "Reads the value of this variable sparsely, using `gather_nd`."
  [ self indices name ]
  (py/call-attr self "gather_nd"  self indices name ))

(defn get-shape 
  "Alias of `Variable.shape`."
  [ self  ]
  (py/call-attr self "get_shape"  self  ))

(defn graph 
  "The `Graph` of this variable."
  [ self ]
    (py/call-attr self "graph"))

(defn handle 
  "The handle by which this variable can be accessed."
  [ self ]
    (py/call-attr self "handle"))

(defn initial-value 
  "Returns the Tensor used as the initial value for the variable."
  [ self ]
    (py/call-attr self "initial_value"))

(defn initialized-value 
  "Returns the value of the initialized variable. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use Variable.read_value. Variables in 2.X are initialized automatically both in eager and graph (inside tf.defun) contexts.

You should use this instead of the variable itself to initialize another
variable with a value that depends on the value of this variable.

```python
# Initialize 'v' with a random tensor.
v = tf.Variable(tf.random.truncated_normal([10, 40]))
# Use `initialized_value` to guarantee that `v` has been
# initialized before its value is used to initialize `w`.
# The random values are picked only once.
w = tf.Variable(v.initialized_value() * 2.0)
```

Returns:
  A `Tensor` holding the value of this variable after its initializer
  has run."
  [ self  ]
  (py/call-attr self "initialized_value"  self  ))

(defn initializer 
  "The op responsible for initializing this variable."
  [ self ]
    (py/call-attr self "initializer"))

(defn is-initialized 
  "Checks whether a resource variable has been initialized.

    Outputs boolean scalar indicating whether the tensor has been initialized.

    Args:
      name: A name for the operation (optional).

    Returns:
      A `Tensor` of type `bool`.
    "
  [ self name ]
  (py/call-attr self "is_initialized"  self name ))

(defn load 
  "Load new value into this variable. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Prefer Variable.assign which has equivalent behavior in 2.X.

Writes new value to variable's memory. Doesn't add ops to the graph.

This convenience method requires a session where the graph
containing this variable has been launched. If no session is
passed, the default session is used.  See `tf.compat.v1.Session` for more
information on launching a graph and on sessions.

```python
v = tf.Variable([1, 2])
init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)
    # Usage passing the session explicitly.
    v.load([2, 3], sess)
    print(v.eval(sess)) # prints [2 3]
    # Usage with the default session.  The 'with' block
    # above makes 'sess' the default session.
    v.load([3, 4], sess)
    print(v.eval()) # prints [3 4]
```

Args:
    value: New variable value
    session: The session to use to evaluate this variable. If none, the
      default session is used.

Raises:
    ValueError: Session is not passed and no default session"
  [ self value session ]
  (py/call-attr self "load"  self value session ))

(defn name 
  "The name of the handle for this variable."
  [ self ]
    (py/call-attr self "name"))

(defn numpy 
  ""
  [ self  ]
  (py/call-attr self "numpy"  self  ))

(defn op 
  "The op for this variable."
  [ self ]
    (py/call-attr self "op"))

(defn read-value 
  "Constructs an op which reads the value of this variable.

    Should be used when there are multiple reads, or when it is desirable to
    read the value only after some condition is true.

    Returns:
     the read operation.
    "
  [ self  ]
  (py/call-attr self "read_value"  self  ))

(defn scatter-add 
  "Adds `tf.IndexedSlices` to this variable.

    Args:
      sparse_delta: `tf.IndexedSlices` to be added to this variable.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.

    Returns:
      A `Tensor` that will hold the new value of this variable after
      the scattered addition has completed.

    Raises:
      TypeError: if `sparse_delta` is not an `IndexedSlices`.
    "
  [self sparse_delta & {:keys [use_locking name]
                       :or {name None}} ]
    (py/call-attr-kw self "scatter_add" [sparse_delta] {:use_locking use_locking :name name }))

(defn scatter-div 
  "Divide this variable by `tf.IndexedSlices`.

    Args:
      sparse_delta: `tf.IndexedSlices` to divide this variable by.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.

    Returns:
      A `Tensor` that will hold the new value of this variable after
      the scattered division has completed.

    Raises:
      TypeError: if `sparse_delta` is not an `IndexedSlices`.
    "
  [self sparse_delta & {:keys [use_locking name]
                       :or {name None}} ]
    (py/call-attr-kw self "scatter_div" [sparse_delta] {:use_locking use_locking :name name }))

(defn scatter-max 
  "Updates this variable with the max of `tf.IndexedSlices` and itself.

    Args:
      sparse_delta: `tf.IndexedSlices` to use as an argument of max
        with this variable.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.

    Returns:
      A `Tensor` that will hold the new value of this variable after
      the scattered maximization has completed.

    Raises:
      TypeError: if `sparse_delta` is not an `IndexedSlices`.
    "
  [self sparse_delta & {:keys [use_locking name]
                       :or {name None}} ]
    (py/call-attr-kw self "scatter_max" [sparse_delta] {:use_locking use_locking :name name }))

(defn scatter-min 
  "Updates this variable with the min of `tf.IndexedSlices` and itself.

    Args:
      sparse_delta: `tf.IndexedSlices` to use as an argument of min
        with this variable.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.

    Returns:
      A `Tensor` that will hold the new value of this variable after
      the scattered minimization has completed.

    Raises:
      TypeError: if `sparse_delta` is not an `IndexedSlices`.
    "
  [self sparse_delta & {:keys [use_locking name]
                       :or {name None}} ]
    (py/call-attr-kw self "scatter_min" [sparse_delta] {:use_locking use_locking :name name }))

(defn scatter-mul 
  "Multiply this variable by `tf.IndexedSlices`.

    Args:
      sparse_delta: `tf.IndexedSlices` to multiply this variable by.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.

    Returns:
      A `Tensor` that will hold the new value of this variable after
      the scattered multiplication has completed.

    Raises:
      TypeError: if `sparse_delta` is not an `IndexedSlices`.
    "
  [self sparse_delta & {:keys [use_locking name]
                       :or {name None}} ]
    (py/call-attr-kw self "scatter_mul" [sparse_delta] {:use_locking use_locking :name name }))

(defn scatter-nd-add 
  "Applies sparse addition to individual values or slices in a Variable.

    `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.

    `indices` must be integer tensor, containing indices into `ref`.
    It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.

    The innermost dimension of `indices` (with length `K`) corresponds to
    indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
    dimension of `ref`.

    `updates` is `Tensor` of rank `Q-1+P-K` with shape:

    ```
    [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
    ```

    For example, say we want to add 4 scattered elements to a rank-1 tensor to
    8 elements. In Python, that update would look like this:

    ```python
        ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
        indices = tf.constant([[4], [3], [1] ,[7]])
        updates = tf.constant([9, 10, 11, 12])
        add = ref.scatter_nd_add(indices, updates)
        with tf.compat.v1.Session() as sess:
          print sess.run(add)
    ```

    The resulting update to ref would look like this:

        [1, 13, 3, 14, 14, 6, 7, 20]

    See `tf.scatter_nd` for more details about how to make updates to
    slices.

    Args:
      indices: The indices to be used in the operation.
      updates: The values to be used in the operation.
      name: the name of the operation.

    Returns:
      A `Tensor` that will hold the new value of this variable after
      the scattered subtraction has completed.
    "
  [ self indices updates name ]
  (py/call-attr self "scatter_nd_add"  self indices updates name ))

(defn scatter-nd-sub 
  "Applies sparse subtraction to individual values or slices in a Variable.

    `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.

    `indices` must be integer tensor, containing indices into `ref`.
    It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.

    The innermost dimension of `indices` (with length `K`) corresponds to
    indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
    dimension of `ref`.

    `updates` is `Tensor` of rank `Q-1+P-K` with shape:

    ```
    [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
    ```

    For example, say we want to add 4 scattered elements to a rank-1 tensor to
    8 elements. In Python, that update would look like this:

    ```python
        ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
        indices = tf.constant([[4], [3], [1] ,[7]])
        updates = tf.constant([9, 10, 11, 12])
        op = ref.scatter_nd_sub(indices, updates)
        with tf.compat.v1.Session() as sess:
          print sess.run(op)
    ```

    The resulting update to ref would look like this:

        [1, -9, 3, -6, -6, 6, 7, -4]

    See `tf.scatter_nd` for more details about how to make updates to
    slices.

    Args:
      indices: The indices to be used in the operation.
      updates: The values to be used in the operation.
      name: the name of the operation.

    Returns:
      A `Tensor` that will hold the new value of this variable after
      the scattered subtraction has completed.
    "
  [ self indices updates name ]
  (py/call-attr self "scatter_nd_sub"  self indices updates name ))

(defn scatter-nd-update 
  "Applies sparse assignment to individual values or slices in a Variable.

    `ref` is a `Tensor` with rank `P` and `indices` is a `Tensor` of rank `Q`.

    `indices` must be integer tensor, containing indices into `ref`.
    It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.

    The innermost dimension of `indices` (with length `K`) corresponds to
    indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
    dimension of `ref`.

    `updates` is `Tensor` of rank `Q-1+P-K` with shape:

    ```
    [d_0, ..., d_{Q-2}, ref.shape[K], ..., ref.shape[P-1]].
    ```

    For example, say we want to add 4 scattered elements to a rank-1 tensor to
    8 elements. In Python, that update would look like this:

    ```python
        ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
        indices = tf.constant([[4], [3], [1] ,[7]])
        updates = tf.constant([9, 10, 11, 12])
        op = ref.scatter_nd_update(indices, updates)
        with tf.compat.v1.Session() as sess:
          print sess.run(op)
    ```

    The resulting update to ref would look like this:

        [1, 11, 3, 10, 9, 6, 7, 12]

    See `tf.scatter_nd` for more details about how to make updates to
    slices.

    Args:
      indices: The indices to be used in the operation.
      updates: The values to be used in the operation.
      name: the name of the operation.

    Returns:
      A `Tensor` that will hold the new value of this variable after
      the scattered subtraction has completed.
    "
  [ self indices updates name ]
  (py/call-attr self "scatter_nd_update"  self indices updates name ))

(defn scatter-sub 
  "Subtracts `tf.IndexedSlices` from this variable.

    Args:
      sparse_delta: `tf.IndexedSlices` to be subtracted from this variable.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.

    Returns:
      A `Tensor` that will hold the new value of this variable after
      the scattered subtraction has completed.

    Raises:
      TypeError: if `sparse_delta` is not an `IndexedSlices`.
    "
  [self sparse_delta & {:keys [use_locking name]
                       :or {name None}} ]
    (py/call-attr-kw self "scatter_sub" [sparse_delta] {:use_locking use_locking :name name }))

(defn scatter-update 
  "Assigns `tf.IndexedSlices` to this variable.

    Args:
      sparse_delta: `tf.IndexedSlices` to be assigned to this variable.
      use_locking: If `True`, use locking during the operation.
      name: the name of the operation.

    Returns:
      A `Tensor` that will hold the new value of this variable after
      the scattered subtraction has completed.

    Raises:
      TypeError: if `sparse_delta` is not an `IndexedSlices`.
    "
  [self sparse_delta & {:keys [use_locking name]
                       :or {name None}} ]
    (py/call-attr-kw self "scatter_update" [sparse_delta] {:use_locking use_locking :name name }))

(defn set-shape 
  "Unsupported."
  [ self shape ]
  (py/call-attr self "set_shape"  self shape ))

(defn shape 
  "The shape of this variable."
  [ self ]
    (py/call-attr self "shape"))

(defn sparse-read 
  "Reads the value of this variable sparsely, using `gather`."
  [ self indices name ]
  (py/call-attr self "sparse_read"  self indices name ))

(defn synchronization 
  ""
  [ self ]
    (py/call-attr self "synchronization"))

(defn to-proto 
  "Converts a `ResourceVariable` to a `VariableDef` protocol buffer.

    Args:
      export_scope: Optional `string`. Name scope to remove.

    Raises:
      RuntimeError: If run in EAGER mode.

    Returns:
      A `VariableDef` protocol buffer, or `None` if the `Variable` is not
      in the specified name scope.
    "
  [ self export_scope ]
  (py/call-attr self "to_proto"  self export_scope ))

(defn trainable 
  ""
  [ self ]
    (py/call-attr self "trainable"))

(defn value 
  "A cached operation which reads the value of this variable."
  [ self  ]
  (py/call-attr self "value"  self  ))
