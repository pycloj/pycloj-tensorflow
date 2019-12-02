(ns tensorflow.-api.v1.compat.v1.Variable
  "See the [Variables Guide](https://tensorflow.org/guide/variables).

  A variable maintains state in the graph across calls to `run()`. You add a
  variable to the graph by constructing an instance of the class `Variable`.

  The `Variable()` constructor requires an initial value for the variable,
  which can be a `Tensor` of any type and shape. The initial value defines the
  type and shape of the variable. After construction, the type and shape of
  the variable are fixed. The value can be changed using one of the assign
  methods.

  If you want to change the shape of a variable later you have to use an
  `assign` Op with `validate_shape=False`.

  Just like any `Tensor`, variables created with `Variable()` can be used as
  inputs for other Ops in the graph. Additionally, all the operators
  overloaded for the `Tensor` class are carried over to variables, so you can
  also add nodes to the graph by just doing arithmetic on variables.

  ```python
  import tensorflow as tf

  # Create a variable.
  w = tf.Variable(<initial-value>, name=<optional-name>)

  # Use the variable in the graph like any Tensor.
  y = tf.matmul(w, ...another variable or tensor...)

  # The overloaded operators are available too.
  z = tf.sigmoid(w + y)

  # Assign a new value to the variable with `assign()` or a related method.
  w.assign(w + 1.0)
  w.assign_add(1.0)
  ```

  When you launch the graph, variables have to be explicitly initialized before
  you can run Ops that use their value. You can initialize a variable by
  running its *initializer op*, restoring the variable from a save file, or
  simply running an `assign` Op that assigns a value to the variable. In fact,
  the variable *initializer op* is just an `assign` Op that assigns the
  variable's initial value to the variable itself.

  ```python
  # Launch the graph in a session.
  with tf.compat.v1.Session() as sess:
      # Run the variable initializer.
      sess.run(w.initializer)
      # ...you now can run ops that use the value of 'w'...
  ```

  The most common initialization pattern is to use the convenience function
  `global_variables_initializer()` to add an Op to the graph that initializes
  all the variables. You then run that Op after launching the graph.

  ```python
  # Add an Op to initialize global variables.
  init_op = tf.compat.v1.global_variables_initializer()

  # Launch the graph in a session.
  with tf.compat.v1.Session() as sess:
      # Run the Op that initializes global variables.
      sess.run(init_op)
      # ...you can now run any Op that uses variable values...
  ```

  If you need to create a variable with an initial value dependent on another
  variable, use the other variable's `initialized_value()`. This ensures that
  variables are initialized in the right order.

  All variables are automatically collected in the graph where they are
  created. By default, the constructor adds the new variable to the graph
  collection `GraphKeys.GLOBAL_VARIABLES`. The convenience function
  `global_variables()` returns the contents of that collection.

  When building a machine learning model it is often convenient to distinguish
  between variables holding the trainable model parameters and other variables
  such as a `global step` variable used to count training steps. To make this
  easier, the variable constructor supports a `trainable=<bool>` parameter. If
  `True`, the new variable is also added to the graph collection
  `GraphKeys.TRAINABLE_VARIABLES`. The convenience function
  `trainable_variables()` returns the contents of this collection. The
  various `Optimizer` classes use this collection as the default list of
  variables to optimize.

  WARNING: tf.Variable objects by default have a non-intuitive memory model. A
  Variable is represented internally as a mutable Tensor which can
  non-deterministically alias other Tensors in a graph. The set of operations
  which consume a Variable and can lead to aliasing is undetermined and can
  change across TensorFlow versions. Avoid writing code which relies on the
  value of a Variable either changing or not changing as other operations
  happen. For example, using Variable objects or simple functions thereof as
  predicates in a `tf.cond` is dangerous and error-prone:

  ```
  v = tf.Variable(True)
  tf.cond(v, lambda: v.assign(False), my_false_fn)  # Note: this is broken.
  ```

  Here, adding `use_resource=True` when constructing the variable will
  fix any nondeterminism issues:
  ```
  v = tf.Variable(True, use_resource=True)
  tf.cond(v, lambda: v.assign(False), my_false_fn)
  ```

  To use the replacement for variables which does
  not have these issues:

  * Add `use_resource=True` when constructing `tf.Variable`;
  * Call `tf.compat.v1.get_variable_scope().set_use_resource(True)` inside a
    `tf.compat.v1.variable_scope` before the `tf.compat.v1.get_variable()` call.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce v1 (import-module "tensorflow._api.v1.compat.v1"))

(defn Variable 
  "See the [Variables Guide](https://tensorflow.org/guide/variables).

  A variable maintains state in the graph across calls to `run()`. You add a
  variable to the graph by constructing an instance of the class `Variable`.

  The `Variable()` constructor requires an initial value for the variable,
  which can be a `Tensor` of any type and shape. The initial value defines the
  type and shape of the variable. After construction, the type and shape of
  the variable are fixed. The value can be changed using one of the assign
  methods.

  If you want to change the shape of a variable later you have to use an
  `assign` Op with `validate_shape=False`.

  Just like any `Tensor`, variables created with `Variable()` can be used as
  inputs for other Ops in the graph. Additionally, all the operators
  overloaded for the `Tensor` class are carried over to variables, so you can
  also add nodes to the graph by just doing arithmetic on variables.

  ```python
  import tensorflow as tf

  # Create a variable.
  w = tf.Variable(<initial-value>, name=<optional-name>)

  # Use the variable in the graph like any Tensor.
  y = tf.matmul(w, ...another variable or tensor...)

  # The overloaded operators are available too.
  z = tf.sigmoid(w + y)

  # Assign a new value to the variable with `assign()` or a related method.
  w.assign(w + 1.0)
  w.assign_add(1.0)
  ```

  When you launch the graph, variables have to be explicitly initialized before
  you can run Ops that use their value. You can initialize a variable by
  running its *initializer op*, restoring the variable from a save file, or
  simply running an `assign` Op that assigns a value to the variable. In fact,
  the variable *initializer op* is just an `assign` Op that assigns the
  variable's initial value to the variable itself.

  ```python
  # Launch the graph in a session.
  with tf.compat.v1.Session() as sess:
      # Run the variable initializer.
      sess.run(w.initializer)
      # ...you now can run ops that use the value of 'w'...
  ```

  The most common initialization pattern is to use the convenience function
  `global_variables_initializer()` to add an Op to the graph that initializes
  all the variables. You then run that Op after launching the graph.

  ```python
  # Add an Op to initialize global variables.
  init_op = tf.compat.v1.global_variables_initializer()

  # Launch the graph in a session.
  with tf.compat.v1.Session() as sess:
      # Run the Op that initializes global variables.
      sess.run(init_op)
      # ...you can now run any Op that uses variable values...
  ```

  If you need to create a variable with an initial value dependent on another
  variable, use the other variable's `initialized_value()`. This ensures that
  variables are initialized in the right order.

  All variables are automatically collected in the graph where they are
  created. By default, the constructor adds the new variable to the graph
  collection `GraphKeys.GLOBAL_VARIABLES`. The convenience function
  `global_variables()` returns the contents of that collection.

  When building a machine learning model it is often convenient to distinguish
  between variables holding the trainable model parameters and other variables
  such as a `global step` variable used to count training steps. To make this
  easier, the variable constructor supports a `trainable=<bool>` parameter. If
  `True`, the new variable is also added to the graph collection
  `GraphKeys.TRAINABLE_VARIABLES`. The convenience function
  `trainable_variables()` returns the contents of this collection. The
  various `Optimizer` classes use this collection as the default list of
  variables to optimize.

  WARNING: tf.Variable objects by default have a non-intuitive memory model. A
  Variable is represented internally as a mutable Tensor which can
  non-deterministically alias other Tensors in a graph. The set of operations
  which consume a Variable and can lead to aliasing is undetermined and can
  change across TensorFlow versions. Avoid writing code which relies on the
  value of a Variable either changing or not changing as other operations
  happen. For example, using Variable objects or simple functions thereof as
  predicates in a `tf.cond` is dangerous and error-prone:

  ```
  v = tf.Variable(True)
  tf.cond(v, lambda: v.assign(False), my_false_fn)  # Note: this is broken.
  ```

  Here, adding `use_resource=True` when constructing the variable will
  fix any nondeterminism issues:
  ```
  v = tf.Variable(True, use_resource=True)
  tf.cond(v, lambda: v.assign(False), my_false_fn)
  ```

  To use the replacement for variables which does
  not have these issues:

  * Add `use_resource=True` when constructing `tf.Variable`;
  * Call `tf.compat.v1.get_variable_scope().set_use_resource(True)` inside a
    `tf.compat.v1.variable_scope` before the `tf.compat.v1.get_variable()` call.
  "
  [  ]
  (py/call-attr v1 "Variable"  ))

(defn aggregation 
  ""
  [ self ]
    (py/call-attr self "aggregation"))

(defn assign 
  "Assigns a new value to the variable.

    This is essentially a shortcut for `assign(self, value)`.

    Args:
      value: A `Tensor`. The new value for this variable.
      use_locking: If `True`, use locking during the assignment.
      name: The name of the operation to be created
      read_value: if True, will return something which evaluates to the new
        value of the variable; if False will return the assign op.

    Returns:
      A `Tensor` that will hold the new value of this variable after
      the assignment has completed.
    "
  [self value & {:keys [use_locking name read_value]
                       :or {name None}} ]
    (py/call-attr-kw self "assign" [value] {:use_locking use_locking :name name :read_value read_value }))

(defn assign-add 
  "Adds a value to this variable.

     This is essentially a shortcut for `assign_add(self, delta)`.

    Args:
      delta: A `Tensor`. The value to add to this variable.
      use_locking: If `True`, use locking during the operation.
      name: The name of the operation to be created
      read_value: if True, will return something which evaluates to the new
        value of the variable; if False will return the assign op.

    Returns:
      A `Tensor` that will hold the new value of this variable after
      the addition has completed.
    "
  [self delta & {:keys [use_locking name read_value]
                       :or {name None}} ]
    (py/call-attr-kw self "assign_add" [delta] {:use_locking use_locking :name name :read_value read_value }))

(defn assign-sub 
  "Subtracts a value from this variable.

    This is essentially a shortcut for `assign_sub(self, delta)`.

    Args:
      delta: A `Tensor`. The value to subtract from this variable.
      use_locking: If `True`, use locking during the operation.
      name: The name of the operation to be created
      read_value: if True, will return something which evaluates to the new
        value of the variable; if False will return the assign op.

    Returns:
      A `Tensor` that will hold the new value of this variable after
      the subtraction has completed.
    "
  [self delta & {:keys [use_locking name read_value]
                       :or {name None}} ]
    (py/call-attr-kw self "assign_sub" [delta] {:use_locking use_locking :name name :read_value read_value }))

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
      the scattered assignment has completed.

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

(defn device 
  "The device of this variable."
  [ self ]
    (py/call-attr self "device"))

(defn dtype 
  "The `DType` of this variable."
  [ self ]
    (py/call-attr self "dtype"))

(defn eval 
  "In a session, computes and returns the value of this variable.

    This is not a graph construction method, it does not add ops to the graph.

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
        print(v.eval(sess))
        # Usage with the default session.  The 'with' block
        # above makes 'sess' the default session.
        print(v.eval())
    ```

    Args:
      session: The session to use to evaluate this variable. If none, the
        default session is used.

    Returns:
      A numpy `ndarray` with a copy of the value of this variable.
    "
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
  "Returns a `Variable` object created from `variable_def`."
  [ self variable_def import_scope ]
  (py/call-attr self "from_proto"  self variable_def import_scope ))

(defn gather-nd 
  "Gather slices from `params` into a Tensor with shape specified by `indices`.

    See tf.gather_nd for details.

    Args:
      indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
        Index tensor.
      name: A name for the operation (optional).

    Returns:
      A `Tensor`. Has the same type as `params`.
    "
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

(defn initial-value 
  "Returns the Tensor used as the initial value for the variable.

    Note that this is different from `initialized_value()` which runs
    the op that initializes the variable before returning its value.
    This method returns the tensor that is used by the op that initializes
    the variable.

    Returns:
      A `Tensor`.
    "
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
  "The initializer operation for this variable."
  [ self ]
    (py/call-attr self "initializer"))

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
  "The name of this variable."
  [ self ]
    (py/call-attr self "name"))

(defn op 
  "The `Operation` of this variable."
  [ self ]
    (py/call-attr self "op"))

(defn read-value 
  "Returns the value of this variable, read in the current context.

    Can be different from value() if it's on another device, with control
    dependencies, etc.

    Returns:
      A `Tensor` containing the value of the variable.
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
      sparse_delta: `tf.IndexedSlices` to use as an argument of max with this
        variable.
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
      sparse_delta: `tf.IndexedSlices` to use as an argument of min with this
        variable.
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

    The Variable has rank `P` and `indices` is a `Tensor` of rank `Q`.

    `indices` must be integer tensor, containing indices into self.
    It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.

    The innermost dimension of `indices` (with length `K`) corresponds to
    indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
    dimension of self.

    `updates` is `Tensor` of rank `Q-1+P-K` with shape:

    ```
    [d_0, ..., d_{Q-2}, self.shape[K], ..., self.shape[P-1]].
    ```

    For example, say we want to add 4 scattered elements to a rank-1 tensor to
    8 elements. In Python, that update would look like this:

    ```python
        v = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
        indices = tf.constant([[4], [3], [1] ,[7]])
        updates = tf.constant([9, 10, 11, 12])
        add = v.scatter_nd_add(indices, updates)
        with tf.compat.v1.Session() as sess:
          print sess.run(add)
    ```

    The resulting update to v would look like this:

        [1, 13, 3, 14, 14, 6, 7, 20]

    See `tf.scatter_nd` for more details about how to make updates to
    slices.

    Args:
      indices: The indices to be used in the operation.
      updates: The values to be used in the operation.
      name: the name of the operation.

    Returns:
      A `Tensor` that will hold the new value of this variable after
      the scattered addition has completed.
    "
  [ self indices updates name ]
  (py/call-attr self "scatter_nd_add"  self indices updates name ))

(defn scatter-nd-sub 
  "Applies sparse subtraction to individual values or slices in a Variable.

    Assuming the variable has rank `P` and `indices` is a `Tensor` of rank `Q`.

    `indices` must be integer tensor, containing indices into self.
    It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.

    The innermost dimension of `indices` (with length `K`) corresponds to
    indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
    dimension of self.

    `updates` is `Tensor` of rank `Q-1+P-K` with shape:

    ```
    [d_0, ..., d_{Q-2}, self.shape[K], ..., self.shape[P-1]].
    ```

    For example, say we want to add 4 scattered elements to a rank-1 tensor to
    8 elements. In Python, that update would look like this:

    ```python
        v = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
        indices = tf.constant([[4], [3], [1] ,[7]])
        updates = tf.constant([9, 10, 11, 12])
        op = v.scatter_nd_sub(indices, updates)
        with tf.compat.v1.Session() as sess:
          print sess.run(op)
    ```

    The resulting update to v would look like this:

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

    The Variable has rank `P` and `indices` is a `Tensor` of rank `Q`.

    `indices` must be integer tensor, containing indices into self.
    It must be shape `[d_0, ..., d_{Q-2}, K]` where `0 < K <= P`.

    The innermost dimension of `indices` (with length `K`) corresponds to
    indices into elements (if `K = P`) or slices (if `K < P`) along the `K`th
    dimension of self.

    `updates` is `Tensor` of rank `Q-1+P-K` with shape:

    ```
    [d_0, ..., d_{Q-2}, self.shape[K], ..., self.shape[P-1]].
    ```

    For example, say we want to add 4 scattered elements to a rank-1 tensor to
    8 elements. In Python, that update would look like this:

    ```python
        v = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8])
        indices = tf.constant([[4], [3], [1] ,[7]])
        updates = tf.constant([9, 10, 11, 12])
        op = v.scatter_nd_assign(indices, updates)
        with tf.compat.v1.Session() as sess:
          print sess.run(op)
    ```

    The resulting update to v would look like this:

        [1, 11, 3, 10, 9, 6, 7, 12]

    See `tf.scatter_nd` for more details about how to make updates to
    slices.

    Args:
      indices: The indices to be used in the operation.
      updates: The values to be used in the operation.
      name: the name of the operation.

    Returns:
      A `Tensor` that will hold the new value of this variable after
      the scattered assignment has completed.
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
      the scattered assignment has completed.

    Raises:
      TypeError: if `sparse_delta` is not an `IndexedSlices`.
    "
  [self sparse_delta & {:keys [use_locking name]
                       :or {name None}} ]
    (py/call-attr-kw self "scatter_update" [sparse_delta] {:use_locking use_locking :name name }))

(defn set-shape 
  "Overrides the shape for this variable.

    Args:
      shape: the `TensorShape` representing the overridden shape.
    "
  [ self shape ]
  (py/call-attr self "set_shape"  self shape ))

(defn shape 
  "The `TensorShape` of this variable.

    Returns:
      A `TensorShape`.
    "
  [ self ]
    (py/call-attr self "shape"))

(defn sparse-read 
  "Gather slices from params axis axis according to indices.

    This function supports a subset of tf.gather, see tf.gather for details on
    usage.

    Args:
      indices: The index `Tensor`.  Must be one of the following types: `int32`,
        `int64`. Must be in range `[0, params.shape[axis])`.
      name: A name for the operation (optional).

    Returns:
      A `Tensor`. Has the same type as `params`.
    "
  [ self indices name ]
  (py/call-attr self "sparse_read"  self indices name ))

(defn synchronization 
  ""
  [ self ]
    (py/call-attr self "synchronization"))

(defn to-proto 
  "Converts a `Variable` to a `VariableDef` protocol buffer.

    Args:
      export_scope: Optional `string`. Name scope to remove.

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
  "Returns the last snapshot of this variable.

    You usually do not need to call this method as all ops that need the value
    of the variable call it automatically through a `convert_to_tensor()` call.

    Returns a `Tensor` which holds the value of the variable.  You can not
    assign a new value to this tensor as it is not a reference to the variable.

    To avoid copies, if the consumer of the returned value is on the same device
    as the variable, this actually returns the live value of the variable, not
    a copy.  Updates to the variable are seen by the consumer.  If the consumer
    is on a different device it will get a copy of the variable.

    Returns:
      A `Tensor` containing the value of the variable.
    "
  [ self  ]
  (py/call-attr self "value"  self  ))
