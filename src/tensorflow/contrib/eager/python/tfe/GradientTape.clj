(ns tensorflow.contrib.eager.python.tfe.GradientTape
  "Record operations for automatic differentiation.

  Operations are recorded if they are executed within this context manager and
  at least one of their inputs is being \"watched\".

  Trainable variables (created by `tf.Variable` or `tf.compat.v1.get_variable`,
  where `trainable=True` is default in both cases) are automatically watched.
  Tensors can be manually watched by invoking the `watch` method on this context
  manager.

  For example, consider the function `y = x * x`. The gradient at `x = 3.0` can
  be computed as:

  ```python
  x = tf.constant(3.0)
  with tf.GradientTape() as g:
    g.watch(x)
    y = x * x
  dy_dx = g.gradient(y, x) # Will compute to 6.0
  ```

  GradientTapes can be nested to compute higher-order derivatives. For example,

  ```python
  x = tf.constant(3.0)
  with tf.GradientTape() as g:
    g.watch(x)
    with tf.GradientTape() as gg:
      gg.watch(x)
      y = x * x
    dy_dx = gg.gradient(y, x)     # Will compute to 6.0
  d2y_dx2 = g.gradient(dy_dx, x)  # Will compute to 2.0
  ```

  By default, the resources held by a GradientTape are released as soon as
  GradientTape.gradient() method is called. To compute multiple gradients over
  the same computation, create a persistent gradient tape. This allows multiple
  calls to the gradient() method as resources are released when the tape object
  is garbage collected. For example:

  ```python
  x = tf.constant(3.0)
  with tf.GradientTape(persistent=True) as g:
    g.watch(x)
    y = x * x
    z = y * y
  dz_dx = g.gradient(z, x)  # 108.0 (4*x^3 at x = 3)
  dy_dx = g.gradient(y, x)  # 6.0
  del g  # Drop the reference to the tape
  ```

  By default GradientTape will automatically watch any trainable variables that
  are accessed inside the context. If you want fine grained control over which
  variables are watched you can disable automatic tracking by passing
  `watch_accessed_variables=False` to the tape constructor:

  ```python
  with tf.GradientTape(watch_accessed_variables=False) as tape:
    tape.watch(variable_a)
    y = variable_a ** 2  # Gradients will be available for `variable_a`.
    z = variable_b ** 3  # No gradients will be available since `variable_b` is
                         # not being watched.
  ```

  Note that when using models you should ensure that your variables exist when
  using `watch_accessed_variables=False`. Otherwise it's quite easy to make your
  first iteration not have any gradients:

  ```python
  a = tf.keras.layers.Dense(32)
  b = tf.keras.layers.Dense(32)

  with tf.GradientTape(watch_accessed_variables=False) as tape:
    tape.watch(a.variables)  # Since `a.build` has not been called at this point
                             # `a.variables` will return an empty list and the
                             # tape will not be watching anything.
    result = b(a(inputs))
    tape.gradient(result, a.variables)  # The result of this computation will be
                                        # a list of `None`s since a's variables
                                        # are not being watched.
  ```

  Note that only tensors with real or complex dtypes are differentiable.
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

(defn GradientTape 
  "Record operations for automatic differentiation.

  Operations are recorded if they are executed within this context manager and
  at least one of their inputs is being \"watched\".

  Trainable variables (created by `tf.Variable` or `tf.compat.v1.get_variable`,
  where `trainable=True` is default in both cases) are automatically watched.
  Tensors can be manually watched by invoking the `watch` method on this context
  manager.

  For example, consider the function `y = x * x`. The gradient at `x = 3.0` can
  be computed as:

  ```python
  x = tf.constant(3.0)
  with tf.GradientTape() as g:
    g.watch(x)
    y = x * x
  dy_dx = g.gradient(y, x) # Will compute to 6.0
  ```

  GradientTapes can be nested to compute higher-order derivatives. For example,

  ```python
  x = tf.constant(3.0)
  with tf.GradientTape() as g:
    g.watch(x)
    with tf.GradientTape() as gg:
      gg.watch(x)
      y = x * x
    dy_dx = gg.gradient(y, x)     # Will compute to 6.0
  d2y_dx2 = g.gradient(dy_dx, x)  # Will compute to 2.0
  ```

  By default, the resources held by a GradientTape are released as soon as
  GradientTape.gradient() method is called. To compute multiple gradients over
  the same computation, create a persistent gradient tape. This allows multiple
  calls to the gradient() method as resources are released when the tape object
  is garbage collected. For example:

  ```python
  x = tf.constant(3.0)
  with tf.GradientTape(persistent=True) as g:
    g.watch(x)
    y = x * x
    z = y * y
  dz_dx = g.gradient(z, x)  # 108.0 (4*x^3 at x = 3)
  dy_dx = g.gradient(y, x)  # 6.0
  del g  # Drop the reference to the tape
  ```

  By default GradientTape will automatically watch any trainable variables that
  are accessed inside the context. If you want fine grained control over which
  variables are watched you can disable automatic tracking by passing
  `watch_accessed_variables=False` to the tape constructor:

  ```python
  with tf.GradientTape(watch_accessed_variables=False) as tape:
    tape.watch(variable_a)
    y = variable_a ** 2  # Gradients will be available for `variable_a`.
    z = variable_b ** 3  # No gradients will be available since `variable_b` is
                         # not being watched.
  ```

  Note that when using models you should ensure that your variables exist when
  using `watch_accessed_variables=False`. Otherwise it's quite easy to make your
  first iteration not have any gradients:

  ```python
  a = tf.keras.layers.Dense(32)
  b = tf.keras.layers.Dense(32)

  with tf.GradientTape(watch_accessed_variables=False) as tape:
    tape.watch(a.variables)  # Since `a.build` has not been called at this point
                             # `a.variables` will return an empty list and the
                             # tape will not be watching anything.
    result = b(a(inputs))
    tape.gradient(result, a.variables)  # The result of this computation will be
                                        # a list of `None`s since a's variables
                                        # are not being watched.
  ```

  Note that only tensors with real or complex dtypes are differentiable.
  "
  [ & {:keys [persistent watch_accessed_variables]} ]
   (py/call-attr-kw eager "GradientTape" [] {:persistent persistent :watch_accessed_variables watch_accessed_variables }))

(defn batch-jacobian 
  "Computes and stacks per-example jacobians.

    See [wikipedia article](http://en.wikipedia.org/wiki/jacobian_matrix_and_determinant) for the
    definition of a Jacobian. This function is essentially an efficient
    implementation of the following:

    `tf.stack([self.jacobian(y[i], x[i]) for i in range(x.shape[0])])`.

    Note that compared to `GradientTape.jacobian` which computes gradient of
    each output value w.r.t each input value, this function is useful when
    `target[i,...]` is independent of `source[j,...]` for `j != i`. This
    assumption allows more efficient computation as compared to
    `GradientTape.jacobian`. The output, as well as intermediate activations,
    are lower dimensional and avoid a bunch of redundant zeros which would
    result in the jacobian computation given the independence assumption.

    Example usage:

    ```python
    with tf.GradientTape() as g:
      x = tf.constant([[1., 2.], [3., 4.]], dtype=tf.float32)
      g.watch(x)
      y = x * x
    batch_jacobian = g.batch_jacobian(y, x)
    # batch_jacobian is [[[2,  0], [0,  4]], [[6,  0], [0,  8]]]
    ```

    Args:
      target: A tensor with rank 2 or higher and with shape [b, y1, ..., y_n].
        `target[i,...]` should only depend on `source[i,...]`.
      source: A tensor with rank 2 or higher and with shape [b, x1, ..., x_m].
      unconnected_gradients: a value which can either hold 'none' or 'zero' and
        alters the value which will be returned if the target and sources are
        unconnected. The possible values and effects are detailed in
        'UnconnectedGradients' and it defaults to 'none'.
      parallel_iterations: A knob to control how many iterations are dispatched
        in parallel. This knob can be used to control the total memory usage.
      experimental_use_pfor: If true, uses pfor for computing the Jacobian. Else
        uses a tf.while_loop.

    Returns:
      A tensor `t` with shape [b, y_1, ..., y_n, x1, ..., x_m] where `t[i, ...]`
      is the jacobian of `target[i, ...]` w.r.t. `source[i, ...]`, i.e. stacked
      per-example jacobians.

    Raises:
      RuntimeError: If called on a non-persistent tape with eager execution
        enabled and without enabling experimental_use_pfor.
      ValueError: If vectorization of jacobian computation fails or if first
        dimension of `target` and `source` do not match.
    "
  [self target source & {:keys [unconnected_gradients parallel_iterations experimental_use_pfor]
                       :or {parallel_iterations None}} ]
    (py/call-attr-kw self "batch_jacobian" [target source] {:unconnected_gradients unconnected_gradients :parallel_iterations parallel_iterations :experimental_use_pfor experimental_use_pfor }))
(defn gradient 
  "Computes the gradient using operations recorded in context of this tape.

    Args:
      target: a list or nested structure of Tensors or Variables to be
        differentiated.
      sources: a list or nested structure of Tensors or Variables. `target`
        will be differentiated against elements in `sources`.
      output_gradients: a list of gradients, one for each element of
        target. Defaults to None.
      unconnected_gradients: a value which can either hold 'none' or 'zero' and
        alters the value which will be returned if the target and sources are
        unconnected. The possible values and effects are detailed in
        'UnconnectedGradients' and it defaults to 'none'.

    Returns:
      a list or nested structure of Tensors (or IndexedSlices, or None),
      one for each element in `sources`. Returned structure is the same as
      the structure of `sources`.

    Raises:
      RuntimeError: if called inside the context of the tape, or if called more
       than once on a non-persistent tape.
      ValueError: if the target is a variable or if unconnected gradients is
       called with an unknown value.
    "
  [self target sources output_gradients  & {:keys [unconnected_gradients]} ]
    (py/call-attr-kw self "gradient" [target sources output_gradients] {:unconnected_gradients unconnected_gradients }))

(defn jacobian 
  "Computes the jacobian using operations recorded in context of this tape.

    See [wikipedia article](http://en.wikipedia.org/wiki/jacobian_matrix_and_determinant) for the
    definition of a Jacobian.

    Example usage:

    ```python
    with tf.GradientTape() as g:
      x  = tf.constant([1.0, 2.0])
      g.watch(x)
      y = x * x
    jacobian = g.jacobian(y, x)
    # jacobian value is [[2., 0.], [0., 4.]]
    ```

    Args:
      target: Tensor to be differentiated.
      sources: a list or nested structure of Tensors or Variables. `target`
        will be differentiated against elements in `sources`.
      unconnected_gradients: a value which can either hold 'none' or 'zero' and
        alters the value which will be returned if the target and sources are
        unconnected. The possible values and effects are detailed in
        'UnconnectedGradients' and it defaults to 'none'.
      parallel_iterations: A knob to control how many iterations are dispatched
        in parallel. This knob can be used to control the total memory usage.
      experimental_use_pfor: If true, vectorizes the jacobian computation. Else
        falls back to a sequential while_loop. Vectorization can sometimes fail
        or lead to excessive memory usage. This option can be used to disable
        vectorization in such cases.

    Returns:
      A list or nested structure of Tensors (or None), one for each element in
      `sources`. Returned structure is the same as the structure of `sources`.
      Note if any gradient is sparse (IndexedSlices), jacobian function
      currently makes it dense and returns a Tensor instead. This may change in
      the future.


    Raises:
      RuntimeError: If called on a non-persistent tape with eager execution
        enabled and without enabling experimental_use_pfor.
      ValueError: If vectorization of jacobian computation fails.
    "
  [self target sources & {:keys [unconnected_gradients parallel_iterations experimental_use_pfor]
                       :or {parallel_iterations None}} ]
    (py/call-attr-kw self "jacobian" [target sources] {:unconnected_gradients unconnected_gradients :parallel_iterations parallel_iterations :experimental_use_pfor experimental_use_pfor }))

(defn reset 
  "Clears all information stored in this tape.

    Equivalent to exiting and reentering the tape context manager with a new
    tape. For example, the two following code blocks are equivalent:

    ```
    with tf.GradientTape() as t:
      loss = loss_fn()
    with tf.GradientTape() as t:
      loss += other_loss_fn()
    t.gradient(loss, ...)  # Only differentiates other_loss_fn, not loss_fn


    # The following is equivalent to the above
    with tf.GradientTape() as t:
      loss = loss_fn()
      t.reset()
      loss += other_loss_fn()
    t.gradient(loss, ...)  # Only differentiates other_loss_fn, not loss_fn
    ```

    This is useful if you don't want to exit the context manager for the tape,
    or can't because the desired reset point is inside a control flow construct:

    ```
    with tf.GradientTape() as t:
      loss = ...
      if loss > k:
        t.reset()
    ```
    "
  [ self  ]
  (py/call-attr self "reset"  self  ))

(defn stop-recording 
  "Temporarily stops recording operations on this tape.

    Operations executed while this context manager is active will not be
    recorded on the tape. This is useful for reducing the memory used by tracing
    all computations.

    For example:

    ```
      with tf.GradientTape(persistent=True) as t:
        loss = compute_loss(model)
        with t.stop_recording():
          # The gradient computation below is not traced, saving memory.
          grads = t.gradient(loss, model.variables)
    ```

    Yields:
      None
    Raises:
      RuntimeError: if the tape is not currently recording.
    "
  [ self  ]
  (py/call-attr self "stop_recording"  self  ))

(defn watch 
  "Ensures that `tensor` is being traced by this tape.

    Args:
      tensor: a Tensor or list of Tensors.

    Raises:
      ValueError: if it encounters something that is not a tensor.
    "
  [ self tensor ]
  (py/call-attr self "watch"  self tensor ))

(defn watched-variables 
  "Returns variables watched by this tape in order of construction."
  [ self  ]
  (py/call-attr self "watched_variables"  self  ))
