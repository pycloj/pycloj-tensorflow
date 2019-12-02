(ns tensorflow.data.Iterator
  "Represents the state of iterating through a `Dataset`."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce data (import-module "tensorflow.data"))

(defn Iterator 
  "Represents the state of iterating through a `Dataset`."
  [ iterator_resource initializer output_types output_shapes output_classes ]
  (py/call-attr data "Iterator"  iterator_resource initializer output_types output_shapes output_classes ))

(defn element-spec 
  "The type specification of an element of this iterator.

    Returns:
      A nested structure of `tf.TypeSpec` objects matching the structure of an
      element of this iterator and specifying the type of individual components.
    "
  [ self ]
    (py/call-attr self "element_spec"))

(defn from-string-handle 
  "Creates a new, uninitialized `Iterator` based on the given handle.

    This method allows you to define a \"feedable\" iterator where you can choose
    between concrete iterators by feeding a value in a `tf.Session.run` call.
    In that case, `string_handle` would be a `tf.compat.v1.placeholder`, and you
    would
    feed it with the value of `tf.data.Iterator.string_handle` in each step.

    For example, if you had two iterators that marked the current position in
    a training dataset and a test dataset, you could choose which to use in
    each step as follows:

    ```python
    train_iterator = tf.data.Dataset(...).make_one_shot_iterator()
    train_iterator_handle = sess.run(train_iterator.string_handle())

    test_iterator = tf.data.Dataset(...).make_one_shot_iterator()
    test_iterator_handle = sess.run(test_iterator.string_handle())

    handle = tf.compat.v1.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_iterator.output_types)

    next_element = iterator.get_next()
    loss = f(next_element)

    train_loss = sess.run(loss, feed_dict={handle: train_iterator_handle})
    test_loss = sess.run(loss, feed_dict={handle: test_iterator_handle})
    ```

    Args:
      string_handle: A scalar `tf.Tensor` of type `tf.string` that evaluates to
        a handle produced by the `Iterator.string_handle()` method.
      output_types: A nested structure of `tf.DType` objects corresponding to
        each component of an element of this dataset.
      output_shapes: (Optional.) A nested structure of `tf.TensorShape` objects
        corresponding to each component of an element of this dataset. If
        omitted, each component will have an unconstrainted shape.
      output_classes: (Optional.) A nested structure of Python `type` objects
        corresponding to each component of an element of this iterator. If
        omitted, each component is assumed to be of type `tf.Tensor`.

    Returns:
      An `Iterator`.
    "
  [ self string_handle output_types output_shapes output_classes ]
  (py/call-attr self "from_string_handle"  self string_handle output_types output_shapes output_classes ))

(defn from-structure 
  "Creates a new, uninitialized `Iterator` with the given structure.

    This iterator-constructing method can be used to create an iterator that
    is reusable with many different datasets.

    The returned iterator is not bound to a particular dataset, and it has
    no `initializer`. To initialize the iterator, run the operation returned by
    `Iterator.make_initializer(dataset)`.

    The following is an example

    ```python
    iterator = Iterator.from_structure(tf.int64, tf.TensorShape([]))

    dataset_range = Dataset.range(10)
    range_initializer = iterator.make_initializer(dataset_range)

    dataset_evens = dataset_range.filter(lambda x: x % 2 == 0)
    evens_initializer = iterator.make_initializer(dataset_evens)

    # Define a model based on the iterator; in this example, the model_fn
    # is expected to take scalar tf.int64 Tensors as input (see
    # the definition of 'iterator' above).
    prediction, loss = model_fn(iterator.get_next())

    # Train for `num_epochs`, where for each epoch, we first iterate over
    # dataset_range, and then iterate over dataset_evens.
    for _ in range(num_epochs):
      # Initialize the iterator to `dataset_range`
      sess.run(range_initializer)
      while True:
        try:
          pred, loss_val = sess.run([prediction, loss])
        except tf.errors.OutOfRangeError:
          break

      # Initialize the iterator to `dataset_evens`
      sess.run(evens_initializer)
      while True:
        try:
          pred, loss_val = sess.run([prediction, loss])
        except tf.errors.OutOfRangeError:
          break
    ```

    Args:
      output_types: A nested structure of `tf.DType` objects corresponding to
        each component of an element of this dataset.
      output_shapes: (Optional.) A nested structure of `tf.TensorShape` objects
        corresponding to each component of an element of this dataset. If
        omitted, each component will have an unconstrainted shape.
      shared_name: (Optional.) If non-empty, this iterator will be shared under
        the given name across multiple sessions that share the same devices
        (e.g. when using a remote server).
      output_classes: (Optional.) A nested structure of Python `type` objects
        corresponding to each component of an element of this iterator. If
        omitted, each component is assumed to be of type `tf.Tensor`.

    Returns:
      An `Iterator`.

    Raises:
      TypeError: If the structures of `output_shapes` and `output_types` are
        not the same.
    "
  [ self output_types output_shapes shared_name output_classes ]
  (py/call-attr self "from_structure"  self output_types output_shapes shared_name output_classes ))

(defn get-next 
  "Returns a nested structure of `tf.Tensor`s representing the next element.

    In graph mode, you should typically call this method *once* and use its
    result as the input to another computation. A typical loop will then call
    `tf.Session.run` on the result of that computation. The loop will terminate
    when the `Iterator.get_next()` operation raises
    `tf.errors.OutOfRangeError`. The following skeleton shows how to use
    this method when building a training loop:

    ```python
    dataset = ...  # A `tf.data.Dataset` object.
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    # Build a TensorFlow graph that does something with each element.
    loss = model_function(next_element)
    optimizer = ...  # A `tf.compat.v1.train.Optimizer` object.
    train_op = optimizer.minimize(loss)

    with tf.compat.v1.Session() as sess:
      try:
        while True:
          sess.run(train_op)
      except tf.errors.OutOfRangeError:
        pass
    ```

    NOTE: It is legitimate to call `Iterator.get_next()` multiple times, e.g.
    when you are distributing different elements to multiple devices in a single
    step. However, a common pitfall arises when users call `Iterator.get_next()`
    in each iteration of their training loop. `Iterator.get_next()` adds ops to
    the graph, and executing each op allocates resources (including threads); as
    a consequence, invoking it in every iteration of a training loop causes
    slowdown and eventual resource exhaustion. To guard against this outcome, we
    log a warning when the number of uses crosses a fixed threshold of
    suspiciousness.

    Args:
      name: (Optional.) A name for the created operation.

    Returns:
      A nested structure of `tf.Tensor` objects.
    "
  [ self name ]
  (py/call-attr self "get_next"  self name ))

(defn initializer 
  "A `tf.Operation` that should be run to initialize this iterator.

    Returns:
      A `tf.Operation` that should be run to initialize this iterator

    Raises:
      ValueError: If this iterator initializes itself automatically.
    "
  [ self ]
    (py/call-attr self "initializer"))

(defn make-initializer 
  "Returns a `tf.Operation` that initializes this iterator on `dataset`.

    Args:
      dataset: A `Dataset` with compatible structure to this iterator.
      name: (Optional.) A name for the created operation.

    Returns:
      A `tf.Operation` that can be run to initialize this iterator on the given
      `dataset`.

    Raises:
      TypeError: If `dataset` and this iterator do not have a compatible
        element structure.
    "
  [ self dataset name ]
  (py/call-attr self "make_initializer"  self dataset name ))

(defn output-classes 
  "Returns the class of each component of an element of this iterator. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use `tf.compat.v1.data.get_output_classes(iterator)`.

The expected values are `tf.Tensor` and `tf.SparseTensor`.

Returns:
  A nested structure of Python `type` objects corresponding to each
  component of an element of this dataset."
  [ self ]
    (py/call-attr self "output_classes"))

(defn output-shapes 
  "Returns the shape of each component of an element of this iterator. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use `tf.compat.v1.data.get_output_shapes(iterator)`.

Returns:
  A nested structure of `tf.TensorShape` objects corresponding to each
  component of an element of this dataset."
  [ self ]
    (py/call-attr self "output_shapes"))

(defn output-types 
  "Returns the type of each component of an element of this iterator. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use `tf.compat.v1.data.get_output_types(iterator)`.

Returns:
  A nested structure of `tf.DType` objects corresponding to each component
  of an element of this dataset."
  [ self ]
    (py/call-attr self "output_types"))

(defn string-handle 
  "Returns a string-valued `tf.Tensor` that represents this iterator.

    Args:
      name: (Optional.) A name for the created operation.

    Returns:
      A scalar `tf.Tensor` of type `tf.string`.
    "
  [ self name ]
  (py/call-attr self "string_handle"  self name ))
