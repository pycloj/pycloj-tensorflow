(ns tensorflow.Tensor
  "Represents one of the outputs of an `Operation`.

  A `Tensor` is a symbolic handle to one of the outputs of an
  `Operation`. It does not hold the values of that operation's output,
  but instead provides a means of computing those values in a
  TensorFlow `tf.compat.v1.Session`.

  This class has two primary purposes:

  1. A `Tensor` can be passed as an input to another `Operation`.
     This builds a dataflow connection between operations, which
     enables TensorFlow to execute an entire `Graph` that represents a
     large, multi-step computation.

  2. After the graph has been launched in a session, the value of the
     `Tensor` can be computed by passing it to
     `tf.Session.run`.
     `t.eval()` is a shortcut for calling
     `tf.compat.v1.get_default_session().run(t)`.

  In the following example, `c`, `d`, and `e` are symbolic `Tensor`
  objects, whereas `result` is a numpy array that stores a concrete
  value:

  ```python
  # Build a dataflow graph.
  c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
  d = tf.constant([[1.0, 1.0], [0.0, 1.0]])
  e = tf.matmul(c, d)

  # Construct a `Session` to execute the graph.
  sess = tf.compat.v1.Session()

  # Execute the graph and store the value that `e` represents in `result`.
  result = sess.run(e)
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
(defonce tensorflow (import-module "tensorflow"))

(defn Tensor 
  "Represents one of the outputs of an `Operation`.

  A `Tensor` is a symbolic handle to one of the outputs of an
  `Operation`. It does not hold the values of that operation's output,
  but instead provides a means of computing those values in a
  TensorFlow `tf.compat.v1.Session`.

  This class has two primary purposes:

  1. A `Tensor` can be passed as an input to another `Operation`.
     This builds a dataflow connection between operations, which
     enables TensorFlow to execute an entire `Graph` that represents a
     large, multi-step computation.

  2. After the graph has been launched in a session, the value of the
     `Tensor` can be computed by passing it to
     `tf.Session.run`.
     `t.eval()` is a shortcut for calling
     `tf.compat.v1.get_default_session().run(t)`.

  In the following example, `c`, `d`, and `e` are symbolic `Tensor`
  objects, whereas `result` is a numpy array that stores a concrete
  value:

  ```python
  # Build a dataflow graph.
  c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
  d = tf.constant([[1.0, 1.0], [0.0, 1.0]])
  e = tf.matmul(c, d)

  # Construct a `Session` to execute the graph.
  sess = tf.compat.v1.Session()

  # Execute the graph and store the value that `e` represents in `result`.
  result = sess.run(e)
  ```
  "
  [ op value_index dtype ]
  (py/call-attr tensorflow "Tensor"  op value_index dtype ))

(defn consumers 
  "Returns a list of `Operation`s that consume this tensor.

    Returns:
      A list of `Operation`s.
    "
  [ self  ]
  (py/call-attr self "consumers"  self  ))

(defn device 
  "The name of the device on which this tensor will be produced, or None."
  [ self ]
    (py/call-attr self "device"))

(defn dtype 
  "The `DType` of elements in this tensor."
  [ self ]
    (py/call-attr self "dtype"))

(defn eval 
  "Evaluates this tensor in a `Session`.

    Calling this method will execute all preceding operations that
    produce the inputs needed for the operation that produces this
    tensor.

    *N.B.* Before invoking `Tensor.eval()`, its graph must have been
    launched in a session, and either a default session must be
    available, or `session` must be specified explicitly.

    Args:
      feed_dict: A dictionary that maps `Tensor` objects to feed values. See
        `tf.Session.run` for a description of the valid feed values.
      session: (Optional.) The `Session` to be used to evaluate this tensor. If
        none, the default session will be used.

    Returns:
      A numpy array corresponding to the value of this tensor.

    "
  [ self feed_dict session ]
  (py/call-attr self "eval"  self feed_dict session ))

(defn experimental-ref 
  "Returns a hashable reference object to this Tensor.

    Warning: Experimental API that could be changed or removed.

    The primary usecase for this API is to put tensors in a set/dictionary.
    We can't put tensors in a set/dictionary as `tensor.__hash__()` is no longer
    available starting Tensorflow 2.0.

    ```python
    import tensorflow as tf

    x = tf.constant(5)
    y = tf.constant(10)
    z = tf.constant(10)

    # The followings will raise an exception starting 2.0
    # TypeError: Tensor is unhashable if Tensor equality is enabled.
    tensor_set = {x, y, z}
    tensor_dict = {x: 'five', y: 'ten', z: 'ten'}
    ```

    Instead, we can use `tensor.experimental_ref()`.

    ```python
    tensor_set = {x.experimental_ref(),
                  y.experimental_ref(),
                  z.experimental_ref()}

    print(x.experimental_ref() in tensor_set)
    ==> True

    tensor_dict = {x.experimental_ref(): 'five',
                   y.experimental_ref(): 'ten',
                   z.experimental_ref(): 'ten'}

    print(tensor_dict[y.experimental_ref()])
    ==> ten
    ```

    Also, the reference object provides `.deref()` function that returns the
    original Tensor.

    ```python
    x = tf.constant(5)
    print(x.experimental_ref().deref())
    ==> tf.Tensor(5, shape=(), dtype=int32)
    ```
    "
  [ self  ]
  (py/call-attr self "experimental_ref"  self  ))

(defn get-shape 
  "Alias of Tensor.shape."
  [ self  ]
  (py/call-attr self "get_shape"  self  ))

(defn graph 
  "The `Graph` that contains this tensor."
  [ self ]
    (py/call-attr self "graph"))

(defn name 
  "The string name of this tensor."
  [ self ]
    (py/call-attr self "name"))

(defn op 
  "The `Operation` that produces this tensor as an output."
  [ self ]
    (py/call-attr self "op"))

(defn set-shape 
  "Updates the shape of this tensor.

    This method can be called multiple times, and will merge the given
    `shape` with the current shape of this tensor. It can be used to
    provide additional information about the shape of this tensor that
    cannot be inferred from the graph alone. For example, this can be used
    to provide additional information about the shapes of images:

    ```python
    _, image_data = tf.compat.v1.TFRecordReader(...).read(...)
    image = tf.image.decode_png(image_data, channels=3)

    # The height and width dimensions of `image` are data dependent, and
    # cannot be computed without executing the op.
    print(image.shape)
    ==> TensorShape([Dimension(None), Dimension(None), Dimension(3)])

    # We know that each image in this dataset is 28 x 28 pixels.
    image.set_shape([28, 28, 3])
    print(image.shape)
    ==> TensorShape([Dimension(28), Dimension(28), Dimension(3)])
    ```

    NOTE: This shape is not enforced at runtime. Setting incorrect shapes can
    result in inconsistencies between the statically-known graph and the runtime
    value of tensors. For runtime validation of the shape, use `tf.ensure_shape`
    instead.

    Args:
      shape: A `TensorShape` representing the shape of this tensor, a
        `TensorShapeProto`, a list, a tuple, or None.

    Raises:
      ValueError: If `shape` is not compatible with the current shape of
        this tensor.
    "
  [ self shape ]
  (py/call-attr self "set_shape"  self shape ))

(defn shape 
  "Returns the `TensorShape` that represents the shape of this tensor.

    The shape is computed using shape inference functions that are
    registered in the Op for each `Operation`.  See
    `tf.TensorShape`
    for more details of what a shape represents.

    The inferred shape of a tensor is used to provide shape
    information without having to launch the graph in a session. This
    can be used for debugging, and providing early error messages. For
    example:

    ```python
    c = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    print(c.shape)
    ==> TensorShape([Dimension(2), Dimension(3)])

    d = tf.constant([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

    print(d.shape)
    ==> TensorShape([Dimension(4), Dimension(2)])

    # Raises a ValueError, because `c` and `d` do not have compatible
    # inner dimensions.
    e = tf.matmul(c, d)

    f = tf.matmul(c, d, transpose_a=True, transpose_b=True)

    print(f.shape)
    ==> TensorShape([Dimension(3), Dimension(4)])
    ```

    In some cases, the inferred shape may have unknown dimensions. If
    the caller has additional information about the values of these
    dimensions, `Tensor.set_shape()` can be used to augment the
    inferred shape.

    Returns:
      A `TensorShape` representing the shape of this tensor.

    "
  [ self ]
    (py/call-attr self "shape"))

(defn value-index 
  "The index of this tensor in the outputs of its `Operation`."
  [ self ]
    (py/call-attr self "value_index"))
