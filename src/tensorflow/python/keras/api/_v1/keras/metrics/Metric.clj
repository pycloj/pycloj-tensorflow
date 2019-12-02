(ns tensorflow.python.keras.api.-v1.keras.metrics.Metric
  "Encapsulates metric logic and state.

  Usage:

  ```python
  m = SomeMetric(...)
  for input in ...:
    m.update_state(input)
  print('Final result: ', m.result().numpy())
  ```

  Usage with tf.keras API:

  ```python
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(64, activation='relu'))
  model.add(tf.keras.layers.Dense(64, activation='relu'))
  model.add(tf.keras.layers.Dense(10, activation='softmax'))

  model.compile(optimizer=tf.compat.v1.train.RMSPropOptimizer(0.01),
                loss=tf.keras.losses.categorical_crossentropy,
                metrics=[tf.keras.metrics.CategoricalAccuracy()])

  data = np.random.random((1000, 32))
  labels = np.random.random((1000, 10))

  dataset = tf.data.Dataset.from_tensor_slices((data, labels))
  dataset = dataset.batch(32)
  dataset = dataset.repeat()

  model.fit(dataset, epochs=10, steps_per_epoch=30)
  ```

  To be implemented by subclasses:
  * `__init__()`: All state variables should be created in this method by
    calling `self.add_weight()` like: `self.var = self.add_weight(...)`
  * `update_state()`: Has all updates to the state variables like:
    self.var.assign_add(...).
  * `result()`: Computes and returns a value for the metric
    from the state variables.

  Example subclass implementation:

  ```
  class BinaryTruePositives(tf.keras.metrics.Metric):

    def __init__(self, name='binary_true_positives', **kwargs):
      super(BinaryTruePositives, self).__init__(name=name, **kwargs)
      self.true_positives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
      y_true = tf.cast(y_true, tf.bool)
      y_pred = tf.cast(y_pred, tf.bool)

      values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
      values = tf.cast(values, self.dtype)
      if sample_weight is not None:
        sample_weight = tf.cast(sample_weight, self.dtype)
        sample_weight = tf.broadcast_weights(sample_weight, values)
        values = tf.multiply(values, sample_weight)
      self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
      return self.true_positives
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
(defonce metrics (import-module "tensorflow.python.keras.api._v1.keras.metrics"))

(defn Metric 
  "Encapsulates metric logic and state.

  Usage:

  ```python
  m = SomeMetric(...)
  for input in ...:
    m.update_state(input)
  print('Final result: ', m.result().numpy())
  ```

  Usage with tf.keras API:

  ```python
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(64, activation='relu'))
  model.add(tf.keras.layers.Dense(64, activation='relu'))
  model.add(tf.keras.layers.Dense(10, activation='softmax'))

  model.compile(optimizer=tf.compat.v1.train.RMSPropOptimizer(0.01),
                loss=tf.keras.losses.categorical_crossentropy,
                metrics=[tf.keras.metrics.CategoricalAccuracy()])

  data = np.random.random((1000, 32))
  labels = np.random.random((1000, 10))

  dataset = tf.data.Dataset.from_tensor_slices((data, labels))
  dataset = dataset.batch(32)
  dataset = dataset.repeat()

  model.fit(dataset, epochs=10, steps_per_epoch=30)
  ```

  To be implemented by subclasses:
  * `__init__()`: All state variables should be created in this method by
    calling `self.add_weight()` like: `self.var = self.add_weight(...)`
  * `update_state()`: Has all updates to the state variables like:
    self.var.assign_add(...).
  * `result()`: Computes and returns a value for the metric
    from the state variables.

  Example subclass implementation:

  ```
  class BinaryTruePositives(tf.keras.metrics.Metric):

    def __init__(self, name='binary_true_positives', **kwargs):
      super(BinaryTruePositives, self).__init__(name=name, **kwargs)
      self.true_positives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
      y_true = tf.cast(y_true, tf.bool)
      y_pred = tf.cast(y_pred, tf.bool)

      values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
      values = tf.cast(values, self.dtype)
      if sample_weight is not None:
        sample_weight = tf.cast(sample_weight, self.dtype)
        sample_weight = tf.broadcast_weights(sample_weight, values)
        values = tf.multiply(values, sample_weight)
      self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
      return self.true_positives
  ```
  "
  [  ]
  (py/call-attr metrics "Metric"  ))

(defn activity-regularizer 
  "Optional regularizer function for the output of this layer."
  [ self ]
    (py/call-attr self "activity_regularizer"))

(defn add-loss 
  "Add loss tensor(s), potentially dependent on layer inputs.

    Some losses (for instance, activity regularization losses) may be dependent
    on the inputs passed when calling a layer. Hence, when reusing the same
    layer on different inputs `a` and `b`, some entries in `layer.losses` may
    be dependent on `a` and some on `b`. This method automatically keeps track
    of dependencies.

    This method can be used inside a subclassed layer or model's `call`
    function, in which case `losses` should be a Tensor or list of Tensors.

    Example:

    ```python
    class MyLayer(tf.keras.layers.Layer):
      def call(inputs, self):
        self.add_loss(tf.abs(tf.reduce_mean(inputs)), inputs=True)
        return inputs
    ```

    This method can also be called directly on a Functional Model during
    construction. In this case, any loss Tensors passed to this Model must
    be symbolic and be able to be traced back to the model's `Input`s. These
    losses become part of the model's topology and are tracked in `get_config`.

    Example:

    ```python
    inputs = tf.keras.Input(shape=(10,))
    x = tf.keras.layers.Dense(10)(inputs)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    # Actvity regularization.
    model.add_loss(tf.abs(tf.reduce_mean(x)))
    ```

    If this is not the case for your loss (if, for example, your loss references
    a `Variable` of one of the model's layers), you can wrap your loss in a
    zero-argument lambda. These losses are not tracked as part of the model's
    topology since they can't be serialized.

    Example:

    ```python
    inputs = tf.keras.Input(shape=(10,))
    x = tf.keras.layers.Dense(10)(inputs)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    # Weight regularization.
    model.add_loss(lambda: tf.reduce_mean(x.kernel))
    ```

    The `get_losses_for` method allows to retrieve the losses relevant to a
    specific set of inputs.

    Arguments:
      losses: Loss tensor, or list/tuple of tensors. Rather than tensors, losses
        may also be zero-argument callables which create a loss tensor.
      inputs: Ignored when executing eagerly. If anything other than None is
        passed, it signals the losses are conditional on some of the layer's
        inputs, and thus they should only be run where these inputs are
        available. This is the case for activity regularization losses, for
        instance. If `None` is passed, the losses are assumed
        to be unconditional, and will apply across all dataflows of the layer
        (e.g. weight regularization losses).
    "
  [ self losses inputs ]
  (py/call-attr self "add_loss"  self losses inputs ))

(defn add-metric 
  "Adds metric tensor to the layer.

    Args:
      value: Metric tensor.
      aggregation: Sample-wise metric reduction function. If `aggregation=None`,
        it indicates that the metric tensor provided has been aggregated
        already. eg, `bin_acc = BinaryAccuracy(name='acc')` followed by
        `model.add_metric(bin_acc(y_true, y_pred))`. If aggregation='mean', the
        given metric tensor will be sample-wise reduced using `mean` function.
        eg, `model.add_metric(tf.reduce_sum(outputs), name='output_mean',
        aggregation='mean')`.
      name: String metric name.

    Raises:
      ValueError: If `aggregation` is anything other than None or `mean`.
    "
  [ self value aggregation name ]
  (py/call-attr self "add_metric"  self value aggregation name ))

(defn add-update 
  "Add update op(s), potentially dependent on layer inputs. (deprecated arguments)

Warning: SOME ARGUMENTS ARE DEPRECATED: `(inputs)`. They will be removed in a future version.
Instructions for updating:
`inputs` is now automatically inferred

Weight updates (for instance, the updates of the moving mean and variance
in a BatchNormalization layer) may be dependent on the inputs passed
when calling a layer. Hence, when reusing the same layer on
different inputs `a` and `b`, some entries in `layer.updates` may be
dependent on `a` and some on `b`. This method automatically keeps track
of dependencies.

The `get_updates_for` method allows to retrieve the updates relevant to a
specific set of inputs.

This call is ignored when eager execution is enabled (in that case, variable
updates are run on the fly and thus do not need to be tracked for later
execution).

Arguments:
  updates: Update op, or list/tuple of update ops, or zero-arg callable
    that returns an update op. A zero-arg callable should be passed in
    order to disable running the updates by setting `trainable=False`
    on this Layer, when executing in Eager mode.
  inputs: Deprecated, will be automatically inferred."
  [ self updates inputs ]
  (py/call-attr self "add_update"  self updates inputs ))

(defn add-variable 
  "Deprecated, do NOT use! Alias for `add_weight`. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please use `layer.add_weight` method instead."
  [ self  ]
  (py/call-attr self "add_variable"  self  ))

(defn add-weight 
  "Adds state variable. Only for use by subclasses."
  [self name & {:keys [shape aggregation synchronization initializer dtype]
                       :or {initializer None dtype None}} ]
    (py/call-attr-kw self "add_weight" [name] {:shape shape :aggregation aggregation :synchronization synchronization :initializer initializer :dtype dtype }))

(defn apply 
  "Deprecated, do NOT use! (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please use `layer.__call__` method instead.

This is an alias of `self.__call__`.

Arguments:
  inputs: Input tensor(s).
  *args: additional positional arguments to be passed to `self.call`.
  **kwargs: additional keyword arguments to be passed to `self.call`.

Returns:
  Output tensor(s)."
  [ self inputs ]
  (py/call-attr self "apply"  self inputs ))

(defn build 
  "Creates the variables of the layer (optional, for subclass implementers).

    This is a method that implementers of subclasses of `Layer` or `Model`
    can override if they need a state-creation step in-between
    layer instantiation and layer call.

    This is typically used to create the weights of `Layer` subclasses.

    Arguments:
      input_shape: Instance of `TensorShape`, or list of instances of
        `TensorShape` if the layer expects a list of inputs
        (one instance per input).
    "
  [ self input_shape ]
  (py/call-attr self "build"  self input_shape ))

(defn call 
  "This is where the layer's logic lives.

    Arguments:
        inputs: Input tensor, or list/tuple of input tensors.
        **kwargs: Additional keyword arguments.

    Returns:
        A tensor or list/tuple of tensors.
    "
  [ self inputs ]
  (py/call-attr self "call"  self inputs ))

(defn compute-mask 
  "Computes an output mask tensor.

    Arguments:
        inputs: Tensor or list of tensors.
        mask: Tensor or list of tensors.

    Returns:
        None or a tensor (or list of tensors,
            one per output tensor of the layer).
    "
  [ self inputs mask ]
  (py/call-attr self "compute_mask"  self inputs mask ))

(defn compute-output-shape 
  "Computes the output shape of the layer.

    If the layer has not been built, this method will call `build` on the
    layer. This assumes that the layer will later be used with inputs that
    match the input shape provided here.

    Arguments:
        input_shape: Shape tuple (tuple of integers)
            or list of shape tuples (one per output tensor of the layer).
            Shape tuples can include None for free dimensions,
            instead of an integer.

    Returns:
        An input shape tuple.
    "
  [ self input_shape ]
  (py/call-attr self "compute_output_shape"  self input_shape ))

(defn compute-output-signature 
  "Compute the output tensor signature of the layer based on the inputs.

    Unlike a TensorShape object, a TensorSpec object contains both shape
    and dtype information for a tensor. This method allows layers to provide
    output dtype information if it is different from the input dtype.
    For any layer that doesn't implement this function,
    the framework will fall back to use `compute_output_shape`, and will
    assume that the output dtype matches the input dtype.

    Args:
      input_signature: Single TensorSpec or nested structure of TensorSpec
        objects, describing a candidate input for the layer.

    Returns:
      Single TensorSpec or nested structure of TensorSpec objects, describing
        how the layer would transform the provided input.

    Raises:
      TypeError: If input_signature contains a non-TensorSpec object.
    "
  [ self input_signature ]
  (py/call-attr self "compute_output_signature"  self input_signature ))

(defn count-params 
  "Count the total number of scalars composing the weights.

    Returns:
        An integer count.

    Raises:
        ValueError: if the layer isn't yet built
          (in which case its weights aren't yet defined).
    "
  [ self  ]
  (py/call-attr self "count_params"  self  ))

(defn dtype 
  ""
  [ self ]
    (py/call-attr self "dtype"))

(defn dynamic 
  ""
  [ self ]
    (py/call-attr self "dynamic"))

(defn get-config 
  "Returns the serializable config of the metric."
  [ self  ]
  (py/call-attr self "get_config"  self  ))

(defn get-input-at 
  "Retrieves the input tensor(s) of a layer at a given node.

    Arguments:
        node_index: Integer, index of the node
            from which to retrieve the attribute.
            E.g. `node_index=0` will correspond to the
            first time the layer was called.

    Returns:
        A tensor (or list of tensors if the layer has multiple inputs).

    Raises:
      RuntimeError: If called in Eager mode.
    "
  [ self node_index ]
  (py/call-attr self "get_input_at"  self node_index ))

(defn get-input-mask-at 
  "Retrieves the input mask tensor(s) of a layer at a given node.

    Arguments:
        node_index: Integer, index of the node
            from which to retrieve the attribute.
            E.g. `node_index=0` will correspond to the
            first time the layer was called.

    Returns:
        A mask tensor
        (or list of tensors if the layer has multiple inputs).
    "
  [ self node_index ]
  (py/call-attr self "get_input_mask_at"  self node_index ))

(defn get-input-shape-at 
  "Retrieves the input shape(s) of a layer at a given node.

    Arguments:
        node_index: Integer, index of the node
            from which to retrieve the attribute.
            E.g. `node_index=0` will correspond to the
            first time the layer was called.

    Returns:
        A shape tuple
        (or list of shape tuples if the layer has multiple inputs).

    Raises:
      RuntimeError: If called in Eager mode.
    "
  [ self node_index ]
  (py/call-attr self "get_input_shape_at"  self node_index ))

(defn get-losses-for 
  "Retrieves losses relevant to a specific set of inputs.

    Arguments:
      inputs: Input tensor or list/tuple of input tensors.

    Returns:
      List of loss tensors of the layer that depend on `inputs`.
    "
  [ self inputs ]
  (py/call-attr self "get_losses_for"  self inputs ))

(defn get-output-at 
  "Retrieves the output tensor(s) of a layer at a given node.

    Arguments:
        node_index: Integer, index of the node
            from which to retrieve the attribute.
            E.g. `node_index=0` will correspond to the
            first time the layer was called.

    Returns:
        A tensor (or list of tensors if the layer has multiple outputs).

    Raises:
      RuntimeError: If called in Eager mode.
    "
  [ self node_index ]
  (py/call-attr self "get_output_at"  self node_index ))

(defn get-output-mask-at 
  "Retrieves the output mask tensor(s) of a layer at a given node.

    Arguments:
        node_index: Integer, index of the node
            from which to retrieve the attribute.
            E.g. `node_index=0` will correspond to the
            first time the layer was called.

    Returns:
        A mask tensor
        (or list of tensors if the layer has multiple outputs).
    "
  [ self node_index ]
  (py/call-attr self "get_output_mask_at"  self node_index ))

(defn get-output-shape-at 
  "Retrieves the output shape(s) of a layer at a given node.

    Arguments:
        node_index: Integer, index of the node
            from which to retrieve the attribute.
            E.g. `node_index=0` will correspond to the
            first time the layer was called.

    Returns:
        A shape tuple
        (or list of shape tuples if the layer has multiple outputs).

    Raises:
      RuntimeError: If called in Eager mode.
    "
  [ self node_index ]
  (py/call-attr self "get_output_shape_at"  self node_index ))

(defn get-updates-for 
  "Retrieves updates relevant to a specific set of inputs.

    Arguments:
      inputs: Input tensor or list/tuple of input tensors.

    Returns:
      List of update ops of the layer that depend on `inputs`.
    "
  [ self inputs ]
  (py/call-attr self "get_updates_for"  self inputs ))

(defn get-weights 
  "Returns the current weights of the layer.

    Returns:
        Weights values as a list of numpy arrays.
    "
  [ self  ]
  (py/call-attr self "get_weights"  self  ))

(defn inbound-nodes 
  "Deprecated, do NOT use! Only for compatibility with external Keras."
  [ self ]
    (py/call-attr self "inbound_nodes"))

(defn input 
  "Retrieves the input tensor(s) of a layer.

    Only applicable if the layer has exactly one input,
    i.e. if it is connected to one incoming layer.

    Returns:
        Input tensor or list of input tensors.

    Raises:
      RuntimeError: If called in Eager mode.
      AttributeError: If no inbound nodes are found.
    "
  [ self ]
    (py/call-attr self "input"))

(defn input-mask 
  "Retrieves the input mask tensor(s) of a layer.

    Only applicable if the layer has exactly one inbound node,
    i.e. if it is connected to one incoming layer.

    Returns:
        Input mask tensor (potentially None) or list of input
        mask tensors.

    Raises:
        AttributeError: if the layer is connected to
        more than one incoming layers.
    "
  [ self ]
    (py/call-attr self "input_mask"))

(defn input-shape 
  "Retrieves the input shape(s) of a layer.

    Only applicable if the layer has exactly one input,
    i.e. if it is connected to one incoming layer, or if all inputs
    have the same shape.

    Returns:
        Input shape, as an integer shape tuple
        (or list of shape tuples, one tuple per input tensor).

    Raises:
        AttributeError: if the layer has no defined input_shape.
        RuntimeError: if called in Eager mode.
    "
  [ self ]
    (py/call-attr self "input_shape"))

(defn input-spec 
  ""
  [ self ]
    (py/call-attr self "input_spec"))

(defn losses 
  "Losses which are associated with this `Layer`.

    Variable regularization tensors are created when this property is accessed,
    so it is eager safe: accessing `losses` under a `tf.GradientTape` will
    propagate gradients back to the corresponding variables.

    Returns:
      A list of tensors.
    "
  [ self ]
    (py/call-attr self "losses"))

(defn metrics 
  ""
  [ self ]
    (py/call-attr self "metrics"))

(defn name 
  ""
  [ self ]
    (py/call-attr self "name"))

(defn name-scope 
  "Returns a `tf.name_scope` instance for this class."
  [ self ]
    (py/call-attr self "name_scope"))

(defn non-trainable-variables 
  ""
  [ self ]
    (py/call-attr self "non_trainable_variables"))

(defn non-trainable-weights 
  ""
  [ self ]
    (py/call-attr self "non_trainable_weights"))

(defn outbound-nodes 
  "Deprecated, do NOT use! Only for compatibility with external Keras."
  [ self ]
    (py/call-attr self "outbound_nodes"))

(defn output 
  "Retrieves the output tensor(s) of a layer.

    Only applicable if the layer has exactly one output,
    i.e. if it is connected to one incoming layer.

    Returns:
      Output tensor or list of output tensors.

    Raises:
      AttributeError: if the layer is connected to more than one incoming
        layers.
      RuntimeError: if called in Eager mode.
    "
  [ self ]
    (py/call-attr self "output"))

(defn output-mask 
  "Retrieves the output mask tensor(s) of a layer.

    Only applicable if the layer has exactly one inbound node,
    i.e. if it is connected to one incoming layer.

    Returns:
        Output mask tensor (potentially None) or list of output
        mask tensors.

    Raises:
        AttributeError: if the layer is connected to
        more than one incoming layers.
    "
  [ self ]
    (py/call-attr self "output_mask"))

(defn output-shape 
  "Retrieves the output shape(s) of a layer.

    Only applicable if the layer has one output,
    or if all outputs have the same shape.

    Returns:
        Output shape, as an integer shape tuple
        (or list of shape tuples, one tuple per output tensor).

    Raises:
        AttributeError: if the layer has no defined output shape.
        RuntimeError: if called in Eager mode.
    "
  [ self ]
    (py/call-attr self "output_shape"))

(defn reset-states 
  "Resets all of the metric state variables.

    This function is called between epochs/steps,
    when a metric is evaluated during training.
    "
  [ self  ]
  (py/call-attr self "reset_states"  self  ))

(defn result 
  "Computes and returns the metric value tensor.

    Result computation is an idempotent operation that simply calculates the
    metric value using the state variables.
    "
  [ self  ]
  (py/call-attr self "result"  self  ))

(defn set-weights 
  "Sets the weights of the layer, from Numpy arrays.

    Arguments:
        weights: a list of Numpy arrays. The number
            of arrays and their shape must match
            number of the dimensions of the weights
            of the layer (i.e. it should match the
            output of `get_weights`).

    Raises:
        ValueError: If the provided weights list does not match the
            layer's specifications.
    "
  [ self weights ]
  (py/call-attr self "set_weights"  self weights ))

(defn submodules 
  "Sequence of all sub-modules.

    Submodules are modules which are properties of this module, or found as
    properties of modules which are properties of this module (and so on).

    ```
    a = tf.Module()
    b = tf.Module()
    c = tf.Module()
    a.b = b
    b.c = c
    assert list(a.submodules) == [b, c]
    assert list(b.submodules) == [c]
    assert list(c.submodules) == []
    ```

    Returns:
      A sequence of all submodules.
    "
  [ self ]
    (py/call-attr self "submodules"))

(defn trainable 
  ""
  [ self ]
    (py/call-attr self "trainable"))

(defn trainable-variables 
  ""
  [ self ]
    (py/call-attr self "trainable_variables"))

(defn trainable-weights 
  ""
  [ self ]
    (py/call-attr self "trainable_weights"))

(defn update-state 
  "Accumulates statistics for the metric.

    Note: This function is executed as a graph function in graph mode.
    This means:
      a) Operations on the same resource are executed in textual order.
         This should make it easier to do things like add the updated
         value of a variable to another, for example.
      b) You don't need to worry about collecting the update ops to execute.
         All update ops added to the graph by this function will be executed.
      As a result, code should generally work the same way with graph or
      eager execution.

    Please use `tf.config.experimental_run_functions_eagerly(True)` to execute
    this function eagerly for debugging or profiling.

    Args:
      *args:
      **kwargs: A mini-batch of inputs to the Metric.
    "
  [ self  ]
  (py/call-attr self "update_state"  self  ))

(defn updates 
  ""
  [ self ]
    (py/call-attr self "updates"))

(defn variables 
  "Returns the list of all layer variables/weights.

    Alias of `self.weights`.

    Returns:
      A list of variables.
    "
  [ self ]
    (py/call-attr self "variables"))

(defn weights 
  "Returns the list of all layer variables/weights.

    Returns:
      A list of variables.
    "
  [ self ]
    (py/call-attr self "weights"))
