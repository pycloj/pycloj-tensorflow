(ns tensorflow.contrib.eager.python.tfe.Network
  "Represents the composition of a set of Layers.

  *Deprecated*. Please inherit from `tf.keras.Model`, and see its documentation
  for details. `tf.keras.Model` should be a drop-in replacement for
  `tfe.Network` in most cases, but note that `track_layer` is no longer
  necessary or supported. Instead, `Layer` instances are tracked on attribute
  assignment (see the section of `tf.keras.Model`'s documentation on
  subclassing). Since the output of `track_layer` is often assigned to an
  attribute anyway, most code can be ported by simply removing the `track_layer`
  calls.

  `tf.keras.Model` works with all TensorFlow `Layer` instances, including those
  from `tf.layers`, but switching to the `tf.keras.layers` versions along with
  the migration to `tf.keras.Model` is recommended, since it will preserve
  variable names.  Feel free to import it with an alias to avoid excess typing
  :).

  `Network` implements the `Layer` interface and adds convenience methods for
  managing sub-`Layer`s, such as listing variables.

  `Layer`s (including other `Network`s) should be added via `track_layer`. They
  can then be used when overriding the `Network.call` method:

  ```python
  class TwoLayerNetwork(tfe.Network):

    def __init__(self, name):
      super(TwoLayerNetwork, self).__init__(name=name)
      self.layer_one = self.track_layer(tf.compat.v1.layers.Dense(16,
      input_shape=(8,)))
      self.layer_two = self.track_layer(tf.compat.v1.layers.Dense(1,
      input_shape=(16,)))

    def call(self, inputs):
      return self.layer_two(self.layer_one(inputs))
  ```

  After constructing an object and calling the `Network`, a list of variables
  created by tracked `Layer`s is available via `Network.variables`:

  ```python
  net = TwoLayerNetwork(name=\"net\")
  output = net(tf.ones([1, 8]))
  print([v.name for v in net.variables])
  ```

  This example prints variable names, one kernel and one bias per
  `tf.compat.v1.layers.Dense` layer:

  ```
  ['net/dense/kernel:0',
   'net/dense/bias:0',
   'net/dense_1/kernel:0',
   'net/dense_1/bias:0']
  ```

  These variables can be passed to a `Saver` (`tf.compat.v1.train.Saver`, or
  `tf.contrib.eager.Saver` when executing eagerly) to save or restore the
  `Network`, typically alongside a global step and
  `tf.compat.v1.train.Optimizer`
  variables when checkpointing during training.

  Note that the semantics of calling a `Network` with graph execution (i.e. not
  executing eagerly) may change slightly in the future. Currently stateful ops
  are pruned from the graph unless they or something that depends on them is
  executed in a session, but this behavior is not consistent with eager
  execution (where stateful ops are executed eagerly). `Layer`s from `tf.layers`
  do not depend on this pruning and so will not be affected, but `Network`s
  which rely on stateful ops being added to the graph but not executed (e.g. via
  custom `Layer`s which manage stateful ops) may break with this change.
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

(defn Network 
  "Represents the composition of a set of Layers.

  *Deprecated*. Please inherit from `tf.keras.Model`, and see its documentation
  for details. `tf.keras.Model` should be a drop-in replacement for
  `tfe.Network` in most cases, but note that `track_layer` is no longer
  necessary or supported. Instead, `Layer` instances are tracked on attribute
  assignment (see the section of `tf.keras.Model`'s documentation on
  subclassing). Since the output of `track_layer` is often assigned to an
  attribute anyway, most code can be ported by simply removing the `track_layer`
  calls.

  `tf.keras.Model` works with all TensorFlow `Layer` instances, including those
  from `tf.layers`, but switching to the `tf.keras.layers` versions along with
  the migration to `tf.keras.Model` is recommended, since it will preserve
  variable names.  Feel free to import it with an alias to avoid excess typing
  :).

  `Network` implements the `Layer` interface and adds convenience methods for
  managing sub-`Layer`s, such as listing variables.

  `Layer`s (including other `Network`s) should be added via `track_layer`. They
  can then be used when overriding the `Network.call` method:

  ```python
  class TwoLayerNetwork(tfe.Network):

    def __init__(self, name):
      super(TwoLayerNetwork, self).__init__(name=name)
      self.layer_one = self.track_layer(tf.compat.v1.layers.Dense(16,
      input_shape=(8,)))
      self.layer_two = self.track_layer(tf.compat.v1.layers.Dense(1,
      input_shape=(16,)))

    def call(self, inputs):
      return self.layer_two(self.layer_one(inputs))
  ```

  After constructing an object and calling the `Network`, a list of variables
  created by tracked `Layer`s is available via `Network.variables`:

  ```python
  net = TwoLayerNetwork(name=\"net\")
  output = net(tf.ones([1, 8]))
  print([v.name for v in net.variables])
  ```

  This example prints variable names, one kernel and one bias per
  `tf.compat.v1.layers.Dense` layer:

  ```
  ['net/dense/kernel:0',
   'net/dense/bias:0',
   'net/dense_1/kernel:0',
   'net/dense_1/bias:0']
  ```

  These variables can be passed to a `Saver` (`tf.compat.v1.train.Saver`, or
  `tf.contrib.eager.Saver` when executing eagerly) to save or restore the
  `Network`, typically alongside a global step and
  `tf.compat.v1.train.Optimizer`
  variables when checkpointing during training.

  Note that the semantics of calling a `Network` with graph execution (i.e. not
  executing eagerly) may change slightly in the future. Currently stateful ops
  are pruned from the graph unless they or something that depends on them is
  executed in a session, but this behavior is not consistent with eager
  execution (where stateful ops are executed eagerly). `Layer`s from `tf.layers`
  do not depend on this pruning and so will not be affected, but `Network`s
  which rely on stateful ops being added to the graph but not executed (e.g. via
  custom `Layer`s which manage stateful ops) may break with this change.
  "
  [ name ]
  (py/call-attr eager "Network"  name ))

(defn activity-regularizer 
  "Optional regularizer function for the output of this layer."
  [ self ]
    (py/call-attr self "activity_regularizer"))

(defn add-loss 
  ""
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
  ""
  [self name shape dtype initializer regularizer & {:keys [trainable constraint]
                       :or {constraint None}} ]
    (py/call-attr-kw self "add_variable" [name shape dtype initializer regularizer] {:trainable trainable :constraint constraint }))

(defn add-weight 
  "Adds a new variable to the layer, or gets an existing one; returns it.

    Arguments:
      name: variable name.
      shape: variable shape.
      dtype: The type of the variable. Defaults to `self.dtype` or `float32`.
      initializer: initializer instance (callable).
      regularizer: regularizer instance (callable).
      trainable: whether the variable should be part of the layer's
        \"trainable_variables\" (e.g. variables, biases)
        or \"non_trainable_variables\" (e.g. BatchNorm mean, stddev).
        Note, if the current variable scope is marked as non-trainable
        then this parameter is ignored and any added variables are also
        marked as non-trainable. `trainable` defaults to `True` unless
        `synchronization` is set to `ON_READ`.
      constraint: constraint instance (callable).
      use_resource: Whether to use `ResourceVariable`.
      synchronization: Indicates when a distributed a variable will be
        aggregated. Accepted values are constants defined in the class
        `tf.VariableSynchronization`. By default the synchronization is set to
        `AUTO` and the current `DistributionStrategy` chooses
        when to synchronize. If `synchronization` is set to `ON_READ`,
        `trainable` must not be set to `True`.
      aggregation: Indicates how a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableAggregation`.
      partitioner: (optional) partitioner instance (callable).  If
        provided, when the requested variable is created it will be split
        into multiple partitions according to `partitioner`.  In this case,
        an instance of `PartitionedVariable` is returned.  Available
        partitioners include `tf.compat.v1.fixed_size_partitioner` and
        `tf.compat.v1.variable_axis_size_partitioner`.  For more details, see
        the documentation of `tf.compat.v1.get_variable` and the  \"Variable
        Partitioners and Sharding\" section of the API guide.
      **kwargs: Additional keyword arguments.

    Returns:
      The created variable.  Usually either a `Variable` or `ResourceVariable`
      instance.  If `partitioner` is not `None`, a `PartitionedVariable`
      instance is returned.

    Raises:
      RuntimeError: If called with partitioned variable regularization and
        eager execution is enabled.
      ValueError: When trainable has been set to True with synchronization
        set as `ON_READ`.
    "
  [self name shape dtype initializer regularizer trainable constraint use_resource & {:keys [synchronization aggregation partitioner]
                       :or {partitioner None}} ]
    (py/call-attr-kw self "add_weight" [name shape dtype initializer regularizer trainable constraint use_resource] {:synchronization synchronization :aggregation aggregation :partitioner partitioner }))

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
  "Returns the config of the layer.

    A layer config is a Python dictionary (serializable)
    containing the configuration of a layer.
    The same layer can be reinstantiated later
    (without its trained weights) from this configuration.

    The config of a layer does not include connectivity
    information, nor the layer class name. These are handled
    by `Network` (one layer of abstraction above).

    Returns:
        Python dictionary.
    "
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

(defn get-layer 
  "Get a contained `tf.compat.v1.layers.Layer` either by name or index.

    Args:
      name: String matching one of the names of a contained `Layer`. Note that
        the names of `Layer`s added to `Network`s may not be unique when doing
        layer sharing (i.e. adding a `Layer` to this `Network` which was already
        added to another `Network`). The lowest index `Layer` with a matching
        name will be returned.
      index: Integer in [0, number of layers). Layers are assigned an index by
        the order they are added.

    Returns:
      A `tf.compat.v1.layers.Layer` object.

    Raises:
      ValueError: If neither or both of 'index' or 'name' is specified, or the
        lookup failed.
    "
  [ self name index ]
  (py/call-attr self "get_layer"  self name index ))

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

(defn graph 
  "DEPRECATED FUNCTION

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Stop using this property because tf.layers layers no longer track their graph."
  [ self ]
    (py/call-attr self "graph"))

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

(defn layers 
  ""
  [ self ]
    (py/call-attr self "layers"))

(defn losses 
  "Gather losses from `Layer`s in the `Network`.

    Note that when executing eagerly, `Layer.losses` evaluates
    regularizers. When using graph execution, variable regularization ops have
    already been created and are simply returned here.

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

(defn scope-name 
  ""
  [ self ]
    (py/call-attr self "scope_name"))

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

(defn track-layer 
  "Track a Layer in this Network.

    `Network` requires that all `Layer`s used in `call()` be tracked so that the
    `Network` can export a complete list of variables.

    Args:
      layer: A `tf.compat.v1.layers.Layer` object.

    Returns:
      The passed in `layer`.

    Raises:
      RuntimeError: If __init__ has not been called.
      TypeError: If `layer` is the wrong type.
      ValueError: If a `Layer` with the same name has already been added.
    "
  [ self layer ]
  (py/call-attr self "track_layer"  self layer ))

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
  ""
  [ self ]
    (py/call-attr self "weights"))
