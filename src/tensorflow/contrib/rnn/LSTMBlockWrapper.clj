(ns tensorflow.contrib.rnn.LSTMBlockWrapper
  "This is a helper class that provides housekeeping for LSTM cells.

  This may be useful for alternative LSTM and similar type of cells.
  The subclasses must implement `_call_cell` method and `num_units` property.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce rnn (import-module "tensorflow.contrib.rnn"))

(defn LSTMBlockWrapper 
  "This is a helper class that provides housekeeping for LSTM cells.

  This may be useful for alternative LSTM and similar type of cells.
  The subclasses must implement `_call_cell` method and `num_units` property.
  "
  [ & {:keys [trainable name dtype]
       :or {name None dtype None}} ]
  
   (py/call-attr-kw rnn "LSTMBlockWrapper" [] {:trainable trainable :name name :dtype dtype }))

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
  "Deprecated, do NOT use! Alias for `add_weight`. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please use `layer.add_weight` method instead."
  [ self  ]
  (py/call-attr self "add_variable"  self  ))

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
  "Run this LSTM on inputs, starting from the given state.

    Args:
      inputs: `3-D` tensor with shape `[time_len, batch_size, input_size]`.
      initial_state: a tuple `(initial_cell_state, initial_output)` with tensors
        of shape `[batch_size, self._num_units]`. If this is not provided, the
        cell is expected to create a zero initial state of type `dtype`.
      dtype: The data type for the initial state and expected output. Required
        if `initial_state` is not provided or RNN state has a heterogeneous
        dtype.
      sequence_length: Specifies the length of each sequence in inputs. An
        `int32` or `int64` vector (tensor) size `[batch_size]`, values in `[0,
        time_len).`
        Defaults to `time_len` for each element.

    Returns:
      A pair containing:

      - Output: A `3-D` tensor of shape `[time_len, batch_size, output_size]`
        or a list of time_len tensors of shape `[batch_size, output_size]`,
        to match the type of the `inputs`.
      - Final state: a tuple `(cell_state, output)` matching `initial_state`.

    Raises:
      ValueError: in case of shape mismatches
    "
  [ self inputs initial_state dtype sequence_length ]
  (py/call-attr self "call"  self inputs initial_state dtype sequence_length ))

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

(defn num-units 
  "Number of units in this cell (output dimension)."
  [ self ]
    (py/call-attr self "num_units"))

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
  "Returns the list of all layer variables/weights.

    Returns:
      A list of variables.
    "
  [ self ]
    (py/call-attr self "weights"))
