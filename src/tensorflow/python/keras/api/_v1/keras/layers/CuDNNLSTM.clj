(ns tensorflow.python.keras.api.-v1.keras.layers.CuDNNLSTM
  "Fast LSTM implementation backed by cuDNN.

  More information about cuDNN can be found on the [NVIDIA
  developer website](https://developer.nvidia.com/cudnn).
  Can only be run on GPU.

  Arguments:
      units: Positive integer, dimensionality of the output space.
      kernel_initializer: Initializer for the `kernel` weights matrix, used for
        the linear transformation of the inputs.
      unit_forget_bias: Boolean. If True, add 1 to the bias of the forget gate
        at initialization. Setting it to true will also force
        `bias_initializer=\"zeros\"`. This is recommended in [Jozefowicz et
        al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
      recurrent_initializer: Initializer for the `recurrent_kernel` weights
        matrix, used for the linear transformation of the recurrent state.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to the `kernel` weights
        matrix.
      recurrent_regularizer: Regularizer function applied to the
        `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to the output of the
        layer (its \"activation\").
      kernel_constraint: Constraint function applied to the `kernel` weights
        matrix.
      recurrent_constraint: Constraint function applied to the
        `recurrent_kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
      return_sequences: Boolean. Whether to return the last output. in the
        output sequence, or the full sequence.
      return_state: Boolean. Whether to return the last state in addition to the
        output.
      go_backwards: Boolean (default False). If True, process the input sequence
        backwards and return the reversed sequence.
      stateful: Boolean (default False). If True, the last state for each sample
        at index i in a batch will be used as initial state for the sample of
        index i in the following batch.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce layers (import-module "tensorflow.python.keras.api._v1.keras.layers"))

(defn CuDNNLSTM 
  "Fast LSTM implementation backed by cuDNN.

  More information about cuDNN can be found on the [NVIDIA
  developer website](https://developer.nvidia.com/cudnn).
  Can only be run on GPU.

  Arguments:
      units: Positive integer, dimensionality of the output space.
      kernel_initializer: Initializer for the `kernel` weights matrix, used for
        the linear transformation of the inputs.
      unit_forget_bias: Boolean. If True, add 1 to the bias of the forget gate
        at initialization. Setting it to true will also force
        `bias_initializer=\"zeros\"`. This is recommended in [Jozefowicz et
        al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
      recurrent_initializer: Initializer for the `recurrent_kernel` weights
        matrix, used for the linear transformation of the recurrent state.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to the `kernel` weights
        matrix.
      recurrent_regularizer: Regularizer function applied to the
        `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to the output of the
        layer (its \"activation\").
      kernel_constraint: Constraint function applied to the `kernel` weights
        matrix.
      recurrent_constraint: Constraint function applied to the
        `recurrent_kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
      return_sequences: Boolean. Whether to return the last output. in the
        output sequence, or the full sequence.
      return_state: Boolean. Whether to return the last state in addition to the
        output.
      go_backwards: Boolean (default False). If True, process the input sequence
        backwards and return the reversed sequence.
      stateful: Boolean (default False). If True, the last state for each sample
        at index i in a batch will be used as initial state for the sample of
        index i in the following batch.
  "
  [units & {:keys [kernel_initializer recurrent_initializer bias_initializer unit_forget_bias kernel_regularizer recurrent_regularizer bias_regularizer activity_regularizer kernel_constraint recurrent_constraint bias_constraint return_sequences return_state go_backwards stateful]
                       :or {kernel_regularizer None recurrent_regularizer None bias_regularizer None activity_regularizer None kernel_constraint None recurrent_constraint None bias_constraint None}} ]
    (py/call-attr-kw layers "CuDNNLSTM" [units] {:kernel_initializer kernel_initializer :recurrent_initializer recurrent_initializer :bias_initializer bias_initializer :unit_forget_bias unit_forget_bias :kernel_regularizer kernel_regularizer :recurrent_regularizer recurrent_regularizer :bias_regularizer bias_regularizer :activity_regularizer activity_regularizer :kernel_constraint kernel_constraint :recurrent_constraint recurrent_constraint :bias_constraint bias_constraint :return_sequences return_sequences :return_state return_state :go_backwards go_backwards :stateful stateful }))

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
  "Adds a new variable to the layer.

    Arguments:
      name: Variable name.
      shape: Variable shape. Defaults to scalar if unspecified.
      dtype: The type of the variable. Defaults to `self.dtype` or `float32`.
      initializer: Initializer instance (callable).
      regularizer: Regularizer instance (callable).
      trainable: Boolean, whether the variable should be part of the layer's
        \"trainable_variables\" (e.g. variables, biases)
        or \"non_trainable_variables\" (e.g. BatchNorm mean and variance).
        Note that `trainable` cannot be `True` if `synchronization`
        is set to `ON_READ`.
      constraint: Constraint instance (callable).
      partitioner: Partitioner to be passed to the `Trackable` API.
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
      **kwargs: Additional keyword arguments. Accepted values are `getter` and
        `collections`.

    Returns:
      The created variable. Usually either a `Variable` or `ResourceVariable`
      instance. If `partitioner` is not `None`, a `PartitionedVariable`
      instance is returned.

    Raises:
      RuntimeError: If called with partitioned variable regularization and
        eager execution is enabled.
      ValueError: When giving unsupported dtype and no initializer or when
        trainable has been set to True with synchronization set as `ON_READ`.
    "
  [self name shape dtype initializer regularizer trainable constraint partitioner use_resource  & {:keys [synchronization aggregation]} ]
    (py/call-attr-kw self "add_weight" [name shape dtype initializer regularizer trainable constraint partitioner use_resource] {:synchronization synchronization :aggregation aggregation }))

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
  ""
  [ self input_shape ]
  (py/call-attr self "build"  self input_shape ))

(defn call 
  ""
  [ self inputs mask training initial_state ]
  (py/call-attr self "call"  self inputs mask training initial_state ))

(defn cell 
  ""
  [ self ]
    (py/call-attr self "cell"))

(defn compute-mask 
  ""
  [ self inputs mask ]
  (py/call-attr self "compute_mask"  self inputs mask ))

(defn compute-output-shape 
  ""
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
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))

(defn get-initial-state 
  ""
  [ self inputs ]
  (py/call-attr self "get_initial_state"  self inputs ))

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
  ""
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
  ""
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
  ""
  [ self states ]
  (py/call-attr self "reset_states"  self states ))

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

(defn states 
  ""
  [ self ]
    (py/call-attr self "states"))

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
