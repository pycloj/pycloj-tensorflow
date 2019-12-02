(ns tensorflow.contrib.layers.GDN
  "Generalized divisive normalization layer.

  Based on the papers:

    \"Density Modeling of Images using a Generalized Normalization
    Transformation\"

    Johannes Ballé, Valero Laparra, Eero P. Simoncelli

    https://arxiv.org/abs/1511.06281

    \"End-to-end Optimized Image Compression\"

    Johannes Ballé, Valero Laparra, Eero P. Simoncelli

    https://arxiv.org/abs/1611.01704

  Implements an activation function that is essentially a multivariate
  generalization of a particular sigmoid-type function:

  ```
  y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
  ```

  where `i` and `j` run over channels. This implementation never sums across
  spatial dimensions. It is similar to local response normalization, but much
  more flexible, as `beta` and `gamma` are trainable parameters.

  Arguments:
    inverse: If `False` (default), compute GDN response. If `True`, compute IGDN
      response (one step of fixed point iteration to invert GDN; the division is
      replaced by multiplication).
    beta_min: Lower bound for beta, to prevent numerical error from causing
      square root of zero or negative values.
    gamma_init: The gamma matrix will be initialized as the identity matrix
      multiplied with this value. If set to zero, the layer is effectively
      initialized to the identity operation, since beta is initialized as one. A
      good default setting is somewhere between 0 and 0.5.
    reparam_offset: Offset added to the reparameterization of beta and gamma.
      The reparameterization of beta and gamma as their square roots lets the
      training slow down when their values are close to zero, which is desirable
      as small values in the denominator can lead to a situation where gradient
      noise on beta/gamma leads to extreme amounts of noise in the GDN
      activations. However, without the offset, we would get zero gradients if
      any elements of beta or gamma were exactly zero, and thus the training
      could get stuck. To prevent this, we add this small constant. The default
      value was empirically determined as a good starting point. Making it
      bigger potentially leads to more gradient noise on the activations, making
      it too small may lead to numerical precision issues.
    data_format: Format of input tensor. Currently supports `'channels_first'`
      and `'channels_last'`.
    activity_regularizer: Regularizer function for the output.
    trainable: Boolean, if `True`, also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: String, the name of the layer. Layers with the same name will share
      weights, but to avoid mistakes we require `reuse=True` in such cases.
  Properties:
    inverse: Boolean, whether GDN is computed (`True`) or IGDN (`False`).
    data_format: Format of input tensor. Currently supports `'channels_first'`
      and `'channels_last'`.
    beta: The beta parameter as defined above (1D `Tensor`).
    gamma: The gamma parameter as defined above (2D `Tensor`).
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce layers (import-module "tensorflow.contrib.layers"))

(defn GDN 
  "Generalized divisive normalization layer.

  Based on the papers:

    \"Density Modeling of Images using a Generalized Normalization
    Transformation\"

    Johannes Ballé, Valero Laparra, Eero P. Simoncelli

    https://arxiv.org/abs/1511.06281

    \"End-to-end Optimized Image Compression\"

    Johannes Ballé, Valero Laparra, Eero P. Simoncelli

    https://arxiv.org/abs/1611.01704

  Implements an activation function that is essentially a multivariate
  generalization of a particular sigmoid-type function:

  ```
  y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
  ```

  where `i` and `j` run over channels. This implementation never sums across
  spatial dimensions. It is similar to local response normalization, but much
  more flexible, as `beta` and `gamma` are trainable parameters.

  Arguments:
    inverse: If `False` (default), compute GDN response. If `True`, compute IGDN
      response (one step of fixed point iteration to invert GDN; the division is
      replaced by multiplication).
    beta_min: Lower bound for beta, to prevent numerical error from causing
      square root of zero or negative values.
    gamma_init: The gamma matrix will be initialized as the identity matrix
      multiplied with this value. If set to zero, the layer is effectively
      initialized to the identity operation, since beta is initialized as one. A
      good default setting is somewhere between 0 and 0.5.
    reparam_offset: Offset added to the reparameterization of beta and gamma.
      The reparameterization of beta and gamma as their square roots lets the
      training slow down when their values are close to zero, which is desirable
      as small values in the denominator can lead to a situation where gradient
      noise on beta/gamma leads to extreme amounts of noise in the GDN
      activations. However, without the offset, we would get zero gradients if
      any elements of beta or gamma were exactly zero, and thus the training
      could get stuck. To prevent this, we add this small constant. The default
      value was empirically determined as a good starting point. Making it
      bigger potentially leads to more gradient noise on the activations, making
      it too small may lead to numerical precision issues.
    data_format: Format of input tensor. Currently supports `'channels_first'`
      and `'channels_last'`.
    activity_regularizer: Regularizer function for the output.
    trainable: Boolean, if `True`, also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: String, the name of the layer. Layers with the same name will share
      weights, but to avoid mistakes we require `reuse=True` in such cases.
  Properties:
    inverse: Boolean, whether GDN is computed (`True`) or IGDN (`False`).
    data_format: Format of input tensor. Currently supports `'channels_first'`
      and `'channels_last'`.
    beta: The beta parameter as defined above (1D `Tensor`).
    gamma: The gamma parameter as defined above (2D `Tensor`).
  "
  [ & {:keys [inverse beta_min gamma_init reparam_offset data_format activity_regularizer trainable name]
       :or {activity_regularizer None name None}} ]
  
   (py/call-attr-kw layers "GDN" [] {:inverse inverse :beta_min beta_min :gamma_init gamma_init :reparam_offset reparam_offset :data_format data_format :activity_regularizer activity_regularizer :trainable trainable :name name }))

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
  ""
  [ self input_shape ]
  (py/call-attr self "build"  self input_shape ))

(defn call 
  ""
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
