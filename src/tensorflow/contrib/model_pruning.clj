(ns tensorflow.contrib.model-pruning
  "Model pruning implementation in tensorflow."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce model-pruning (import-module "tensorflow.contrib.model_pruning"))
(defn apply-mask 
  "Apply mask to a given weight tensor.

  Args:
    x: Input weight tensor
    scope: The current variable scope. Defaults to \"\".
  Returns:
    Tensor representing masked_weights
  "
  [x  & {:keys [scope]} ]
    (py/call-attr-kw model-pruning "apply_mask" [x] {:scope scope }))

(defn get-masked-weights 
  ""
  [  ]
  (py/call-attr model-pruning "get_masked_weights"  ))

(defn get-masks 
  ""
  [  ]
  (py/call-attr model-pruning "get_masks"  ))

(defn get-pruning-hparams 
  "Get a tf.HParams object with the default values for the hyperparameters.

    name: string
      name of the pruning specification. Used for adding summaries and ops under
      a common tensorflow name_scope
    begin_pruning_step: integer
      the global step at which to begin pruning
    end_pruning_step: integer
      the global step at which to terminate pruning. Defaults to -1 implying
      that pruning continues till the training stops
    weight_sparsity_map: list of strings
       comma separed list of {weight_variable_name:target sparsity} or
       {regex:target sparsity} pairs.
       For layers/weights not in this list, sparsity as specified by the
       target_sparsity hyperparameter is used.
       Eg. [conv1:0.9,conv2/kernel:0.8]
    block_dims_map: list of strings
       comma separated list of {weight variable name:block_height x block_width}
       or {regex:block_height x block_width} pairs. For layers/weights not in
       this list, block dims are specified by the block_height, block_width
       hyperparameters are used Eg. [dense1:4x4,dense2:1x16,dense3:1x1]
    threshold_decay: float
      the decay factor to use for exponential decay of the thresholds
    pruning_frequency: integer
      How often should the masks be updated? (in # of global_steps)
    nbins: integer
      number of bins to use for histogram computation
    block_height: integer
      number of rows in a block (defaults to 1), can be -1 in which
      case it is set to the size of the corresponding weight tensor.
    block_width: integer
      number of cols in a block (defaults to 1), can be -1 in which
      case it is set to the size of the corresponding weight tensor.
    block_pooling_function: string
      Whether to perform average (AVG) or max (MAX) pooling in the block
      (default: AVG)
    initial_sparsity: float
      initial sparsity value
    target_sparsity: float
      target sparsity value
    sparsity_function_begin_step: integer
      the global step at this which the gradual sparsity function begins to
      take effect
    sparsity_function_end_step: integer
      the global step used as the end point for the gradual sparsity function
    sparsity_function_exponent: float
      exponent = 1 is linearly varying sparsity between initial and final.
      exponent > 1 varies more slowly towards the end than the beginning
    use_tpu: False
      Indicates whether to use TPU

    We use the following sparsity function:

    num_steps = (sparsity_function_end_step -
                 sparsity_function_begin_step)/pruning_frequency
    sparsity(step) = (initial_sparsity - target_sparsity)*
                     [1-step/(num_steps -1)]**exponent + target_sparsity

  Args:
    None

  Returns:
    tf.HParams object initialized to default values

  "
  [  ]
  (py/call-attr model-pruning "get_pruning_hparams"  ))

(defn get-thresholds 
  ""
  [  ]
  (py/call-attr model-pruning "get_thresholds"  ))

(defn get-weight-sparsity 
  "Get sparsity of the weights.

  Args:
    None

  Returns:
    A list containing the sparsity of each of the weight tensors
  "
  [  ]
  (py/call-attr model-pruning "get_weight_sparsity"  ))

(defn get-weights 
  ""
  [  ]
  (py/call-attr model-pruning "get_weights"  ))

(defn graph-def-from-checkpoint 
  "Converts checkpoint data to GraphDef.

  Reads the latest checkpoint data and produces a GraphDef in which the
  variables have been converted to constants.

  Args:
    checkpoint_dir: Path to the checkpoints.
    output_node_names: List of name strings for the result nodes of the graph.

  Returns:
    A GraphDef from the latest checkpoint

  Raises:
    ValueError: if no checkpoint is found
  "
  [ checkpoint_dir output_node_names ]
  (py/call-attr model-pruning "graph_def_from_checkpoint"  checkpoint_dir output_node_names ))

(defn masked-conv2d 
  "Adds an 2D convolution followed by an optional batch_norm layer.
  The layer creates a mask variable on top of the weight variable. The input to
  the convolution operation is the elementwise multiplication of the mask
  variable and the weigh

  It is required that 1 <= N <= 3.

  `convolution` creates a variable called `weights`, representing the
  convolutional kernel, that is convolved (actually cross-correlated) with the
  `inputs` to produce a `Tensor` of activations. If a `normalizer_fn` is
  provided (such as `batch_norm`), it is then applied. Otherwise, if
  `normalizer_fn` is None and a `biases_initializer` is provided then a `biases`
  variable would be created and added the activations. Finally, if
  `activation_fn` is not `None`, it is applied to the activations as well.

  Performs atrous convolution with input stride/dilation rate equal to `rate`
  if a value > 1 for any dimension of `rate` is specified.  In this case
  `stride` values != 1 are not supported.

  Args:
    inputs: A Tensor of rank N+2 of shape
      `[batch_size] + input_spatial_shape + [in_channels]` if data_format does
      not start with \"NC\" (default), or
      `[batch_size, in_channels] + input_spatial_shape` if data_format starts
      with \"NC\".
    num_outputs: Integer, the number of output filters.
    kernel_size: A sequence of N positive integers specifying the spatial
      dimensions of the filters.  Can be a single integer to specify the same
      value for all spatial dimensions.
    stride: A sequence of N positive integers specifying the stride at which to
      compute output.  Can be a single integer to specify the same value for all
      spatial dimensions.  Specifying any `stride` value != 1 is incompatible
      with specifying any `rate` value != 1.
    padding: One of `\"VALID\"` or `\"SAME\"`.
    data_format: A string or None.  Specifies whether the channel dimension of
      the `input` and output is the last dimension (default, or if `data_format`
      does not start with \"NC\"), or the second dimension (if `data_format`
      starts with \"NC\").  For N=1, the valid values are \"NWC\" (default) and
      \"NCW\".  For N=2, the valid values are \"NHWC\" (default) and \"NCHW\".
      For N=3, the valid values are \"NDHWC\" (default) and \"NCDHW\".
    rate: A sequence of N positive integers specifying the dilation rate to use
      for atrous convolution.  Can be a single integer to specify the same
      value for all spatial dimensions.  Specifying any `rate` value != 1 is
      incompatible with specifying any `stride` value != 1.
    activation_fn: Activation function. The default value is a ReLU function.
      Explicitly set it to None to skip it and maintain a linear activation.
    normalizer_fn: Normalization function to use instead of `biases`. If
      `normalizer_fn` is provided then `biases_initializer` and
      `biases_regularizer` are ignored and `biases` are not created nor added.
      default set to None for no normalizer function
    normalizer_params: Normalization function parameters.
    weights_initializer: An initializer for the weights.
    weights_regularizer: Optional regularizer for the weights.
    biases_initializer: An initializer for the biases. If None skip biases.
    biases_regularizer: Optional regularizer for the biases.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional list of collections for all the variables or
      a dictionary containing a different list of collection per variable.
    outputs_collections: Collection to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    scope: Optional scope for `variable_scope`.

  Returns:
    A tensor representing the output of the operation.

  Raises:
    ValueError: If `data_format` is invalid.
    ValueError: Both 'rate' and `stride` are not uniformly 1.
  "
  [inputs num_outputs kernel_size & {:keys [stride padding data_format rate activation_fn normalizer_fn normalizer_params weights_initializer weights_regularizer biases_initializer biases_regularizer reuse variables_collections outputs_collections trainable scope]
                       :or {data_format None normalizer_fn None normalizer_params None weights_regularizer None biases_regularizer None reuse None variables_collections None outputs_collections None scope None}} ]
    (py/call-attr-kw model-pruning "masked_conv2d" [inputs num_outputs kernel_size] {:stride stride :padding padding :data_format data_format :rate rate :activation_fn activation_fn :normalizer_fn normalizer_fn :normalizer_params normalizer_params :weights_initializer weights_initializer :weights_regularizer weights_regularizer :biases_initializer biases_initializer :biases_regularizer biases_regularizer :reuse reuse :variables_collections variables_collections :outputs_collections outputs_collections :trainable trainable :scope scope }))

(defn masked-convolution 
  "Adds an 2D convolution followed by an optional batch_norm layer.
  The layer creates a mask variable on top of the weight variable. The input to
  the convolution operation is the elementwise multiplication of the mask
  variable and the weigh

  It is required that 1 <= N <= 3.

  `convolution` creates a variable called `weights`, representing the
  convolutional kernel, that is convolved (actually cross-correlated) with the
  `inputs` to produce a `Tensor` of activations. If a `normalizer_fn` is
  provided (such as `batch_norm`), it is then applied. Otherwise, if
  `normalizer_fn` is None and a `biases_initializer` is provided then a `biases`
  variable would be created and added the activations. Finally, if
  `activation_fn` is not `None`, it is applied to the activations as well.

  Performs atrous convolution with input stride/dilation rate equal to `rate`
  if a value > 1 for any dimension of `rate` is specified.  In this case
  `stride` values != 1 are not supported.

  Args:
    inputs: A Tensor of rank N+2 of shape
      `[batch_size] + input_spatial_shape + [in_channels]` if data_format does
      not start with \"NC\" (default), or
      `[batch_size, in_channels] + input_spatial_shape` if data_format starts
      with \"NC\".
    num_outputs: Integer, the number of output filters.
    kernel_size: A sequence of N positive integers specifying the spatial
      dimensions of the filters.  Can be a single integer to specify the same
      value for all spatial dimensions.
    stride: A sequence of N positive integers specifying the stride at which to
      compute output.  Can be a single integer to specify the same value for all
      spatial dimensions.  Specifying any `stride` value != 1 is incompatible
      with specifying any `rate` value != 1.
    padding: One of `\"VALID\"` or `\"SAME\"`.
    data_format: A string or None.  Specifies whether the channel dimension of
      the `input` and output is the last dimension (default, or if `data_format`
      does not start with \"NC\"), or the second dimension (if `data_format`
      starts with \"NC\").  For N=1, the valid values are \"NWC\" (default) and
      \"NCW\".  For N=2, the valid values are \"NHWC\" (default) and \"NCHW\".
      For N=3, the valid values are \"NDHWC\" (default) and \"NCDHW\".
    rate: A sequence of N positive integers specifying the dilation rate to use
      for atrous convolution.  Can be a single integer to specify the same
      value for all spatial dimensions.  Specifying any `rate` value != 1 is
      incompatible with specifying any `stride` value != 1.
    activation_fn: Activation function. The default value is a ReLU function.
      Explicitly set it to None to skip it and maintain a linear activation.
    normalizer_fn: Normalization function to use instead of `biases`. If
      `normalizer_fn` is provided then `biases_initializer` and
      `biases_regularizer` are ignored and `biases` are not created nor added.
      default set to None for no normalizer function
    normalizer_params: Normalization function parameters.
    weights_initializer: An initializer for the weights.
    weights_regularizer: Optional regularizer for the weights.
    biases_initializer: An initializer for the biases. If None skip biases.
    biases_regularizer: Optional regularizer for the biases.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional list of collections for all the variables or
      a dictionary containing a different list of collection per variable.
    outputs_collections: Collection to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    scope: Optional scope for `variable_scope`.

  Returns:
    A tensor representing the output of the operation.

  Raises:
    ValueError: If `data_format` is invalid.
    ValueError: Both 'rate' and `stride` are not uniformly 1.
  "
  [inputs num_outputs kernel_size & {:keys [stride padding data_format rate activation_fn normalizer_fn normalizer_params weights_initializer weights_regularizer biases_initializer biases_regularizer reuse variables_collections outputs_collections trainable scope]
                       :or {data_format None normalizer_fn None normalizer_params None weights_regularizer None biases_regularizer None reuse None variables_collections None outputs_collections None scope None}} ]
    (py/call-attr-kw model-pruning "masked_convolution" [inputs num_outputs kernel_size] {:stride stride :padding padding :data_format data_format :rate rate :activation_fn activation_fn :normalizer_fn normalizer_fn :normalizer_params normalizer_params :weights_initializer weights_initializer :weights_regularizer weights_regularizer :biases_initializer biases_initializer :biases_regularizer biases_regularizer :reuse reuse :variables_collections variables_collections :outputs_collections outputs_collections :trainable trainable :scope scope }))

(defn masked-fully-connected 
  "Adds a sparse fully connected layer. The weight matrix is masked.

  `fully_connected` creates a variable called `weights`, representing a fully
  connected weight matrix, which is multiplied by the `inputs` to produce a
  `Tensor` of hidden units. If a `normalizer_fn` is provided (such as
  `batch_norm`), it is then applied. Otherwise, if `normalizer_fn` is
  None and a `biases_initializer` is provided then a `biases` variable would be
  created and added the hidden units. Finally, if `activation_fn` is not `None`,
  it is applied to the hidden units as well.

  Note: that if `inputs` have a rank greater than 2, then `inputs` is flattened
  prior to the initial matrix multiply by `weights`.

  Args:
    inputs: A tensor of at least rank 2 and static value for the last dimension;
      i.e. `[batch_size, depth]`, `[None, None, None, channels]`.
    num_outputs: Integer or long, the number of output units in the layer.
    activation_fn: Activation function. The default value is a ReLU function.
      Explicitly set it to None to skip it and maintain a linear activation.
    normalizer_fn: Normalization function to use instead of `biases`. If
      `normalizer_fn` is provided then `biases_initializer` and
      `biases_regularizer` are ignored and `biases` are not created nor added.
      default set to None for no normalizer function
    normalizer_params: Normalization function parameters.
    weights_initializer: An initializer for the weights.
    weights_regularizer: Optional regularizer for the weights.
    biases_initializer: An initializer for the biases. If None skip biases.
    biases_regularizer: Optional regularizer for the biases.
    reuse: Whether or not the layer and its variables should be reused. To be
      able to reuse the layer scope must be given.
    variables_collections: Optional list of collections for all the variables or
      a dictionary containing a different list of collections per variable.
    outputs_collections: Collection to add the outputs.
    trainable: If `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    scope: Optional scope for variable_scope.

  Returns:
     The tensor variable representing the result of the series of operations.

  Raises:
    ValueError: If x has rank less than 2 or if its last dimension is not set.
  "
  [inputs num_outputs & {:keys [activation_fn normalizer_fn normalizer_params weights_initializer weights_regularizer biases_initializer biases_regularizer reuse variables_collections outputs_collections trainable scope]
                       :or {normalizer_fn None normalizer_params None weights_regularizer None biases_regularizer None reuse None variables_collections None outputs_collections None scope None}} ]
    (py/call-attr-kw model-pruning "masked_fully_connected" [inputs num_outputs] {:activation_fn activation_fn :normalizer_fn normalizer_fn :normalizer_params normalizer_params :weights_initializer weights_initializer :weights_regularizer weights_regularizer :biases_initializer biases_initializer :biases_regularizer biases_regularizer :reuse reuse :variables_collections variables_collections :outputs_collections outputs_collections :trainable trainable :scope scope }))

(defn strip-pruning-vars-fn 
  "Removes mask variable from the graph.

  Replaces the masked_weight tensor with element-wise multiplication of mask
  and the corresponding weight variable.

  Args:
    input_graph_def: A GraphDef in which the variables have been converted to
      constants. This is typically the output of
      tf.graph_util.convert_variables_to_constant()
    output_node_names: List of name strings for the result nodes of the graph

  Returns:
    A GraphDef in which pruning-related variables have been removed
  "
  [ input_graph_def output_node_names ]
  (py/call-attr model-pruning "strip_pruning_vars_fn"  input_graph_def output_node_names ))

(defn train 
  "Wrapper around tf-slim's train function.

  Runs a training loop using a TensorFlow supervisor.
  When the sync_optimizer is supplied, gradient updates are applied
  synchronously. Otherwise, gradient updates are applied asynchronous.

  Args:
    train_op: A `Tensor` that, when executed, will apply the gradients and
      return the loss value.
    logdir: The directory where training logs are written to. If None, model
      checkpoints and summaries will not be written.
    mask_update_op: Operation that upon execution updates the weight masks and
      thresholds.
    train_step_fn: The function to call in order to execute a single gradient
      step. The function must have take exactly four arguments: the current
        session, the `train_op` `Tensor`, a global step `Tensor` and a
        dictionary.
    train_step_kwargs: A dictionary which is passed to the `train_step_fn`. By
      default, two `Boolean`, scalar ops called \"should_stop\" and \"should_log\"
      are provided.
    log_every_n_steps: The frequency, in terms of global steps, that the loss
      and global step and logged.
    graph: The graph to pass to the supervisor. If no graph is supplied the
      default graph is used.
    master: The address of the tensorflow master.
    is_chief: Specifies whether or not the training is being run by the primary
      replica during replica training.
    global_step: The `Tensor` representing the global step. If left as `None`,
      then slim.variables.get_or_create_global_step() is used.
    number_of_steps: The max number of gradient steps to take during training,
      as measured by 'global_step': training will stop if global_step is greater
        than 'number_of_steps'. If the value is left as None, training proceeds
        indefinitely.
    init_op: The initialization operation. If left to its default value, then
      the session is initialized by calling
      `tf.compat.v1.global_variables_initializer()`.
    init_feed_dict: A feed dictionary to use when executing the `init_op`.
    local_init_op: The local initialization operation. If left to its default
      value, then the session is initialized by calling
      `tf.compat.v1.local_variables_initializer()` and
      `tf.compat.v1.tables_initializer()`.
    init_fn: An optional callable to be executed after `init_op` is called. The
      callable must accept one argument, the session being initialized.
    ready_op: Operation to check if the model is ready to use. If left to its
      default value, then the session checks for readiness by calling
      `tf.compat.v1.report_uninitialized_variables()`.
    summary_op: The summary operation.
    save_summaries_secs: How often, in seconds, to save summaries.
    summary_writer: `SummaryWriter` to use.  Can be `None` to indicate that no
      summaries should be written. If unset, we create a SummaryWriter.
    startup_delay_steps: The number of steps to wait for before beginning. Note
      that this must be 0 if a sync_optimizer is supplied.
    saver: Saver to save checkpoints. If None, a default one will be created and
      used.
    save_interval_secs: How often, in seconds, to save the model to `logdir`.
    sync_optimizer: an instance of tf.compat.v1.train.SyncReplicasOptimizer, or
      a list of them. If the argument is supplied, gradient updates will be
      synchronous. If left as `None`, gradient updates will be asynchronous.
    session_config: An instance of `tf.compat.v1.ConfigProto` that will be used
      to configure the `Session`. If left as `None`, the default will be used.
    trace_every_n_steps: produce and save a `Timeline` in Chrome trace format
      and add it to the summaries every `trace_every_n_steps`. If None, no trace
      information will be produced or saved.

  Returns:
    the value of the loss function after training.

  Raises:
    ValueError: if `train_op` is empty or if `startup_delay_steps` is
      non-zero when `sync_optimizer` is supplied, if `number_of_steps` is
      negative, or if `trace_every_n_steps` is not `None` and no `logdir` is
      provided.
  "
  [train_op logdir mask_update_op & {:keys [train_step_fn train_step_kwargs log_every_n_steps graph master is_chief global_step number_of_steps init_op init_feed_dict local_init_op init_fn ready_op summary_op save_summaries_secs summary_writer startup_delay_steps saver save_interval_secs sync_optimizer session_config trace_every_n_steps]
                       :or {graph None global_step None number_of_steps None init_feed_dict None init_fn None saver None sync_optimizer None session_config None trace_every_n_steps None}} ]
    (py/call-attr-kw model-pruning "train" [train_op logdir mask_update_op] {:train_step_fn train_step_fn :train_step_kwargs train_step_kwargs :log_every_n_steps log_every_n_steps :graph graph :master master :is_chief is_chief :global_step global_step :number_of_steps number_of_steps :init_op init_op :init_feed_dict init_feed_dict :local_init_op local_init_op :init_fn init_fn :ready_op ready_op :summary_op summary_op :save_summaries_secs save_summaries_secs :summary_writer summary_writer :startup_delay_steps startup_delay_steps :saver saver :save_interval_secs save_interval_secs :sync_optimizer sync_optimizer :session_config session_config :trace_every_n_steps trace_every_n_steps }))
