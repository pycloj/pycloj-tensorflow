(ns tensorflow.-api.v1.summary
  "Operations for writing summary data, for use in analysis and visualization.

See the [Summaries and
TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard) guide.

"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce summary (import-module "tensorflow._api.v1.summary"))

(defn all-v2-summary-ops 
  "Returns all V2-style summary ops defined in the current default graph.

  This includes ops from TF 2.0 tf.summary and TF 1.x tf.contrib.summary (except
  for `tf.contrib.summary.graph` and `tf.contrib.summary.import_event`), but
  does *not* include TF 1.x tf.summary ops.

  Returns:
    List of summary ops, or None if called under eager execution.
  "
  [  ]
  (py/call-attr summary "all_v2_summary_ops"  ))

(defn audio 
  "Outputs a `Summary` protocol buffer with audio.

  The summary has up to `max_outputs` summary values containing audio. The
  audio is built from `tensor` which must be 3-D with shape `[batch_size,
  frames, channels]` or 2-D with shape `[batch_size, frames]`. The values are
  assumed to be in the range of `[-1.0, 1.0]` with a sample rate of
  `sample_rate`.

  The `tag` in the outputted Summary.Value protobufs is generated based on the
  name, with a suffix depending on the max_outputs setting:

  *  If `max_outputs` is 1, the summary value tag is '*name*/audio'.
  *  If `max_outputs` is greater than 1, the summary value tags are
     generated sequentially as '*name*/audio/0', '*name*/audio/1', etc

  Args:
    name: A name for the generated node. Will also serve as a series name in
      TensorBoard.
    tensor: A 3-D `float32` `Tensor` of shape `[batch_size, frames, channels]`
      or a 2-D `float32` `Tensor` of shape `[batch_size, frames]`.
    sample_rate: A Scalar `float32` `Tensor` indicating the sample rate of the
      signal in hertz.
    max_outputs: Max number of batch elements to generate audio for.
    collections: Optional list of ops.GraphKeys.  The collections to add the
      summary to.  Defaults to [_ops.GraphKeys.SUMMARIES]
    family: Optional; if provided, used as the prefix of the summary tag name,
      which controls the tab name used for display on Tensorboard.

  Returns:
    A scalar `Tensor` of type `string`. The serialized `Summary` protocol
    buffer.
  "
  [name tensor sample_rate & {:keys [max_outputs collections family]
                       :or {collections None family None}} ]
    (py/call-attr-kw summary "audio" [name tensor sample_rate] {:max_outputs max_outputs :collections collections :family family }))

(defn get-summary-description 
  "Given a TensorSummary node_def, retrieve its SummaryDescription.

  When a Summary op is instantiated, a SummaryDescription of associated
  metadata is stored in its NodeDef. This method retrieves the description.

  Args:
    node_def: the node_def_pb2.NodeDef of a TensorSummary op

  Returns:
    a summary_pb2.SummaryDescription

  Raises:
    ValueError: if the node is not a summary op.

  @compatibility(eager)
  Not compatible with eager execution. To write TensorBoard
  summaries under eager execution, use `tf.contrib.summary` instead.
  @end_compatibility
  "
  [ node_def ]
  (py/call-attr summary "get_summary_description"  node_def ))

(defn histogram 
  "Outputs a `Summary` protocol buffer with a histogram.

  Adding a histogram summary makes it possible to visualize your data's
  distribution in TensorBoard. You can see a detailed explanation of the
  TensorBoard histogram dashboard
  [here](https://www.tensorflow.org/get_started/tensorboard_histograms).

  The generated
  [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
  has one summary value containing a histogram for `values`.

  This op reports an `InvalidArgument` error if any value is not finite.

  Args:
    name: A name for the generated node. Will also serve as a series name in
      TensorBoard.
    values: A real numeric `Tensor`. Any shape. Values to use to
      build the histogram.
    collections: Optional list of graph collections keys. The new summary op is
      added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
    family: Optional; if provided, used as the prefix of the summary tag name,
      which controls the tab name used for display on Tensorboard.

  Returns:
    A scalar `Tensor` of type `string`. The serialized `Summary` protocol
    buffer.
  "
  [ name values collections family ]
  (py/call-attr summary "histogram"  name values collections family ))

(defn image 
  "Outputs a `Summary` protocol buffer with images.

  The summary has up to `max_outputs` summary values containing images. The
  images are built from `tensor` which must be 4-D with shape `[batch_size,
  height, width, channels]` and where `channels` can be:

  *  1: `tensor` is interpreted as Grayscale.
  *  3: `tensor` is interpreted as RGB.
  *  4: `tensor` is interpreted as RGBA.

  The images have the same number of channels as the input tensor. For float
  input, the values are normalized one image at a time to fit in the range
  `[0, 255]`.  `uint8` values are unchanged.  The op uses two different
  normalization algorithms:

  *  If the input values are all positive, they are rescaled so the largest one
     is 255.

  *  If any input value is negative, the values are shifted so input value 0.0
     is at 127.  They are then rescaled so that either the smallest value is 0,
     or the largest one is 255.

  The `tag` in the outputted Summary.Value protobufs is generated based on the
  name, with a suffix depending on the max_outputs setting:

  *  If `max_outputs` is 1, the summary value tag is '*name*/image'.
  *  If `max_outputs` is greater than 1, the summary value tags are
     generated sequentially as '*name*/image/0', '*name*/image/1', etc.

  Args:
    name: A name for the generated node. Will also serve as a series name in
      TensorBoard.
    tensor: A 4-D `uint8` or `float32` `Tensor` of shape `[batch_size, height,
      width, channels]` where `channels` is 1, 3, or 4.
    max_outputs: Max number of batch elements to generate images for.
    collections: Optional list of ops.GraphKeys.  The collections to add the
      summary to.  Defaults to [_ops.GraphKeys.SUMMARIES]
    family: Optional; if provided, used as the prefix of the summary tag name,
      which controls the tab name used for display on Tensorboard.

  Returns:
    A scalar `Tensor` of type `string`. The serialized `Summary` protocol
    buffer.
  "
  [name tensor & {:keys [max_outputs collections family]
                       :or {collections None family None}} ]
    (py/call-attr-kw summary "image" [name tensor] {:max_outputs max_outputs :collections collections :family family }))

(defn initialize 
  "Initializes summary writing for graph execution mode.

  This operation is a no-op when executing eagerly.

  This helper method provides a higher-level alternative to using
  `tf.contrib.summary.summary_writer_initializer_op` and
  `tf.contrib.summary.graph`.

  Most users will also want to call `tf.compat.v1.train.create_global_step`
  which can happen before or after this function is called.

  Args:
    graph: A `tf.Graph` or `tf.compat.v1.GraphDef` to output to the writer.
      This function will not write the default graph by default. When
      writing to an event log file, the associated step will be zero.
    session: So this method can call `tf.Session.run`. This defaults
      to `tf.compat.v1.get_default_session`.

  Raises:
    RuntimeError: If  the current thread has no default
      `tf.contrib.summary.SummaryWriter`.
    ValueError: If session wasn't passed and no default session.
  "
  [ graph session ]
  (py/call-attr summary "initialize"  graph session ))

(defn merge 
  "Merges summaries.

  This op creates a
  [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
  protocol buffer that contains the union of all the values in the input
  summaries.

  When the Op is run, it reports an `InvalidArgument` error if multiple values
  in the summaries to merge use the same tag.

  Args:
    inputs: A list of `string` `Tensor` objects containing serialized `Summary`
      protocol buffers.
    collections: Optional list of graph collections keys. The new summary op is
      added to these collections. Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A scalar `Tensor` of type `string`. The serialized `Summary` protocol
    buffer resulting from the merging.

  Raises:
    RuntimeError: If called with eager mode enabled.

  @compatibility(eager)
  Not compatible with eager execution. To write TensorBoard
  summaries under eager execution, use `tf.contrib.summary` instead.
  @end_compatibility
  "
  [ inputs collections name ]
  (py/call-attr summary "merge"  inputs collections name ))

(defn merge-all 
  "Merges all summaries collected in the default graph.

  Args:
    key: `GraphKey` used to collect the summaries.  Defaults to
      `GraphKeys.SUMMARIES`.
    scope: Optional scope used to filter the summary ops, using `re.match`

  Returns:
    If no summaries were collected, returns None.  Otherwise returns a scalar
    `Tensor` of type `string` containing the serialized `Summary` protocol
    buffer resulting from the merging.

  Raises:
    RuntimeError: If called with eager execution enabled.

  @compatibility(eager)
  Not compatible with eager execution. To write TensorBoard
  summaries under eager execution, use `tf.contrib.summary` instead.
  @end_compatibility
  "
  [ & {:keys [key scope name]
       :or {scope None name None}} ]
  
   (py/call-attr-kw summary "merge_all" [] {:key key :scope scope :name name }))

(defn scalar 
  "Outputs a `Summary` protocol buffer containing a single scalar value.

  The generated Summary has a Tensor.proto containing the input Tensor.

  Args:
    name: A name for the generated node. Will also serve as the series name in
      TensorBoard.
    tensor: A real numeric Tensor containing a single value.
    collections: Optional list of graph collections keys. The new summary op is
      added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
    family: Optional; if provided, used as the prefix of the summary tag name,
      which controls the tab name used for display on Tensorboard.

  Returns:
    A scalar `Tensor` of type `string`. Which contains a `Summary` protobuf.

  Raises:
    ValueError: If tensor has the wrong shape or type.
  "
  [ name tensor collections family ]
  (py/call-attr summary "scalar"  name tensor collections family ))

(defn tensor-summary 
  "Outputs a `Summary` protocol buffer with a serialized tensor.proto.

  Args:
    name: A name for the generated node. If display_name is not set, it will
      also serve as the tag name in TensorBoard. (In that case, the tag
      name will inherit tf name scopes.)
    tensor: A tensor of any type and shape to serialize.
    summary_description: A long description of the summary sequence. Markdown
      is supported.
    collections: Optional list of graph collections keys. The new summary op is
      added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
    summary_metadata: Optional SummaryMetadata proto (which describes which
      plugins may use the summary value).
    family: Optional; if provided, used as the prefix of the summary tag,
      which controls the name used for display on TensorBoard when
      display_name is not set.
    display_name: A string used to name this data in TensorBoard. If this is
      not set, then the node name will be used instead.

  Returns:
    A scalar `Tensor` of type `string`. The serialized `Summary` protocol
    buffer.
  "
  [ name tensor summary_description collections summary_metadata family display_name ]
  (py/call-attr summary "tensor_summary"  name tensor summary_description collections summary_metadata family display_name ))

(defn text 
  "Summarizes textual data.

  Text data summarized via this plugin will be visible in the Text Dashboard
  in TensorBoard. The standard TensorBoard Text Dashboard will render markdown
  in the strings, and will automatically organize 1d and 2d tensors into tables.
  If a tensor with more than 2 dimensions is provided, a 2d subarray will be
  displayed along with a warning message. (Note that this behavior is not
  intrinsic to the text summary api, but rather to the default TensorBoard text
  plugin.)

  Args:
    name: A name for the generated node. Will also serve as a series name in
      TensorBoard.
    tensor: a string-type Tensor to summarize.
    collections: Optional list of ops.GraphKeys.  The collections to add the
      summary to.  Defaults to [_ops.GraphKeys.SUMMARIES]

  Returns:
    A TensorSummary op that is configured so that TensorBoard will recognize
    that it contains textual data. The TensorSummary is a scalar `Tensor` of
    type `string` which contains `Summary` protobufs.

  Raises:
    ValueError: If tensor has the wrong type.
  "
  [ name tensor collections ]
  (py/call-attr summary "text"  name tensor collections ))
