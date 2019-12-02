(ns tensorflow-core.contrib.deprecated
  "Non-core alias for the deprecated tf.X_summary ops.

For TensorFlow 1.0, we have reorganized the TensorFlow summary ops into a
submodule, and made some semantic tweaks. The first thing to note is that we
moved the APIs around as follows:

```python
tf.scalar_summary -> tf.compat.v1.summary.scalar
tf.histogram_summary -> tf.compat.v1.summary.histogram
tf.audio_summary -> tf.compat.v1.summary.audio
tf.image_summary -> tf.compat.v1.summary.image
tf.merge_summary -> tf.compat.v1.summary.merge
tf.merge_all_summaries -> tf.compat.v1.summary.merge_all
```

We think this API is cleaner and will improve long-term discoverability and
clarity of the TensorFlow API. But we also took the opportunity to make an
important change to how summary \"tags\" work. The \"tag\" of a summary is the
string that is associated with the output data, i.e. the key for organizing the
generated protobufs.

Previously, the tag was allowed to be any unique string; it had no relation
to the summary op generating it, and no relation to the TensorFlow name system.
This behavior made it very difficult to write reusable  that would add
summary ops to the graph. If you had a function to add summary ops, you would
need to pass in a `tf.name_scope`, manually, to that function to create
deduplicated tags. Otherwise your program would fail with a runtime error due
to tag collision.

The new summary APIs under `tf.summary` throw away the \"tag\" as an independent
concept; instead, the first argument is the node name. So summary tags now
automatically inherit the surrounding `tf.name_scope`, and automatically
are deduplicated if there is a conflict. Now however, the only allowed
characters are alphanumerics, underscores, and forward slashes. To make
migration easier, the new APIs automatically convert illegal characters to
underscores.

Just as an example, consider the following \"before\" and \"after\" code snippets:

```python
# Before
def add_activation_summaries(v, scope):
  tf.scalar_summary(\"%s/fraction_of_zero\" % scope, tf.nn.fraction_of_zero(v))
  tf.histogram_summary(\"%s/activations\" % scope, v)

# After
def add_activation_summaries(v):
  tf.compat.v1.summary.scalar(\"fraction_of_zero\", tf.nn.fraction_of_zero(v))
  tf.compat.v1.summary.histogram(\"activations\", v)
```

Now, so long as the add_activation_summaries function is called from within the
right `tf.name_scope`, the behavior is the same.

Because this change does modify the behavior and could break tests, we can't
automatically migrate usage to the new APIs. That is why we are making the old
APIs temporarily available here at `tf.contrib.deprecated`.

In addition to the name change described above, there are two further changes
to the new summary ops:

- the \"max_images\" argument for `tf.image_summary` was renamed to \"max_outputs
  for `tf.compat.v1.summary.image`
- `tf.scalar_summary` accepted arbitrary tensors of tags and values. But
  `tf.compat.v1.summary.scalar` requires a single scalar name and scalar value.
  In most
  cases, you can create `tf.compat.v1.summary.scalar` in a loop to get the same
  behavior

As before, TensorBoard groups charts by the top-level `tf.name_scope` which may
be inconvenient, for in the new summary ops, the summary will inherit that
`tf.name_scope` without user control. We plan to add more grouping mechanisms
to TensorBoard, so it will be possible to specify the TensorBoard group for
each summary via the summary API.

"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce deprecated (import-module "tensorflow_core.contrib.deprecated"))

(defn audio-summary 
  "Outputs a `Summary` protocol buffer with audio. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2016-11-30.
Instructions for updating:
Please switch to tf.summary.audio. Note that tf.summary.audio uses the node name instead of the tag. This means that TensorFlow will automatically de-duplicate summary names based on the scope they are created in.

This op is deprecated. Please switch to tf.summary.audio.
For an explanation of why this op was deprecated, and information on how to
migrate, look
['here'](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/deprecated/__init__.py)

The summary has up to `max_outputs` summary values containing audio. The
audio is built from `tensor` which must be 3-D with shape `[batch_size,
frames, channels]` or 2-D with shape `[batch_size, frames]`. The values are
assumed to be in the range of `[-1.0, 1.0]` with a sample rate of
`sample_rate`.

The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
build the `tag` of the summary values:

*  If `max_outputs` is 1, the summary value tag is '*tag*/audio'.
*  If `max_outputs` is greater than 1, the summary value tags are
   generated sequentially as '*tag*/audio/0', '*tag*/audio/1', etc.

Args:
  tag: A scalar `Tensor` of type `string`. Used to build the `tag` of the
    summary values.
  tensor: A 3-D `float32` `Tensor` of shape `[batch_size, frames, channels]`
    or a 2-D `float32` `Tensor` of shape `[batch_size, frames]`.
  sample_rate: A Scalar `float32` `Tensor` indicating the sample rate of the
    signal in hertz.
  max_outputs: Max number of batch elements to generate audio for.
  collections: Optional list of ops.GraphKeys.  The collections to add the
    summary to.  Defaults to [ops.GraphKeys.SUMMARIES]
  name: A name for the operation (optional).

Returns:
  A scalar `Tensor` of type `string`. The serialized `Summary` protocol
  buffer."
  [tag tensor sample_rate & {:keys [max_outputs collections name]
                       :or {collections None name None}} ]
    (py/call-attr-kw deprecated "audio_summary" [tag tensor sample_rate] {:max_outputs max_outputs :collections collections :name name }))

(defn histogram-summary 
  "Outputs a `Summary` protocol buffer with a histogram. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2016-11-30.
Instructions for updating:
Please switch to tf.summary.histogram. Note that tf.summary.histogram uses the node name instead of the tag. This means that TensorFlow will automatically de-duplicate summary names based on the scope they are created in.

This ops is deprecated. Please switch to tf.summary.histogram.

For an explanation of why this op was deprecated, and information on how to
migrate, look
['here'](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/deprecated/__init__.py)

The generated
[`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
has one summary value containing a histogram for `values`.

This op reports an `InvalidArgument` error if any value is not finite.

Args:
  tag: A `string` `Tensor`. 0-D.  Tag to use for the summary value.
  values: A real numeric `Tensor`. Any shape. Values to use to build the
    histogram.
  collections: Optional list of graph collections keys. The new summary op is
    added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
  name: A name for the operation (optional).

Returns:
  A scalar `Tensor` of type `string`. The serialized `Summary` protocol
  buffer."
  [ tag values collections name ]
  (py/call-attr deprecated "histogram_summary"  tag values collections name ))

(defn image-summary 
  "Outputs a `Summary` protocol buffer with images. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2016-11-30.
Instructions for updating:
Please switch to tf.summary.image. Note that tf.summary.image uses the node name instead of the tag. This means that TensorFlow will automatically de-duplicate summary names based on the scope they are created in. Also, the max_images argument was renamed to max_outputs.

For an explanation of why this op was deprecated, and information on how to
migrate, look
['here'](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/deprecated/__init__.py)

The summary has up to `max_images` summary values containing images. The
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

The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
build the `tag` of the summary values:

*  If `max_images` is 1, the summary value tag is '*tag*/image'.
*  If `max_images` is greater than 1, the summary value tags are
   generated sequentially as '*tag*/image/0', '*tag*/image/1', etc.

Args:
  tag: A scalar `Tensor` of type `string`. Used to build the `tag` of the
    summary values.
  tensor: A 4-D `uint8` or `float32` `Tensor` of shape `[batch_size, height,
    width, channels]` where `channels` is 1, 3, or 4.
  max_images: Max number of batch elements to generate images for.
  collections: Optional list of ops.GraphKeys.  The collections to add the
    summary to.  Defaults to [ops.GraphKeys.SUMMARIES]
  name: A name for the operation (optional).

Returns:
  A scalar `Tensor` of type `string`. The serialized `Summary` protocol
  buffer."
  [tag tensor & {:keys [max_images collections name]
                       :or {collections None name None}} ]
    (py/call-attr-kw deprecated "image_summary" [tag tensor] {:max_images max_images :collections collections :name name }))

(defn merge-all-summaries 
  "Merges all summaries collected in the default graph. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2016-11-30.
Instructions for updating:
Please switch to tf.summary.merge_all.

This op is deprecated. Please switch to tf.compat.v1.summary.merge_all, which
has
identical behavior.

Args:
  key: `GraphKey` used to collect the summaries.  Defaults to
    `GraphKeys.SUMMARIES`.

Returns:
  If no summaries were collected, returns None.  Otherwise returns a scalar
  `Tensor` of type `string` containing the serialized `Summary` protocol
  buffer resulting from the merging."
  [ & {:keys [key]} ]
   (py/call-attr-kw deprecated "merge_all_summaries" [] {:key key }))

(defn merge-summary 
  "Merges summaries. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2016-11-30.
Instructions for updating:
Please switch to tf.summary.merge.

This op is deprecated. Please switch to tf.compat.v1.summary.merge, which has
identical
behavior.

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
    added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
  name: A name for the operation (optional).

Returns:
  A scalar `Tensor` of type `string`. The serialized `Summary` protocol
  buffer resulting from the merging."
  [ inputs collections name ]
  (py/call-attr deprecated "merge_summary"  inputs collections name ))

(defn scalar-summary 
  "Outputs a `Summary` protocol buffer with scalar values. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2016-11-30.
Instructions for updating:
Please switch to tf.summary.scalar. Note that tf.summary.scalar uses the node name instead of the tag. This means that TensorFlow will automatically de-duplicate summary names based on the scope they are created in. Also, passing a tensor or list of tags to a scalar summary op is no longer supported.

This ops is deprecated. Please switch to tf.summary.scalar.
For an explanation of why this op was deprecated, and information on how to
migrate, look
['here'](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/deprecated/__init__.py)

The input `tags` and `values` must have the same shape.  The generated
summary has a summary value for each tag-value pair in `tags` and `values`.

Args:
  tags: A `string` `Tensor`.  Tags for the summaries.
  values: A real numeric Tensor.  Values for the summaries.
  collections: Optional list of graph collections keys. The new summary op is
    added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
  name: A name for the operation (optional).

Returns:
  A scalar `Tensor` of type `string`. The serialized `Summary` protocol
  buffer."
  [ tags values collections name ]
  (py/call-attr deprecated "scalar_summary"  tags values collections name ))
