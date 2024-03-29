(ns tensorflow.contrib.feature-column
  "Experimental utilities for tf.feature_column."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce feature-column (import-module "tensorflow.contrib.feature_column"))
(defn sequence-categorical-column-with-hash-bucket 
  "A sequence of categorical terms where ids are set by hashing.

  Pass this to `embedding_column` or `indicator_column` to convert sequence
  categorical data into dense representation for input to sequence NN, such as
  RNN.

  Example:

  ```python
  tokens = sequence_categorical_column_with_hash_bucket(
      'tokens', hash_bucket_size=1000)
  tokens_embedding = embedding_column(tokens, dimension=10)
  columns = [tokens_embedding]

  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  input_layer, sequence_length = sequence_input_layer(features, columns)

  rnn_cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell(hidden_size)
  outputs, state = tf.compat.v1.nn.dynamic_rnn(
      rnn_cell, inputs=input_layer, sequence_length=sequence_length)
  ```

  Args:
    key: A unique string identifying the input feature.
    hash_bucket_size: An int > 1. The number of buckets.
    dtype: The type of features. Only string and integer types are supported.

  Returns:
    A `_SequenceCategoricalColumn`.

  Raises:
    ValueError: `hash_bucket_size` is not greater than 1.
    ValueError: `dtype` is neither string nor integer.
  "
  [key hash_bucket_size  & {:keys [dtype]} ]
    (py/call-attr-kw feature-column "sequence_categorical_column_with_hash_bucket" [key hash_bucket_size] {:dtype dtype }))

(defn sequence-categorical-column-with-identity 
  "Returns a feature column that represents sequences of integers.

  Pass this to `embedding_column` or `indicator_column` to convert sequence
  categorical data into dense representation for input to sequence NN, such as
  RNN.

  Example:

  ```python
  watches = sequence_categorical_column_with_identity(
      'watches', num_buckets=1000)
  watches_embedding = embedding_column(watches, dimension=10)
  columns = [watches_embedding]

  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  input_layer, sequence_length = sequence_input_layer(features, columns)

  rnn_cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell(hidden_size)
  outputs, state = tf.compat.v1.nn.dynamic_rnn(
      rnn_cell, inputs=input_layer, sequence_length=sequence_length)
  ```

  Args:
    key: A unique string identifying the input feature.
    num_buckets: Range of inputs. Namely, inputs are expected to be in the
      range `[0, num_buckets)`.
    default_value: If `None`, this column's graph operations will fail for
      out-of-range inputs. Otherwise, this value must be in the range
      `[0, num_buckets)`, and will replace out-of-range inputs.

  Returns:
    A `_SequenceCategoricalColumn`.

  Raises:
    ValueError: if `num_buckets` is less than one.
    ValueError: if `default_value` is not in range `[0, num_buckets)`.
  "
  [ key num_buckets default_value ]
  (py/call-attr feature-column "sequence_categorical_column_with_identity"  key num_buckets default_value ))

(defn sequence-categorical-column-with-vocabulary-file 
  "A sequence of categorical terms where ids use a vocabulary file.

  Pass this to `embedding_column` or `indicator_column` to convert sequence
  categorical data into dense representation for input to sequence NN, such as
  RNN.

  Example:

  ```python
  states = sequence_categorical_column_with_vocabulary_file(
      key='states', vocabulary_file='/us/states.txt', vocabulary_size=50,
      num_oov_buckets=5)
  states_embedding = embedding_column(states, dimension=10)
  columns = [states_embedding]

  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  input_layer, sequence_length = sequence_input_layer(features, columns)

  rnn_cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell(hidden_size)
  outputs, state = tf.compat.v1.nn.dynamic_rnn(
      rnn_cell, inputs=input_layer, sequence_length=sequence_length)
  ```

  Args:
    key: A unique string identifying the input feature.
    vocabulary_file: The vocabulary file name.
    vocabulary_size: Number of the elements in the vocabulary. This must be no
      greater than length of `vocabulary_file`, if less than length, later
      values are ignored. If None, it is set to the length of `vocabulary_file`.
    num_oov_buckets: Non-negative integer, the number of out-of-vocabulary
      buckets. All out-of-vocabulary inputs will be assigned IDs in the range
      `[vocabulary_size, vocabulary_size+num_oov_buckets)` based on a hash of
      the input value. A positive `num_oov_buckets` can not be specified with
      `default_value`.
    default_value: The integer ID value to return for out-of-vocabulary feature
      values, defaults to `-1`. This can not be specified with a positive
      `num_oov_buckets`.
    dtype: The type of features. Only string and integer types are supported.

  Returns:
    A `_SequenceCategoricalColumn`.

  Raises:
    ValueError: `vocabulary_file` is missing or cannot be opened.
    ValueError: `vocabulary_size` is missing or < 1.
    ValueError: `num_oov_buckets` is a negative integer.
    ValueError: `num_oov_buckets` and `default_value` are both specified.
    ValueError: `dtype` is neither string nor integer.
  "
  [key vocabulary_file vocabulary_size & {:keys [num_oov_buckets default_value dtype]
                       :or {default_value None}} ]
    (py/call-attr-kw feature-column "sequence_categorical_column_with_vocabulary_file" [key vocabulary_file vocabulary_size] {:num_oov_buckets num_oov_buckets :default_value default_value :dtype dtype }))
(defn sequence-categorical-column-with-vocabulary-list 
  "A sequence of categorical terms where ids use an in-memory list.

  Pass this to `embedding_column` or `indicator_column` to convert sequence
  categorical data into dense representation for input to sequence NN, such as
  RNN.

  Example:

  ```python
  colors = sequence_categorical_column_with_vocabulary_list(
      key='colors', vocabulary_list=('R', 'G', 'B', 'Y'),
      num_oov_buckets=2)
  colors_embedding = embedding_column(colors, dimension=3)
  columns = [colors_embedding]

  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  input_layer, sequence_length = sequence_input_layer(features, columns)

  rnn_cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell(hidden_size)
  outputs, state = tf.compat.v1.nn.dynamic_rnn(
      rnn_cell, inputs=input_layer, sequence_length=sequence_length)
  ```

  Args:
    key: A unique string identifying the input feature.
    vocabulary_list: An ordered iterable defining the vocabulary. Each feature
      is mapped to the index of its value (if present) in `vocabulary_list`.
      Must be castable to `dtype`.
    dtype: The type of features. Only string and integer types are supported.
      If `None`, it will be inferred from `vocabulary_list`.
    default_value: The integer ID value to return for out-of-vocabulary feature
      values, defaults to `-1`. This can not be specified with a positive
      `num_oov_buckets`.
    num_oov_buckets: Non-negative integer, the number of out-of-vocabulary
      buckets. All out-of-vocabulary inputs will be assigned IDs in the range
      `[len(vocabulary_list), len(vocabulary_list)+num_oov_buckets)` based on a
      hash of the input value. A positive `num_oov_buckets` can not be specified
      with `default_value`.

  Returns:
    A `_SequenceCategoricalColumn`.

  Raises:
    ValueError: if `vocabulary_list` is empty, or contains duplicate keys.
    ValueError: `num_oov_buckets` is a negative integer.
    ValueError: `num_oov_buckets` and `default_value` are both specified.
    ValueError: if `dtype` is not integer or string.
  "
  [key vocabulary_list dtype  & {:keys [default_value num_oov_buckets]} ]
    (py/call-attr-kw feature-column "sequence_categorical_column_with_vocabulary_list" [key vocabulary_list dtype] {:default_value default_value :num_oov_buckets num_oov_buckets }))
(defn sequence-input-layer 
  "\"Builds input layer for sequence input.

  All `feature_columns` must be sequence dense columns with the same
  `sequence_length`. The output of this method can be fed into sequence
  networks, such as RNN.

  The output of this method is a 3D `Tensor` of shape `[batch_size, T, D]`.
  `T` is the maximum sequence length for this batch, which could differ from
  batch to batch.

  If multiple `feature_columns` are given with `Di` `num_elements` each, their
  outputs are concatenated. So, the final `Tensor` has shape
  `[batch_size, T, D0 + D1 + ... + Dn]`.

  Example:

  ```python
  rating = sequence_numeric_column('rating')
  watches = sequence_categorical_column_with_identity(
      'watches', num_buckets=1000)
  watches_embedding = embedding_column(watches, dimension=10)
  columns = [rating, watches]

  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  input_layer, sequence_length = sequence_input_layer(features, columns)

  rnn_cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell(hidden_size)
  outputs, state = tf.compat.v1.nn.dynamic_rnn(
      rnn_cell, inputs=input_layer, sequence_length=sequence_length)
  ```

  Args:
    features: A dict mapping keys to tensors.
    feature_columns: An iterable of dense sequence columns. Valid columns are
      - `embedding_column` that wraps a `sequence_categorical_column_with_*`
      - `sequence_numeric_column`.
    weight_collections: A list of collection names to which the Variable will be
      added. Note that variables will also be added to collections
      `tf.GraphKeys.GLOBAL_VARIABLES` and `ops.GraphKeys.MODEL_VARIABLES`.
    trainable: If `True` also add the variable to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES`.

  Returns:
    An `(input_layer, sequence_length)` tuple where:
    - input_layer: A float `Tensor` of shape `[batch_size, T, D]`.
        `T` is the maximum sequence length for this batch, which could differ
        from batch to batch. `D` is the sum of `num_elements` for all
        `feature_columns`.
    - sequence_length: An int `Tensor` of shape `[batch_size]`. The sequence
        length for each example.

  Raises:
    ValueError: If any of the `feature_columns` is the wrong type.
  "
  [features feature_columns weight_collections  & {:keys [trainable]} ]
    (py/call-attr-kw feature-column "sequence_input_layer" [features feature_columns weight_collections] {:trainable trainable }))

(defn sequence-numeric-column 
  "Returns a feature column that represents sequences of numeric data.

  Example:

  ```python
  temperature = sequence_numeric_column('temperature')
  columns = [temperature]

  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  input_layer, sequence_length = sequence_input_layer(features, columns)

  rnn_cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell(hidden_size)
  outputs, state = tf.compat.v1.nn.dynamic_rnn(
      rnn_cell, inputs=input_layer, sequence_length=sequence_length)
  ```

  Args:
    key: A unique string identifying the input features.
    shape: The shape of the input data per sequence id. E.g. if `shape=(2,)`,
      each example must contain `2 * sequence_length` values.
    default_value: A single value compatible with `dtype` that is used for
      padding the sparse data into a dense `Tensor`.
    dtype: The type of values.
    normalizer_fn: If not `None`, a function that can be used to normalize the
      value of the tensor after `default_value` is applied for parsing.
      Normalizer function takes the input `Tensor` as its argument, and returns
      the output `Tensor`. (e.g. lambda x: (x - 3.0) / 4.2). Please note that
      even though the most common use case of this function is normalization, it
      can be used for any kind of Tensorflow transformations.

  Returns:
    A `_SequenceNumericColumn`.

  Raises:
    TypeError: if any dimension in shape is not an int.
    ValueError: if any dimension in shape is not a positive integer.
    ValueError: if `dtype` is not convertible to `tf.float32`.
  "
  [key & {:keys [shape default_value dtype normalizer_fn]
                       :or {normalizer_fn None}} ]
    (py/call-attr-kw feature-column "sequence_numeric_column" [key] {:shape shape :default_value default_value :dtype dtype :normalizer_fn normalizer_fn }))
