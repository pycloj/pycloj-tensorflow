(ns tensorflow.-api.v1.compat.v1.feature-column
  "Public API for tf.feature_column namespace.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce feature-column (import-module "tensorflow._api.v1.compat.v1.feature_column"))

(defn bucketized-column 
  "Represents discretized dense input.

  Buckets include the left boundary, and exclude the right boundary. Namely,
  `boundaries=[0., 1., 2.]` generates buckets `(-inf, 0.)`, `[0., 1.)`,
  `[1., 2.)`, and `[2., +inf)`.

  For example, if the inputs are

  ```python
  boundaries = [0, 10, 100]
  input tensor = [[-5, 10000]
                  [150,   10]
                  [5,    100]]
  ```

  then the output will be

  ```python
  output = [[0, 3]
            [3, 2]
            [1, 3]]
  ```

  Example:

  ```python
  price = numeric_column('price')
  bucketized_price = bucketized_column(price, boundaries=[...])
  columns = [bucketized_price, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction = linear_model(features, columns)

  # or
  columns = [bucketized_price, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  dense_tensor = input_layer(features, columns)
  ```

  `bucketized_column` can also be crossed with another categorical column using
  `crossed_column`:

  ```python
  price = numeric_column('price')
  # bucketized_column converts numerical feature to a categorical one.
  bucketized_price = bucketized_column(price, boundaries=[...])
  # 'keywords' is a string feature.
  price_x_keywords = crossed_column([bucketized_price, 'keywords'], 50K)
  columns = [price_x_keywords, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction = linear_model(features, columns)
  ```

  Args:
    source_column: A one-dimensional dense column which is generated with
      `numeric_column`.
    boundaries: A sorted list or tuple of floats specifying the boundaries.

  Returns:
    A `BucketizedColumn`.

  Raises:
    ValueError: If `source_column` is not a numeric column, or if it is not
      one-dimensional.
    ValueError: If `boundaries` is not a sorted list or tuple.
  "
  [ source_column boundaries ]
  (py/call-attr feature-column "bucketized_column"  source_column boundaries ))
(defn categorical-column-with-hash-bucket 
  "Represents sparse feature where ids are set by hashing.

  Use this when your sparse features are in string or integer format, and you
  want to distribute your inputs into a finite number of buckets by hashing.
  output_id = Hash(input_feature_string) % bucket_size for string type input.
  For int type input, the value is converted to its string representation first
  and then hashed by the same formula.

  For input dictionary `features`, `features[key]` is either `Tensor` or
  `SparseTensor`. If `Tensor`, missing values can be represented by `-1` for int
  and `''` for string, which will be dropped by this feature column.

  Example:

  ```python
  keywords = categorical_column_with_hash_bucket(\"keywords\", 10K)
  columns = [keywords, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction = linear_model(features, columns)

  # or
  keywords_embedded = embedding_column(keywords, 16)
  columns = [keywords_embedded, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  dense_tensor = input_layer(features, columns)
  ```

  Args:
    key: A unique string identifying the input feature. It is used as the
      column name and the dictionary key for feature parsing configs, feature
      `Tensor` objects, and feature columns.
    hash_bucket_size: An int > 1. The number of buckets.
    dtype: The type of features. Only string and integer types are supported.

  Returns:
    A `HashedCategoricalColumn`.

  Raises:
    ValueError: `hash_bucket_size` is not greater than 1.
    ValueError: `dtype` is neither string nor integer.
  "
  [key hash_bucket_size  & {:keys [dtype]} ]
    (py/call-attr-kw feature-column "categorical_column_with_hash_bucket" [key hash_bucket_size] {:dtype dtype }))

(defn categorical-column-with-identity 
  "A `CategoricalColumn` that returns identity values.

  Use this when your inputs are integers in the range `[0, num_buckets)`, and
  you want to use the input value itself as the categorical ID. Values outside
  this range will result in `default_value` if specified, otherwise it will
  fail.

  Typically, this is used for contiguous ranges of integer indexes, but
  it doesn't have to be. This might be inefficient, however, if many of IDs
  are unused. Consider `categorical_column_with_hash_bucket` in that case.

  For input dictionary `features`, `features[key]` is either `Tensor` or
  `SparseTensor`. If `Tensor`, missing values can be represented by `-1` for int
  and `''` for string, which will be dropped by this feature column.

  In the following examples, each input in the range `[0, 1000000)` is assigned
  the same value. All other inputs are assigned `default_value` 0. Note that a
  literal 0 in inputs will result in the same default ID.

  Linear model:

  ```python
  video_id = categorical_column_with_identity(
      key='video_id', num_buckets=1000000, default_value=0)
  columns = [video_id, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction, _, _ = linear_model(features, columns)
  ```

  Embedding for a DNN model:

  ```python
  columns = [embedding_column(video_id, 9),...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  dense_tensor = input_layer(features, columns)
  ```

  Args:
    key: A unique string identifying the input feature. It is used as the
      column name and the dictionary key for feature parsing configs, feature
      `Tensor` objects, and feature columns.
    num_buckets: Range of inputs and outputs is `[0, num_buckets)`.
    default_value: If `None`, this column's graph operations will fail for
      out-of-range inputs. Otherwise, this value must be in the range
      `[0, num_buckets)`, and will replace inputs in that range.

  Returns:
    A `CategoricalColumn` that returns identity values.

  Raises:
    ValueError: if `num_buckets` is less than one.
    ValueError: if `default_value` is not in range `[0, num_buckets)`.
  "
  [ key num_buckets default_value ]
  (py/call-attr feature-column "categorical_column_with_identity"  key num_buckets default_value ))

(defn categorical-column-with-vocabulary-file 
  "A `CategoricalColumn` with a vocabulary file.

  Use this when your inputs are in string or integer format, and you have a
  vocabulary file that maps each value to an integer ID. By default,
  out-of-vocabulary values are ignored. Use either (but not both) of
  `num_oov_buckets` and `default_value` to specify how to include
  out-of-vocabulary values.

  For input dictionary `features`, `features[key]` is either `Tensor` or
  `SparseTensor`. If `Tensor`, missing values can be represented by `-1` for int
  and `''` for string, which will be dropped by this feature column.

  Example with `num_oov_buckets`:
  File '/us/states.txt' contains 50 lines, each with a 2-character U.S. state
  abbreviation. All inputs with values in that file are assigned an ID 0-49,
  corresponding to its line number. All other values are hashed and assigned an
  ID 50-54.

  ```python
  states = categorical_column_with_vocabulary_file(
      key='states', vocabulary_file='/us/states.txt', vocabulary_size=50,
      num_oov_buckets=5)
  columns = [states, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction = linear_model(features, columns)
  ```

  Example with `default_value`:
  File '/us/states.txt' contains 51 lines - the first line is 'XX', and the
  other 50 each have a 2-character U.S. state abbreviation. Both a literal 'XX'
  in input, and other values missing from the file, will be assigned ID 0. All
  others are assigned the corresponding line number 1-50.

  ```python
  states = categorical_column_with_vocabulary_file(
      key='states', vocabulary_file='/us/states.txt', vocabulary_size=51,
      default_value=0)
  columns = [states, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction, _, _ = linear_model(features, columns)
  ```

  And to make an embedding with either:

  ```python
  columns = [embedding_column(states, 3),...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  dense_tensor = input_layer(features, columns)
  ```

  Args:
    key: A unique string identifying the input feature. It is used as the
      column name and the dictionary key for feature parsing configs, feature
      `Tensor` objects, and feature columns.
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
    A `CategoricalColumn` with a vocabulary file.

  Raises:
    ValueError: `vocabulary_file` is missing or cannot be opened.
    ValueError: `vocabulary_size` is missing or < 1.
    ValueError: `num_oov_buckets` is a negative integer.
    ValueError: `num_oov_buckets` and `default_value` are both specified.
    ValueError: `dtype` is neither string nor integer.
  "
  [key vocabulary_file vocabulary_size & {:keys [num_oov_buckets default_value dtype]
                       :or {default_value None}} ]
    (py/call-attr-kw feature-column "categorical_column_with_vocabulary_file" [key vocabulary_file vocabulary_size] {:num_oov_buckets num_oov_buckets :default_value default_value :dtype dtype }))
(defn categorical-column-with-vocabulary-list 
  "A `CategoricalColumn` with in-memory vocabulary.

  Use this when your inputs are in string or integer format, and you have an
  in-memory vocabulary mapping each value to an integer ID. By default,
  out-of-vocabulary values are ignored. Use either (but not both) of
  `num_oov_buckets` and `default_value` to specify how to include
  out-of-vocabulary values.

  For input dictionary `features`, `features[key]` is either `Tensor` or
  `SparseTensor`. If `Tensor`, missing values can be represented by `-1` for int
  and `''` for string, which will be dropped by this feature column.

  Example with `num_oov_buckets`:
  In the following example, each input in `vocabulary_list` is assigned an ID
  0-3 corresponding to its index (e.g., input 'B' produces output 2). All other
  inputs are hashed and assigned an ID 4-5.

  ```python
  colors = categorical_column_with_vocabulary_list(
      key='colors', vocabulary_list=('R', 'G', 'B', 'Y'),
      num_oov_buckets=2)
  columns = [colors, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction, _, _ = linear_model(features, columns)
  ```

  Example with `default_value`:
  In the following example, each input in `vocabulary_list` is assigned an ID
  0-4 corresponding to its index (e.g., input 'B' produces output 3). All other
  inputs are assigned `default_value` 0.


  ```python
  colors = categorical_column_with_vocabulary_list(
      key='colors', vocabulary_list=('X', 'R', 'G', 'B', 'Y'), default_value=0)
  columns = [colors, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction, _, _ = linear_model(features, columns)
  ```

  And to make an embedding with either:

  ```python
  columns = [embedding_column(colors, 3),...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  dense_tensor = input_layer(features, columns)
  ```

  Args:
    key: A unique string identifying the input feature. It is used as the column
      name and the dictionary key for feature parsing configs, feature `Tensor`
      objects, and feature columns.
    vocabulary_list: An ordered iterable defining the vocabulary. Each feature
      is mapped to the index of its value (if present) in `vocabulary_list`.
      Must be castable to `dtype`.
    dtype: The type of features. Only string and integer types are supported. If
      `None`, it will be inferred from `vocabulary_list`.
    default_value: The integer ID value to return for out-of-vocabulary feature
      values, defaults to `-1`. This can not be specified with a positive
      `num_oov_buckets`.
    num_oov_buckets: Non-negative integer, the number of out-of-vocabulary
      buckets. All out-of-vocabulary inputs will be assigned IDs in the range
      `[len(vocabulary_list), len(vocabulary_list)+num_oov_buckets)` based on a
      hash of the input value. A positive `num_oov_buckets` can not be specified
      with `default_value`.

  Returns:
    A `CategoricalColumn` with in-memory vocabulary.

  Raises:
    ValueError: if `vocabulary_list` is empty, or contains duplicate keys.
    ValueError: `num_oov_buckets` is a negative integer.
    ValueError: `num_oov_buckets` and `default_value` are both specified.
    ValueError: if `dtype` is not integer or string.
  "
  [key vocabulary_list dtype  & {:keys [default_value num_oov_buckets]} ]
    (py/call-attr-kw feature-column "categorical_column_with_vocabulary_list" [key vocabulary_list dtype] {:default_value default_value :num_oov_buckets num_oov_buckets }))

(defn crossed-column 
  "Returns a column for performing crosses of categorical features.

  Crossed features will be hashed according to `hash_bucket_size`. Conceptually,
  the transformation can be thought of as:
    Hash(cartesian product of features) % `hash_bucket_size`

  For example, if the input features are:

  * SparseTensor referred by first key:

    ```python
    shape = [2, 2]
    {
        [0, 0]: \"a\"
        [1, 0]: \"b\"
        [1, 1]: \"c\"
    }
    ```

  * SparseTensor referred by second key:

    ```python
    shape = [2, 1]
    {
        [0, 0]: \"d\"
        [1, 0]: \"e\"
    }
    ```

  then crossed feature will look like:

  ```python
   shape = [2, 2]
  {
      [0, 0]: Hash64(\"d\", Hash64(\"a\")) % hash_bucket_size
      [1, 0]: Hash64(\"e\", Hash64(\"b\")) % hash_bucket_size
      [1, 1]: Hash64(\"e\", Hash64(\"c\")) % hash_bucket_size
  }
  ```

  Here is an example to create a linear model with crosses of string features:

  ```python
  keywords_x_doc_terms = crossed_column(['keywords', 'doc_terms'], 50K)
  columns = [keywords_x_doc_terms, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction = linear_model(features, columns)
  ```

  You could also use vocabulary lookup before crossing:

  ```python
  keywords = categorical_column_with_vocabulary_file(
      'keywords', '/path/to/vocabulary/file', vocabulary_size=1K)
  keywords_x_doc_terms = crossed_column([keywords, 'doc_terms'], 50K)
  columns = [keywords_x_doc_terms, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction = linear_model(features, columns)
  ```

  If an input feature is of numeric type, you can use
  `categorical_column_with_identity`, or `bucketized_column`, as in the example:

  ```python
  # vertical_id is an integer categorical feature.
  vertical_id = categorical_column_with_identity('vertical_id', 10K)
  price = numeric_column('price')
  # bucketized_column converts numerical feature to a categorical one.
  bucketized_price = bucketized_column(price, boundaries=[...])
  vertical_id_x_price = crossed_column([vertical_id, bucketized_price], 50K)
  columns = [vertical_id_x_price, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction = linear_model(features, columns)
  ```

  To use crossed column in DNN model, you need to add it in an embedding column
  as in this example:

  ```python
  vertical_id_x_price = crossed_column([vertical_id, bucketized_price], 50K)
  vertical_id_x_price_embedded = embedding_column(vertical_id_x_price, 10)
  dense_tensor = input_layer(features, [vertical_id_x_price_embedded, ...])
  ```

  Args:
    keys: An iterable identifying the features to be crossed. Each element can
      be either:
      * string: Will use the corresponding feature which must be of string type.
      * `CategoricalColumn`: Will use the transformed tensor produced by this
        column. Does not support hashed categorical column.
    hash_bucket_size: An int > 1. The number of buckets.
    hash_key: Specify the hash_key that will be used by the `FingerprintCat64`
      function to combine the crosses fingerprints on SparseCrossOp (optional).

  Returns:
    A `CrossedColumn`.

  Raises:
    ValueError: If `len(keys) < 2`.
    ValueError: If any of the keys is neither a string nor `CategoricalColumn`.
    ValueError: If any of the keys is `HashedCategoricalColumn`.
    ValueError: If `hash_bucket_size < 1`.
  "
  [ keys hash_bucket_size hash_key ]
  (py/call-attr feature-column "crossed_column"  keys hash_bucket_size hash_key ))

(defn embedding-column 
  "`DenseColumn` that converts from sparse, categorical input.

  Use this when your inputs are sparse, but you want to convert them to a dense
  representation (e.g., to feed to a DNN).

  Inputs must be a `CategoricalColumn` created by any of the
  `categorical_column_*` function. Here is an example of using
  `embedding_column` with `DNNClassifier`:

  ```python
  video_id = categorical_column_with_identity(
      key='video_id', num_buckets=1000000, default_value=0)
  columns = [embedding_column(video_id, 9),...]

  estimator = tf.estimator.DNNClassifier(feature_columns=columns, ...)

  label_column = ...
  def input_fn():
    features = tf.io.parse_example(
        ..., features=make_parse_example_spec(columns + [label_column]))
    labels = features.pop(label_column.name)
    return features, labels

  estimator.train(input_fn=input_fn, steps=100)
  ```

  Here is an example using `embedding_column` with model_fn:

  ```python
  def model_fn(features, ...):
    video_id = categorical_column_with_identity(
        key='video_id', num_buckets=1000000, default_value=0)
    columns = [embedding_column(video_id, 9),...]
    dense_tensor = input_layer(features, columns)
    # Form DNN layers, calculate loss, and return EstimatorSpec.
    ...
  ```

  Args:
    categorical_column: A `CategoricalColumn` created by a
      `categorical_column_with_*` function. This column produces the sparse IDs
      that are inputs to the embedding lookup.
    dimension: An integer specifying dimension of the embedding, must be > 0.
    combiner: A string specifying how to reduce if there are multiple entries in
      a single row. Currently 'mean', 'sqrtn' and 'sum' are supported, with
      'mean' the default. 'sqrtn' often achieves good accuracy, in particular
      with bag-of-words columns. Each of this can be thought as example level
      normalizations on the column. For more information, see
      `tf.embedding_lookup_sparse`.
    initializer: A variable initializer function to be used in embedding
      variable initialization. If not specified, defaults to
      `truncated_normal_initializer` with mean `0.0` and
      standard deviation `1/sqrt(dimension)`.
    ckpt_to_load_from: String representing checkpoint name/pattern from which to
      restore column weights. Required if `tensor_name_in_ckpt` is not `None`.
    tensor_name_in_ckpt: Name of the `Tensor` in `ckpt_to_load_from` from which
      to restore the column weights. Required if `ckpt_to_load_from` is not
      `None`.
    max_norm: If not `None`, embedding values are l2-normalized to this value.
    trainable: Whether or not the embedding is trainable. Default is True.

  Returns:
    `DenseColumn` that converts from sparse input.

  Raises:
    ValueError: if `dimension` not > 0.
    ValueError: if exactly one of `ckpt_to_load_from` and `tensor_name_in_ckpt`
      is specified.
    ValueError: if `initializer` is specified and is not callable.
    RuntimeError: If eager execution is enabled.
  "
  [categorical_column dimension & {:keys [combiner initializer ckpt_to_load_from tensor_name_in_ckpt max_norm trainable]
                       :or {initializer None ckpt_to_load_from None tensor_name_in_ckpt None max_norm None}} ]
    (py/call-attr-kw feature-column "embedding_column" [categorical_column dimension] {:combiner combiner :initializer initializer :ckpt_to_load_from ckpt_to_load_from :tensor_name_in_ckpt tensor_name_in_ckpt :max_norm max_norm :trainable trainable }))

(defn indicator-column 
  "Represents multi-hot representation of given categorical column.

  - For DNN model, `indicator_column` can be used to wrap any
    `categorical_column_*` (e.g., to feed to DNN). Consider to Use
    `embedding_column` if the number of buckets/unique(values) are large.

  - For Wide (aka linear) model, `indicator_column` is the internal
    representation for categorical column when passing categorical column
    directly (as any element in feature_columns) to `linear_model`. See
    `linear_model` for details.

  ```python
  name = indicator_column(categorical_column_with_vocabulary_list(
      'name', ['bob', 'george', 'wanda'])
  columns = [name, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  dense_tensor = input_layer(features, columns)

  dense_tensor == [[1, 0, 0]]  # If \"name\" bytes_list is [\"bob\"]
  dense_tensor == [[1, 0, 1]]  # If \"name\" bytes_list is [\"bob\", \"wanda\"]
  dense_tensor == [[2, 0, 0]]  # If \"name\" bytes_list is [\"bob\", \"bob\"]
  ```

  Args:
    categorical_column: A `CategoricalColumn` which is created by
      `categorical_column_with_*` or `crossed_column` functions.

  Returns:
    An `IndicatorColumn`.
  "
  [ categorical_column ]
  (py/call-attr feature-column "indicator_column"  categorical_column ))

(defn input-layer 
  "Returns a dense `Tensor` as input layer based on given `feature_columns`.

  Generally a single example in training data is described with FeatureColumns.
  At the first layer of the model, this column oriented data should be converted
  to a single `Tensor`.

  Example:

  ```python
  price = numeric_column('price')
  keywords_embedded = embedding_column(
      categorical_column_with_hash_bucket(\"keywords\", 10K), dimensions=16)
  columns = [price, keywords_embedded, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  dense_tensor = input_layer(features, columns)
  for units in [128, 64, 32]:
    dense_tensor = tf.compat.v1.layers.dense(dense_tensor, units, tf.nn.relu)
  prediction = tf.compat.v1.layers.dense(dense_tensor, 1)
  ```

  Args:
    features: A mapping from key to tensors. `_FeatureColumn`s look up via these
      keys. For example `numeric_column('price')` will look at 'price' key in
      this dict. Values can be a `SparseTensor` or a `Tensor` depends on
      corresponding `_FeatureColumn`.
    feature_columns: An iterable containing the FeatureColumns to use as inputs
      to your model. All items should be instances of classes derived from
      `_DenseColumn` such as `numeric_column`, `embedding_column`,
      `bucketized_column`, `indicator_column`. If you have categorical features,
      you can wrap them with an `embedding_column` or `indicator_column`.
    weight_collections: A list of collection names to which the Variable will be
      added. Note that variables will also be added to collections
      `tf.GraphKeys.GLOBAL_VARIABLES` and `ops.GraphKeys.MODEL_VARIABLES`.
    trainable: If `True` also add the variable to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    cols_to_vars: If not `None`, must be a dictionary that will be filled with a
      mapping from `_FeatureColumn` to list of `Variable`s.  For example, after
      the call, we might have cols_to_vars =
      {_EmbeddingColumn(
        categorical_column=_HashedCategoricalColumn(
          key='sparse_feature', hash_bucket_size=5, dtype=tf.string),
        dimension=10): [<tf.Variable 'some_variable:0' shape=(5, 10),
                        <tf.Variable 'some_variable:1' shape=(5, 10)]}
      If a column creates no variables, its value will be an empty list.
    cols_to_output_tensors: If not `None`, must be a dictionary that will be
      filled with a mapping from '_FeatureColumn' to the associated
      output `Tensor`s.

  Returns:
    A `Tensor` which represents input layer of a model. Its shape
    is (batch_size, first_layer_dimension) and its dtype is `float32`.
    first_layer_dimension is determined based on given `feature_columns`.

  Raises:
    ValueError: if an item in `feature_columns` is not a `_DenseColumn`.
  "
  [features feature_columns weight_collections & {:keys [trainable cols_to_vars cols_to_output_tensors]
                       :or {cols_to_vars None cols_to_output_tensors None}} ]
    (py/call-attr-kw feature-column "input_layer" [features feature_columns weight_collections] {:trainable trainable :cols_to_vars cols_to_vars :cols_to_output_tensors cols_to_output_tensors }))

(defn linear-model 
  "Returns a linear prediction `Tensor` based on given `feature_columns`.

  This function generates a weighted sum based on output dimension `units`.
  Weighted sum refers to logits in classification problems. It refers to the
  prediction itself for linear regression problems.

  Note on supported columns: `linear_model` treats categorical columns as
  `indicator_column`s. To be specific, assume the input as `SparseTensor` looks
  like:

  ```python
    shape = [2, 2]
    {
        [0, 0]: \"a\"
        [1, 0]: \"b\"
        [1, 1]: \"c\"
    }
  ```
  `linear_model` assigns weights for the presence of \"a\", \"b\", \"c' implicitly,
  just like `indicator_column`, while `input_layer` explicitly requires wrapping
  each of categorical columns with an `embedding_column` or an
  `indicator_column`.

  Example of usage:

  ```python
  price = numeric_column('price')
  price_buckets = bucketized_column(price, boundaries=[0., 10., 100., 1000.])
  keywords = categorical_column_with_hash_bucket(\"keywords\", 10K)
  keywords_price = crossed_column('keywords', price_buckets, ...)
  columns = [price_buckets, keywords, keywords_price ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  prediction = linear_model(features, columns)
  ```

  The `sparse_combiner` argument works as follows
  For example, for two features represented as the categorical columns:

  ```python
    # Feature 1

    shape = [2, 2]
    {
        [0, 0]: \"a\"
        [0, 1]: \"b\"
        [1, 0]: \"c\"
    }

    # Feature 2

    shape = [2, 3]
    {
        [0, 0]: \"d\"
        [1, 0]: \"e\"
        [1, 1]: \"f\"
        [1, 2]: \"f\"
    }
  ```

  with `sparse_combiner` as \"mean\", the linear model outputs consequently
  are:

  ```
    y_0 = 1.0 / 2.0 * ( w_a + w_b ) + w_d + b
    y_1 = w_c + 1.0 / 3.0 * ( w_e + 2.0 * w_f ) + b
  ```

  where `y_i` is the output, `b` is the bias, and `w_x` is the weight
  assigned to the presence of `x` in the input features.

  Args:
    features: A mapping from key to tensors. `_FeatureColumn`s look up via these
      keys. For example `numeric_column('price')` will look at 'price' key in
      this dict. Values are `Tensor` or `SparseTensor` depending on
      corresponding `_FeatureColumn`.
    feature_columns: An iterable containing the FeatureColumns to use as inputs
      to your model. All items should be instances of classes derived from
      `_FeatureColumn`s.
    units: An integer, dimensionality of the output space. Default value is 1.
    sparse_combiner: A string specifying how to reduce if a categorical column
      is multivalent. Except `numeric_column`, almost all columns passed to
      `linear_model` are considered as categorical columns.  It combines each
      categorical column independently. Currently \"mean\", \"sqrtn\" and \"sum\" are
      supported, with \"sum\" the default for linear model. \"sqrtn\" often achieves
      good accuracy, in particular with bag-of-words columns.
        * \"sum\": do not normalize features in the column
        * \"mean\": do l1 normalization on features in the column
        * \"sqrtn\": do l2 normalization on features in the column
    weight_collections: A list of collection names to which the Variable will be
      added. Note that, variables will also be added to collections
      `tf.GraphKeys.GLOBAL_VARIABLES` and `ops.GraphKeys.MODEL_VARIABLES`.
    trainable: If `True` also add the variable to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    cols_to_vars: If not `None`, must be a dictionary that will be filled with a
      mapping from `_FeatureColumn` to associated list of `Variable`s.  For
      example, after the call, we might have cols_to_vars = {
        _NumericColumn(
          key='numeric_feature1', shape=(1,):
        [<tf.Variable 'linear_model/price2/weights:0' shape=(1, 1)>],
        'bias': [<tf.Variable 'linear_model/bias_weights:0' shape=(1,)>],
        _NumericColumn(
          key='numeric_feature2', shape=(2,)):
        [<tf.Variable 'linear_model/price1/weights:0' shape=(2, 1)>]}
      If a column creates no variables, its value will be an empty list. Note
      that cols_to_vars will also contain a string key 'bias' that maps to a
      list of Variables.

  Returns:
    A `Tensor` which represents predictions/logits of a linear model. Its shape
    is (batch_size, units) and its dtype is `float32`.

  Raises:
    ValueError: if an item in `feature_columns` is neither a `_DenseColumn`
      nor `_CategoricalColumn`.
  "
  [features feature_columns & {:keys [units sparse_combiner weight_collections trainable cols_to_vars]
                       :or {weight_collections None cols_to_vars None}} ]
    (py/call-attr-kw feature-column "linear_model" [features feature_columns] {:units units :sparse_combiner sparse_combiner :weight_collections weight_collections :trainable trainable :cols_to_vars cols_to_vars }))

(defn make-parse-example-spec 
  "Creates parsing spec dictionary from input feature_columns.

  The returned dictionary can be used as arg 'features' in
  `tf.io.parse_example`.

  Typical usage example:

  ```python
  # Define features and transformations
  feature_a = categorical_column_with_vocabulary_file(...)
  feature_b = numeric_column(...)
  feature_c_bucketized = bucketized_column(numeric_column(\"feature_c\"), ...)
  feature_a_x_feature_c = crossed_column(
      columns=[\"feature_a\", feature_c_bucketized], ...)

  feature_columns = set(
      [feature_b, feature_c_bucketized, feature_a_x_feature_c])
  features = tf.io.parse_example(
      serialized=serialized_examples,
      features=make_parse_example_spec(feature_columns))
  ```

  For the above example, make_parse_example_spec would return the dict:

  ```python
  {
      \"feature_a\": parsing_ops.VarLenFeature(tf.string),
      \"feature_b\": parsing_ops.FixedLenFeature([1], dtype=tf.float32),
      \"feature_c\": parsing_ops.FixedLenFeature([1], dtype=tf.float32)
  }
  ```

  Args:
    feature_columns: An iterable containing all feature columns. All items
      should be instances of classes derived from `_FeatureColumn`.

  Returns:
    A dict mapping each feature key to a `FixedLenFeature` or `VarLenFeature`
    value.

  Raises:
    ValueError: If any of the given `feature_columns` is not a `_FeatureColumn`
      instance.
  "
  [ feature_columns ]
  (py/call-attr feature-column "make_parse_example_spec"  feature_columns ))

(defn numeric-column 
  "Represents real valued or numerical features.

  Example:

  ```python
  price = numeric_column('price')
  columns = [price, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  dense_tensor = input_layer(features, columns)

  # or
  bucketized_price = bucketized_column(price, boundaries=[...])
  columns = [bucketized_price, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction = linear_model(features, columns)
  ```

  Args:
    key: A unique string identifying the input feature. It is used as the
      column name and the dictionary key for feature parsing configs, feature
      `Tensor` objects, and feature columns.
    shape: An iterable of integers specifies the shape of the `Tensor`. An
      integer can be given which means a single dimension `Tensor` with given
      width. The `Tensor` representing the column will have the shape of
      [batch_size] + `shape`.
    default_value: A single value compatible with `dtype` or an iterable of
      values compatible with `dtype` which the column takes on during
      `tf.Example` parsing if data is missing. A default value of `None` will
      cause `tf.io.parse_example` to fail if an example does not contain this
      column. If a single value is provided, the same value will be applied as
      the default value for every item. If an iterable of values is provided,
      the shape of the `default_value` should be equal to the given `shape`.
    dtype: defines the type of values. Default value is `tf.float32`. Must be a
      non-quantized, real integer or floating point type.
    normalizer_fn: If not `None`, a function that can be used to normalize the
      value of the tensor after `default_value` is applied for parsing.
      Normalizer function takes the input `Tensor` as its argument, and returns
      the output `Tensor`. (e.g. lambda x: (x - 3.0) / 4.2). Please note that
      even though the most common use case of this function is normalization, it
      can be used for any kind of Tensorflow transformations.

  Returns:
    A `NumericColumn`.

  Raises:
    TypeError: if any dimension in shape is not an int
    ValueError: if any dimension in shape is not a positive integer
    TypeError: if `default_value` is an iterable but not compatible with `shape`
    TypeError: if `default_value` is not compatible with `dtype`.
    ValueError: if `dtype` is not convertible to `tf.float32`.
  "
  [key & {:keys [shape default_value dtype normalizer_fn]
                       :or {default_value None normalizer_fn None}} ]
    (py/call-attr-kw feature-column "numeric_column" [key] {:shape shape :default_value default_value :dtype dtype :normalizer_fn normalizer_fn }))
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
  sequence_feature_layer = SequenceFeatures(columns)
  sequence_input, sequence_length = sequence_feature_layer(features)
  sequence_length_mask = tf.sequence_mask(sequence_length)

  rnn_cell = tf.keras.layers.SimpleRNNCell(hidden_size)
  rnn_layer = tf.keras.layers.RNN(rnn_cell)
  outputs, state = rnn_layer(sequence_input, mask=sequence_length_mask)
  ```

  Args:
    key: A unique string identifying the input feature.
    hash_bucket_size: An int > 1. The number of buckets.
    dtype: The type of features. Only string and integer types are supported.

  Returns:
    A `SequenceCategoricalColumn`.

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
  sequence_feature_layer = SequenceFeatures(columns)
  sequence_input, sequence_length = sequence_feature_layer(features)
  sequence_length_mask = tf.sequence_mask(sequence_length)

  rnn_cell = tf.keras.layers.SimpleRNNCell(hidden_size)
  rnn_layer = tf.keras.layers.RNN(rnn_cell)
  outputs, state = rnn_layer(sequence_input, mask=sequence_length_mask)
  ```

  Args:
    key: A unique string identifying the input feature.
    num_buckets: Range of inputs. Namely, inputs are expected to be in the
      range `[0, num_buckets)`.
    default_value: If `None`, this column's graph operations will fail for
      out-of-range inputs. Otherwise, this value must be in the range
      `[0, num_buckets)`, and will replace out-of-range inputs.

  Returns:
    A `SequenceCategoricalColumn`.

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
  sequence_feature_layer = SequenceFeatures(columns)
  sequence_input, sequence_length = sequence_feature_layer(features)
  sequence_length_mask = tf.sequence_mask(sequence_length)

  rnn_cell = tf.keras.layers.SimpleRNNCell(hidden_size)
  rnn_layer = tf.keras.layers.RNN(rnn_cell)
  outputs, state = rnn_layer(sequence_input, mask=sequence_length_mask)
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
    A `SequenceCategoricalColumn`.

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
  sequence_feature_layer = SequenceFeatures(columns)
  sequence_input, sequence_length = sequence_feature_layer(features)
  sequence_length_mask = tf.sequence_mask(sequence_length)

  rnn_cell = tf.keras.layers.SimpleRNNCell(hidden_size)
  rnn_layer = tf.keras.layers.RNN(rnn_cell)
  outputs, state = rnn_layer(sequence_input, mask=sequence_length_mask)
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
    A `SequenceCategoricalColumn`.

  Raises:
    ValueError: if `vocabulary_list` is empty, or contains duplicate keys.
    ValueError: `num_oov_buckets` is a negative integer.
    ValueError: `num_oov_buckets` and `default_value` are both specified.
    ValueError: if `dtype` is not integer or string.
  "
  [key vocabulary_list dtype  & {:keys [default_value num_oov_buckets]} ]
    (py/call-attr-kw feature-column "sequence_categorical_column_with_vocabulary_list" [key vocabulary_list dtype] {:default_value default_value :num_oov_buckets num_oov_buckets }))

(defn sequence-numeric-column 
  "Returns a feature column that represents sequences of numeric data.

  Example:

  ```python
  temperature = sequence_numeric_column('temperature')
  columns = [temperature]

  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  sequence_feature_layer = SequenceFeatures(columns)
  sequence_input, sequence_length = sequence_feature_layer(features)
  sequence_length_mask = tf.sequence_mask(sequence_length)

  rnn_cell = tf.keras.layers.SimpleRNNCell(hidden_size)
  rnn_layer = tf.keras.layers.RNN(rnn_cell)
  outputs, state = rnn_layer(sequence_input, mask=sequence_length_mask)
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
    A `SequenceNumericColumn`.

  Raises:
    TypeError: if any dimension in shape is not an int.
    ValueError: if any dimension in shape is not a positive integer.
    ValueError: if `dtype` is not convertible to `tf.float32`.
  "
  [key & {:keys [shape default_value dtype normalizer_fn]
                       :or {normalizer_fn None}} ]
    (py/call-attr-kw feature-column "sequence_numeric_column" [key] {:shape shape :default_value default_value :dtype dtype :normalizer_fn normalizer_fn }))

(defn shared-embedding-columns 
  "List of dense columns that convert from sparse, categorical input.

  This is similar to `embedding_column`, except that it produces a list of
  embedding columns that share the same embedding weights.

  Use this when your inputs are sparse and of the same type (e.g. watched and
  impression video IDs that share the same vocabulary), and you want to convert
  them to a dense representation (e.g., to feed to a DNN).

  Inputs must be a list of categorical columns created by any of the
  `categorical_column_*` function. They must all be of the same type and have
  the same arguments except `key`. E.g. they can be
  categorical_column_with_vocabulary_file with the same vocabulary_file. Some or
  all columns could also be weighted_categorical_column.

  Here is an example embedding of two features for a DNNClassifier model:

  ```python
  watched_video_id = categorical_column_with_vocabulary_file(
      'watched_video_id', video_vocabulary_file, video_vocabulary_size)
  impression_video_id = categorical_column_with_vocabulary_file(
      'impression_video_id', video_vocabulary_file, video_vocabulary_size)
  columns = shared_embedding_columns(
      [watched_video_id, impression_video_id], dimension=10)

  estimator = tf.estimator.DNNClassifier(feature_columns=columns, ...)

  label_column = ...
  def input_fn():
    features = tf.io.parse_example(
        ..., features=make_parse_example_spec(columns + [label_column]))
    labels = features.pop(label_column.name)
    return features, labels

  estimator.train(input_fn=input_fn, steps=100)
  ```

  Here is an example using `shared_embedding_columns` with model_fn:

  ```python
  def model_fn(features, ...):
    watched_video_id = categorical_column_with_vocabulary_file(
        'watched_video_id', video_vocabulary_file, video_vocabulary_size)
    impression_video_id = categorical_column_with_vocabulary_file(
        'impression_video_id', video_vocabulary_file, video_vocabulary_size)
    columns = shared_embedding_columns(
        [watched_video_id, impression_video_id], dimension=10)
    dense_tensor = input_layer(features, columns)
    # Form DNN layers, calculate loss, and return EstimatorSpec.
    ...
  ```

  Args:
    categorical_columns: List of categorical columns created by a
      `categorical_column_with_*` function. These columns produce the sparse IDs
      that are inputs to the embedding lookup. All columns must be of the same
      type and have the same arguments except `key`. E.g. they can be
      categorical_column_with_vocabulary_file with the same vocabulary_file.
      Some or all columns could also be weighted_categorical_column.
    dimension: An integer specifying dimension of the embedding, must be > 0.
    combiner: A string specifying how to reduce if there are multiple entries in
      a single row. Currently 'mean', 'sqrtn' and 'sum' are supported, with
      'mean' the default. 'sqrtn' often achieves good accuracy, in particular
      with bag-of-words columns. Each of this can be thought as example level
      normalizations on the column. For more information, see
      `tf.embedding_lookup_sparse`.
    initializer: A variable initializer function to be used in embedding
      variable initialization. If not specified, defaults to
      `truncated_normal_initializer` with mean `0.0` and
      standard deviation `1/sqrt(dimension)`.
    shared_embedding_collection_name: Optional name of the collection where
      shared embedding weights are added. If not given, a reasonable name will
      be chosen based on the names of `categorical_columns`. This is also used
      in `variable_scope` when creating shared embedding weights.
    ckpt_to_load_from: String representing checkpoint name/pattern from which to
      restore column weights. Required if `tensor_name_in_ckpt` is not `None`.
    tensor_name_in_ckpt: Name of the `Tensor` in `ckpt_to_load_from` from which
      to restore the column weights. Required if `ckpt_to_load_from` is not
      `None`.
    max_norm: If not `None`, each embedding is clipped if its l2-norm is larger
      than this value, before combining.
    trainable: Whether or not the embedding is trainable. Default is True.

  Returns:
    A list of dense columns that converts from sparse input. The order of
    results follows the ordering of `categorical_columns`.

  Raises:
    ValueError: if `dimension` not > 0.
    ValueError: if any of the given `categorical_columns` is of different type
      or has different arguments than the others.
    ValueError: if exactly one of `ckpt_to_load_from` and `tensor_name_in_ckpt`
      is specified.
    ValueError: if `initializer` is specified and is not callable.
    RuntimeError: if eager execution is enabled.
  "
  [categorical_columns dimension & {:keys [combiner initializer shared_embedding_collection_name ckpt_to_load_from tensor_name_in_ckpt max_norm trainable]
                       :or {initializer None shared_embedding_collection_name None ckpt_to_load_from None tensor_name_in_ckpt None max_norm None}} ]
    (py/call-attr-kw feature-column "shared_embedding_columns" [categorical_columns dimension] {:combiner combiner :initializer initializer :shared_embedding_collection_name shared_embedding_collection_name :ckpt_to_load_from ckpt_to_load_from :tensor_name_in_ckpt tensor_name_in_ckpt :max_norm max_norm :trainable trainable }))
(defn weighted-categorical-column 
  "Applies weight values to a `CategoricalColumn`.

  Use this when each of your sparse inputs has both an ID and a value. For
  example, if you're representing text documents as a collection of word
  frequencies, you can provide 2 parallel sparse input features ('terms' and
  'frequencies' below).

  Example:

  Input `tf.Example` objects:

  ```proto
  [
    features {
      feature {
        key: \"terms\"
        value {bytes_list {value: \"very\" value: \"model\"}}
      }
      feature {
        key: \"frequencies\"
        value {float_list {value: 0.3 value: 0.1}}
      }
    },
    features {
      feature {
        key: \"terms\"
        value {bytes_list {value: \"when\" value: \"course\" value: \"human\"}}
      }
      feature {
        key: \"frequencies\"
        value {float_list {value: 0.4 value: 0.1 value: 0.2}}
      }
    }
  ]
  ```

  ```python
  categorical_column = categorical_column_with_hash_bucket(
      column_name='terms', hash_bucket_size=1000)
  weighted_column = weighted_categorical_column(
      categorical_column=categorical_column, weight_feature_key='frequencies')
  columns = [weighted_column, ...]
  features = tf.io.parse_example(..., features=make_parse_example_spec(columns))
  linear_prediction, _, _ = linear_model(features, columns)
  ```

  This assumes the input dictionary contains a `SparseTensor` for key
  'terms', and a `SparseTensor` for key 'frequencies'. These 2 tensors must have
  the same indices and dense shape.

  Args:
    categorical_column: A `CategoricalColumn` created by
      `categorical_column_with_*` functions.
    weight_feature_key: String key for weight values.
    dtype: Type of weights, such as `tf.float32`. Only float and integer weights
      are supported.

  Returns:
    A `CategoricalColumn` composed of two sparse features: one represents id,
    the other represents weight (value) of the id feature in that example.

  Raises:
    ValueError: if `dtype` is not convertible to float.
  "
  [categorical_column weight_feature_key  & {:keys [dtype]} ]
    (py/call-attr-kw feature-column "weighted_categorical_column" [categorical_column weight_feature_key] {:dtype dtype }))
