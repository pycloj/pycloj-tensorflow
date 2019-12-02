(ns tensorflow.contrib.layers.python.layers.feature-column
  "This API defines FeatureColumn abstraction.

FeatureColumns provide a high level abstraction for ingesting and representing
features in `Estimator` models.

FeatureColumns are the primary way of encoding features for pre-canned
`Estimator` models.

When using FeatureColumns with `Estimator` models, the type of feature column
you should choose depends on (1) the feature type and (2) the model type.

(1) Feature type:

 * Continuous features can be represented by `real_valued_column`.
 * Categorical features can be represented by any `sparse_column_with_*`
 column (`sparse_column_with_keys`, `sparse_column_with_vocabulary_file`,
 `sparse_column_with_hash_bucket`, `sparse_column_with_integerized_feature`).

(2) Model type:

 * Deep neural network models (`DNNClassifier`, `DNNRegressor`).

   Continuous features can be directly fed into deep neural network models.

     age_column = real_valued_column(\"age\")

   To feed sparse features into DNN models, wrap the column with
   `embedding_column` or `one_hot_column`. `one_hot_column` will create a dense
   boolean tensor with an entry for each possible value, and thus the
   computation cost is linear in the number of possible values versus the number
   of values that occur in the sparse tensor. Thus using a \"one_hot_column\" is
   only recommended for features with only a few possible values. For features
   with many possible values or for very sparse features, `embedding_column` is
   recommended.

     embedded_dept_column = embedding_column(
       sparse_column_with_keys(\"department\", [\"math\", \"philosophy\", ...]),
       dimension=10)

* Wide (aka linear) models (`LinearClassifier`, `LinearRegressor`).

   Sparse features can be fed directly into linear models. When doing so
   an embedding_lookups are used to efficiently perform the sparse matrix
   multiplication.

     dept_column = sparse_column_with_keys(\"department\",
       [\"math\", \"philosophy\", \"english\"])

   It is recommended that continuous features be bucketized before being
   fed into linear models.

     bucketized_age_column = bucketized_column(
      source_column=age_column,
      boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

   Sparse features can be crossed (also known as conjuncted or combined) in
   order to form non-linearities, and then fed into linear models.

    cross_dept_age_column = crossed_column(
      columns=[department_column, bucketized_age_column],
      hash_bucket_size=1000)

Example of building an `Estimator` model using FeatureColumns:

  # Define features and transformations
  deep_feature_columns = [age_column, embedded_dept_column]
  wide_feature_columns = [dept_column, bucketized_age_column,
      cross_dept_age_column]

  # Build deep model
  estimator = DNNClassifier(
      feature_columns=deep_feature_columns,
      hidden_units=[500, 250, 50])
  estimator.train(...)

  # Or build a wide model
  estimator = LinearClassifier(
      feature_columns=wide_feature_columns)
  estimator.train(...)

  # Or build a wide and deep model!
  estimator = DNNLinearCombinedClassifier(
      linear_feature_columns=wide_feature_columns,
      dnn_feature_columns=deep_feature_columns,
      dnn_hidden_units=[500, 250, 50])
  estimator.train(...)


FeatureColumns can also be transformed into a generic input layer for
custom models using `input_from_feature_columns` within
`feature_column_ops.py`.

Example of building a non-`Estimator` model using FeatureColumns:

  # Building model via layers

  deep_feature_columns = [age_column, embedded_dept_column]
  columns_to_tensor = parse_feature_columns_from_examples(
      serialized=my_data,
      feature_columns=deep_feature_columns)
  first_layer = input_from_feature_columns(
      columns_to_tensors=columns_to_tensor,
      feature_columns=deep_feature_columns)
  second_layer = fully_connected(first_layer, ...)

See feature_column_ops_test for more examples.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce feature-column (import-module "tensorflow.contrib.layers.python.layers.feature_column"))

(defn bucketized-column 
  "Creates a _BucketizedColumn for discretizing dense input.

  Args:
    source_column: A _RealValuedColumn defining dense column.
    boundaries: A list or tuple of floats specifying the boundaries. It has to
      be sorted.

  Returns:
    A _BucketizedColumn.

  Raises:
    ValueError: if 'boundaries' is empty or not sorted.
  "
  [ source_column boundaries ]
  (py/call-attr feature-column "bucketized_column"  source_column boundaries ))

(defn create-feature-spec-for-parsing 
  "Helper that prepares features config from input feature_columns.

  The returned feature config can be used as arg 'features' in tf.parse_example.

  Typical usage example:

  ```python
  # Define features and transformations
  feature_a = sparse_column_with_vocabulary_file(...)
  feature_b = real_valued_column(...)
  feature_c_bucketized = bucketized_column(real_valued_column(\"feature_c\"), ...)
  feature_a_x_feature_c = crossed_column(
    columns=[feature_a, feature_c_bucketized], ...)

  feature_columns = set(
    [feature_b, feature_c_bucketized, feature_a_x_feature_c])
  batch_examples = tf.io.parse_example(
      serialized=serialized_examples,
      features=create_feature_spec_for_parsing(feature_columns))
  ```

  For the above example, create_feature_spec_for_parsing would return the dict:
  {
    \"feature_a\": parsing_ops.VarLenFeature(tf.string),
    \"feature_b\": parsing_ops.FixedLenFeature([1], dtype=tf.float32),
    \"feature_c\": parsing_ops.FixedLenFeature([1], dtype=tf.float32)
  }

  Args:
    feature_columns: An iterable containing all the feature columns. All items
      should be instances of classes derived from _FeatureColumn, unless
      feature_columns is a dict -- in which case, this should be true of all
      values in the dict.

  Returns:
    A dict mapping feature keys to FixedLenFeature or VarLenFeature values.
  "
  [ feature_columns ]
  (py/call-attr feature-column "create_feature_spec_for_parsing"  feature_columns ))

(defn crossed-column 
  "Creates a _CrossedColumn for performing feature crosses.

  Args:
    columns: An iterable of _FeatureColumn. Items can be an instance of
      _SparseColumn, _CrossedColumn, or _BucketizedColumn.
    hash_bucket_size: An int that is > 1. The number of buckets.
    combiner: A string specifying how to reduce if there are multiple entries in
      a single row. Currently \"mean\", \"sqrtn\" and \"sum\" are supported, with
      \"sum\" the default. \"sqrtn\" often achieves good accuracy, in particular
      with bag-of-words columns. Each of this can be thought as example level
      normalizations on the column::
        * \"sum\": do not normalize
        * \"mean\": do l1 normalization
        * \"sqrtn\": do l2 normalization
      For more information: `tf.embedding_lookup_sparse`.
    ckpt_to_load_from: (Optional). String representing checkpoint name/pattern
      to restore the column weights. Required if `tensor_name_in_ckpt` is not
      None.
    tensor_name_in_ckpt: (Optional). Name of the `Tensor` in the provided
      checkpoint from which to restore the column weights. Required if
      `ckpt_to_load_from` is not None.
    hash_key: Specify the hash_key that will be used by the `FingerprintCat64`
      function to combine the crosses fingerprints on SparseFeatureCrossOp
      (optional).

  Returns:
    A _CrossedColumn.

  Raises:
    TypeError: if any item in columns is not an instance of _SparseColumn,
      _CrossedColumn, or _BucketizedColumn, or
      hash_bucket_size is not an int.
    ValueError: if hash_bucket_size is not > 1 or
      len(columns) is not > 1.
  "
  [columns hash_bucket_size & {:keys [combiner ckpt_to_load_from tensor_name_in_ckpt hash_key]
                       :or {ckpt_to_load_from None tensor_name_in_ckpt None hash_key None}} ]
    (py/call-attr-kw feature-column "crossed_column" [columns hash_bucket_size] {:combiner combiner :ckpt_to_load_from ckpt_to_load_from :tensor_name_in_ckpt tensor_name_in_ckpt :hash_key hash_key }))

(defn embedding-column 
  "Creates an `_EmbeddingColumn` for feeding sparse data into a DNN.

  Args:
    sparse_id_column: A `_SparseColumn` which is created by for example
      `sparse_column_with_*` or crossed_column functions. Note that `combiner`
      defined in `sparse_id_column` is ignored.
    dimension: An integer specifying dimension of the embedding.
    combiner: A string specifying how to reduce if there are multiple entries in
      a single row. Currently \"mean\", \"sqrtn\" and \"sum\" are supported, with
      \"mean\" the default. \"sqrtn\" often achieves good accuracy, in particular
      with bag-of-words columns. Each of this can be thought as example level
      normalizations on the column:
        * \"sum\": do not normalize
        * \"mean\": do l1 normalization
        * \"sqrtn\": do l2 normalization
      For more information: `tf.embedding_lookup_sparse`.
    initializer: A variable initializer function to be used in embedding
      variable initialization. If not specified, defaults to
      `tf.compat.v1.truncated_normal_initializer` with mean 0.0 and standard
      deviation 1/sqrt(sparse_id_column.length).
    ckpt_to_load_from: (Optional). String representing checkpoint name/pattern
      to restore the column weights. Required if `tensor_name_in_ckpt` is not
      None.
    tensor_name_in_ckpt: (Optional). Name of the `Tensor` in the provided
      checkpoint from which to restore the column weights. Required if
      `ckpt_to_load_from` is not None.
    max_norm: (Optional). If not None, embedding values are l2-normalized to the
      value of max_norm.
    trainable: (Optional). Should the embedding be trainable. Default is True

  Returns:
    An `_EmbeddingColumn`.
  "
  [sparse_id_column dimension & {:keys [combiner initializer ckpt_to_load_from tensor_name_in_ckpt max_norm trainable]
                       :or {initializer None ckpt_to_load_from None tensor_name_in_ckpt None max_norm None}} ]
    (py/call-attr-kw feature-column "embedding_column" [sparse_id_column dimension] {:combiner combiner :initializer initializer :ckpt_to_load_from ckpt_to_load_from :tensor_name_in_ckpt tensor_name_in_ckpt :max_norm max_norm :trainable trainable }))

(defn experimental 
  "Decorator for marking functions or methods experimental.

  This decorator logs an experimental warning whenever the decorated function is
  called. It has the following format:

    <function> (from <module>) is experimental and may change or be removed at
    any time, and without warning.

  <function> will include the class name if it is a method.

  It also edits the docstring of the function: ' (experimental)' is appended
  to the first line of the docstring and a notice is prepended to the rest of
  the docstring.

  Args:
    func: A function or method to mark experimental.

  Returns:
    Decorated function or method.
  "
  [ func ]
  (py/call-attr feature-column "experimental"  func ))

(defn make-place-holder-tensors-for-base-features 
  "Returns placeholder tensors for inference.

  Args:
    feature_columns: An iterable containing all the feature columns. All items
      should be instances of classes derived from _FeatureColumn.

  Returns:
    A dict mapping feature keys to SparseTensors (sparse columns) or
    placeholder Tensors (dense columns).
  "
  [ feature_columns ]
  (py/call-attr feature-column "make_place_holder_tensors_for_base_features"  feature_columns ))

(defn one-hot-column 
  "Creates an `_OneHotColumn` for a one-hot or multi-hot repr in a DNN.

  Args:
      sparse_id_column: A _SparseColumn which is created by
        `sparse_column_with_*` or crossed_column functions. Note that `combiner`
        defined in `sparse_id_column` is ignored.

  Returns:
    An _OneHotColumn.
  "
  [ sparse_id_column ]
  (py/call-attr feature-column "one_hot_column"  sparse_id_column ))

(defn real-valued-column 
  "Creates a `_RealValuedColumn` for dense numeric data.

  Args:
    column_name: A string defining real valued column name.
    dimension: An integer specifying dimension of the real valued column. The
      default is 1.
    default_value: A single value compatible with dtype or a list of values
      compatible with dtype which the column takes on during tf.Example parsing
      if data is missing. When dimension is not None, a default value of None
      will cause tf.io.parse_example to fail if an example does not contain this
      column. If a single value is provided, the same value will be applied as
      the default value for every dimension. If a list of values is provided,
      the length of the list should be equal to the value of `dimension`. Only
      scalar default value is supported in case dimension is not specified.
    dtype: defines the type of values. Default value is tf.float32. Must be a
      non-quantized, real integer or floating point type.
    normalizer: If not None, a function that can be used to normalize the value
      of the real valued column after default_value is applied for parsing.
      Normalizer function takes the input tensor as its argument, and returns
      the output tensor. (e.g. lambda x: (x - 3.0) / 4.2). Note that for
        variable length columns, the normalizer should expect an input_tensor of
        type `SparseTensor`.

  Returns:
    A _RealValuedColumn.
  Raises:
    TypeError: if dimension is not an int
    ValueError: if dimension is not a positive integer
    TypeError: if default_value is a list but its length is not equal to the
      value of `dimension`.
    TypeError: if default_value is not compatible with dtype.
    ValueError: if dtype is not convertible to tf.float32.
  "
  [column_name & {:keys [dimension default_value dtype normalizer]
                       :or {default_value None normalizer None}} ]
    (py/call-attr-kw feature-column "real_valued_column" [column_name] {:dimension dimension :default_value default_value :dtype dtype :normalizer normalizer }))

(defn scattered-embedding-column 
  "Creates an embedding column of a sparse feature using parameter hashing.

  This is a useful shorthand when you have a sparse feature you want to use an
  embedding for, but also want to hash the embedding's values in each dimension
  to a variable based on a different hash.

  Specifically, the i-th embedding component of a value v is found by retrieving
  an embedding weight whose index is a fingerprint of the pair (v,i).

  An embedding column with sparse_column_with_hash_bucket such as

      embedding_column(
        sparse_column_with_hash_bucket(column_name, bucket_size),
        dimension)

  could be replaced by

      scattered_embedding_column(
        column_name,
        size=bucket_size * dimension,
        dimension=dimension,
        hash_key=tf.contrib.layers.SPARSE_FEATURE_CROSS_DEFAULT_HASH_KEY)

  for the same number of embedding parameters. This should hopefully reduce the
  impact of collisions, but adds the cost of slowing down training.

  Args:
    column_name: A string defining sparse column name.
    size: An integer specifying the number of parameters in the embedding layer.
    dimension: An integer specifying dimension of the embedding.
    hash_key: Specify the hash_key that will be used by the `FingerprintCat64`
      function to combine the crosses fingerprints on SparseFeatureCrossOp.
    combiner: A string specifying how to reduce if there are multiple entries in
      a single row. Currently \"mean\", \"sqrtn\" and \"sum\" are supported, with
      \"mean\" the default. \"sqrtn\" often achieves good accuracy, in particular
      with bag-of-words columns. Each of this can be thought as example level
      normalizations on the column:
        * \"sum\": do not normalize features in the column
        * \"mean\": do l1 normalization on features in the column
        * \"sqrtn\": do l2 normalization on features in the column
      For more information: `tf.embedding_lookup_sparse`.
    initializer: A variable initializer function to be used in embedding
      variable initialization. If not specified, defaults to
      `tf.compat.v1.truncated_normal_initializer` with mean 0 and standard
      deviation 0.1.

  Returns:
    A _ScatteredEmbeddingColumn.

  Raises:
    ValueError: if dimension or size is not a positive integer; or if combiner
      is not supported.

  "
  [column_name size dimension hash_key & {:keys [combiner initializer]
                       :or {initializer None}} ]
    (py/call-attr-kw feature-column "scattered_embedding_column" [column_name size dimension hash_key] {:combiner combiner :initializer initializer }))

(defn shared-embedding-columns 
  "Creates a list of `_EmbeddingColumn` sharing the same embedding.

  Args:
    sparse_id_columns: An iterable of `_SparseColumn`, such as those created by
      `sparse_column_with_*` or crossed_column functions. Note that `combiner`
      defined in each sparse_id_column is ignored.
    dimension: An integer specifying dimension of the embedding.
    combiner: A string specifying how to reduce if there are multiple entries in
      a single row. Currently \"mean\", \"sqrtn\" and \"sum\" are supported, with
      \"mean\" the default. \"sqrtn\" often achieves good accuracy, in particular
      with bag-of-words columns. Each of this can be thought as example level
      normalizations on the column:
        * \"sum\": do not normalize
        * \"mean\": do l1 normalization
        * \"sqrtn\": do l2 normalization
      For more information: `tf.embedding_lookup_sparse`.
    shared_embedding_name: (Optional). A string specifying the name of shared
      embedding weights. This will be needed if you want to reference the shared
      embedding separately from the generated `_EmbeddingColumn`.
    initializer: A variable initializer function to be used in embedding
      variable initialization. If not specified, defaults to
      `tf.compat.v1.truncated_normal_initializer` with mean 0.0 and standard
      deviation 1/sqrt(sparse_id_columns[0].length).
    ckpt_to_load_from: (Optional). String representing checkpoint name/pattern
      to restore the column weights. Required if `tensor_name_in_ckpt` is not
      None.
    tensor_name_in_ckpt: (Optional). Name of the `Tensor` in the provided
      checkpoint from which to restore the column weights. Required if
      `ckpt_to_load_from` is not None.
    max_norm: (Optional). If not None, embedding values are l2-normalized to the
      value of max_norm.
    trainable: (Optional). Should the embedding be trainable. Default is True

  Returns:
    A tuple of `_EmbeddingColumn` with shared embedding space.

  Raises:
    ValueError: if sparse_id_columns is empty, or its elements are not
      compatible with each other.
    TypeError: if `sparse_id_columns` is not a sequence or is a string. If at
      least one element of `sparse_id_columns` is not a `SparseColumn` or a
      `WeightedSparseColumn`.
  "
  [sparse_id_columns dimension & {:keys [combiner shared_embedding_name initializer ckpt_to_load_from tensor_name_in_ckpt max_norm trainable]
                       :or {shared_embedding_name None initializer None ckpt_to_load_from None tensor_name_in_ckpt None max_norm None}} ]
    (py/call-attr-kw feature-column "shared_embedding_columns" [sparse_id_columns dimension] {:combiner combiner :shared_embedding_name shared_embedding_name :initializer initializer :ckpt_to_load_from ckpt_to_load_from :tensor_name_in_ckpt tensor_name_in_ckpt :max_norm max_norm :trainable trainable }))

(defn sparse-column-with-hash-bucket 
  "Creates a _SparseColumn with hashed bucket configuration.

  Use this when your sparse features are in string or integer format, but you
  don't have a vocab file that maps each value to an integer ID.
  output_id = Hash(input_feature_string) % bucket_size

  When hash_keys is set, multiple integer IDs would be created with each key
  pair in the `hash_keys`. This is useful to reduce the collision of hashed ids.

  Args:
    column_name: A string defining sparse column name.
    hash_bucket_size: An int that is > 1. The number of buckets.
    combiner: A string specifying how to reduce if the sparse column is
      multivalent. Currently \"mean\", \"sqrtn\" and \"sum\" are supported, with \"sum\"
      the default. \"sqrtn\" often achieves good accuracy, in particular with
      bag-of-words columns.
        * \"sum\": do not normalize features in the column
        * \"mean\": do l1 normalization on features in the column
        * \"sqrtn\": do l2 normalization on features in the column
      For more information: `tf.embedding_lookup_sparse`.
    dtype: The type of features. Only string and integer types are supported.
    hash_keys: The hash keys to use. It is a list of lists of two uint64s. If
      None, simple and fast hashing algorithm is used. Otherwise, multiple
      strong hash ids would be produced with each two uint64s in this argument.

  Returns:
    A _SparseColumn with hashed bucket configuration

  Raises:
    ValueError: hash_bucket_size is not greater than 2.
    ValueError: dtype is neither string nor integer.
  "
  [column_name hash_bucket_size & {:keys [combiner dtype hash_keys]
                       :or {hash_keys None}} ]
    (py/call-attr-kw feature-column "sparse_column_with_hash_bucket" [column_name hash_bucket_size] {:combiner combiner :dtype dtype :hash_keys hash_keys }))
(defn sparse-column-with-integerized-feature 
  "Creates an integerized _SparseColumn.

  Use this when your features are already pre-integerized into int64 IDs, that
  is, when the set of values to output is already coming in as what's desired in
  the output. Integerized means we can use the feature value itself as id.

  Typically this is used for reading contiguous ranges of integers indexes, but
  it doesn't have to be. The output value is simply copied from the
  input_feature, whatever it is. Just be aware, however, that if you have large
  gaps of unused integers it might affect what you feed those in (for instance,
  if you make up a one-hot tensor from these, the unused integers will appear as
  values in the tensor which are always zero.)

  Args:
    column_name: A string defining sparse column name.
    bucket_size: An int that is >= 1. The number of buckets. It should be bigger
      than maximum feature. In other words features in this column should be an
      int64 in range [0, bucket_size)
    combiner: A string specifying how to reduce if the sparse column is
      multivalent. Currently \"mean\", \"sqrtn\" and \"sum\" are supported, with \"sum\"
      the default. \"sqrtn\" often achieves good accuracy, in particular with
      bag-of-words columns.
        * \"sum\": do not normalize features in the column
        * \"mean\": do l1 normalization on features in the column
        * \"sqrtn\": do l2 normalization on features in the column
      For more information: `tf.embedding_lookup_sparse`.
    dtype: Type of features. It should be an integer type. Default value is
      dtypes.int64.

  Returns:
    An integerized _SparseColumn definition.

  Raises:
    ValueError: bucket_size is less than 1.
    ValueError: dtype is not integer.
  "
  [column_name bucket_size  & {:keys [combiner dtype]} ]
    (py/call-attr-kw feature-column "sparse_column_with_integerized_feature" [column_name bucket_size] {:combiner combiner :dtype dtype }))
(defn sparse-column-with-keys 
  "Creates a _SparseColumn with keys.

  Look up logic is as follows:
  lookup_id = index_of_feature_in_keys if feature in keys else default_value

  Args:
    column_name: A string defining sparse column name.
    keys: A list or tuple defining vocabulary. Must be castable to `dtype`.
    default_value: The value to use for out-of-vocabulary feature values.
      Default is -1.
    combiner: A string specifying how to reduce if the sparse column is
      multivalent. Currently \"mean\", \"sqrtn\" and \"sum\" are supported, with \"sum\"
      the default. \"sqrtn\" often achieves good accuracy, in particular with
      bag-of-words columns.
        * \"sum\": do not normalize features in the column
        * \"mean\": do l1 normalization on features in the column
        * \"sqrtn\": do l2 normalization on features in the column
      For more information: `tf.embedding_lookup_sparse`.
    dtype: Type of features. Only integer and string are supported.

  Returns:
    A _SparseColumnKeys with keys configuration.
  "
  [column_name keys  & {:keys [default_value combiner dtype]} ]
    (py/call-attr-kw feature-column "sparse_column_with_keys" [column_name keys] {:default_value default_value :combiner combiner :dtype dtype }))

(defn sparse-column-with-vocabulary-file 
  "Creates a _SparseColumn with vocabulary file configuration.

  Use this when your sparse features are in string or integer format, and you
  have a vocab file that maps each value to an integer ID.
  output_id = LookupIdFromVocab(input_feature_string)

  Args:
    column_name: A string defining sparse column name.
    vocabulary_file: The vocabulary filename.
    num_oov_buckets: The number of out-of-vocabulary buckets. If zero all out of
      vocabulary features will be ignored.
    vocab_size: Number of the elements in the vocabulary.
    default_value: The value to use for out-of-vocabulary feature values.
      Defaults to -1.
    combiner: A string specifying how to reduce if the sparse column is
      multivalent. Currently \"mean\", \"sqrtn\" and \"sum\" are supported, with \"sum\"
      the default. \"sqrtn\" often achieves good accuracy, in particular with
      bag-of-words columns.
        * \"sum\": do not normalize features in the column
        * \"mean\": do l1 normalization on features in the column
        * \"sqrtn\": do l2 normalization on features in the column
      For more information: `tf.embedding_lookup_sparse`.
    dtype: The type of features. Only string and integer types are supported.

  Returns:
    A _SparseColumn with vocabulary file configuration.

  Raises:
    ValueError: vocab_size is not defined.
    ValueError: dtype is neither string nor integer.
  "
  [column_name vocabulary_file & {:keys [num_oov_buckets vocab_size default_value combiner dtype]
                       :or {vocab_size None}} ]
    (py/call-attr-kw feature-column "sparse_column_with_vocabulary_file" [column_name vocabulary_file] {:num_oov_buckets num_oov_buckets :vocab_size vocab_size :default_value default_value :combiner combiner :dtype dtype }))
(defn weighted-sparse-column 
  "Creates a _SparseColumn by combining sparse_id_column with a weight column.

  Example:

    ```python
    sparse_feature = sparse_column_with_hash_bucket(column_name=\"sparse_col\",
                                                    hash_bucket_size=1000)
    weighted_feature = weighted_sparse_column(sparse_id_column=sparse_feature,
                                              weight_column_name=\"weights_col\")
    ```

    This configuration assumes that input dictionary of model contains the
    following two items:
      * (key=\"sparse_col\", value=sparse_tensor) where sparse_tensor is
        a SparseTensor.
      * (key=\"weights_col\", value=weights_tensor) where weights_tensor
        is a SparseTensor.
     Following are assumed to be true:
       * sparse_tensor.indices = weights_tensor.indices
       * sparse_tensor.dense_shape = weights_tensor.dense_shape

  Args:
    sparse_id_column: A `_SparseColumn` which is created by
      `sparse_column_with_*` functions.
    weight_column_name: A string defining a sparse column name which represents
      weight or value of the corresponding sparse id feature.
    dtype: Type of weights, such as `tf.float32`. Only floating and integer
      weights are supported.

  Returns:
    A _WeightedSparseColumn composed of two sparse features: one represents id,
    the other represents weight (value) of the id feature in that example.

  Raises:
    ValueError: if dtype is not convertible to float.
  "
  [sparse_id_column weight_column_name  & {:keys [dtype]} ]
    (py/call-attr-kw feature-column "weighted_sparse_column" [sparse_id_column weight_column_name] {:dtype dtype }))
