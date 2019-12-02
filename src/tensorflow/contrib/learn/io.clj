(ns tensorflow.contrib.learn.python.learn.learn-io
  "Tools to allow different io formats (deprecated).

This module and all its submodules are deprecated. See
[contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
for migration instructions.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce io (import-module "tensorflow.contrib.learn.python.learn.learn_io"))

(defn extract-dask-data 
  "Extract data from dask.Series or dask.DataFrame for predictors. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please feed input to tf.data to support dask.

Given a distributed dask.DataFrame or dask.Series containing columns or names
for one or more predictors, this operation returns a single dask.DataFrame or
dask.Series that can be iterated over.

Args:
  data: A distributed dask.DataFrame or dask.Series.

Returns:
  A dask.DataFrame or dask.Series that can be iterated over.
  If the supplied argument is neither a dask.DataFrame nor a dask.Series this
  operation returns it without modification."
  [ data ]
  (py/call-attr io "extract_dask_data"  data ))

(defn extract-dask-labels 
  "Extract data from dask.Series or dask.DataFrame for labels. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please feed input to tf.data to support dask.

Given a distributed dask.DataFrame or dask.Series containing exactly one
column or name, this operation returns a single dask.DataFrame or dask.Series
that can be iterated over.

Args:
  labels: A distributed dask.DataFrame or dask.Series with exactly one
          column or name.

Returns:
  A dask.DataFrame or dask.Series that can be iterated over.
  If the supplied argument is neither a dask.DataFrame nor a dask.Series this
  operation returns it without modification.

Raises:
  ValueError: If the supplied dask.DataFrame contains more than one
              column or the supplied dask.Series contains more than
              one name."
  [ labels ]
  (py/call-attr io "extract_dask_labels"  labels ))

(defn extract-pandas-data 
  "Extract data from pandas.DataFrame for predictors. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please access pandas data directly.

Given a DataFrame, will extract the values and cast them to float. The
DataFrame is expected to contain values of type int, float or bool.

Args:
  data: `pandas.DataFrame` containing the data to be extracted.

Returns:
  A numpy `ndarray` of the DataFrame's values as floats.

Raises:
  ValueError: if data contains types other than int, float or bool."
  [ data ]
  (py/call-attr io "extract_pandas_data"  data ))

(defn extract-pandas-labels 
  "Extract data from pandas.DataFrame for labels. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please access pandas data directly.

Args:
  labels: `pandas.DataFrame` or `pandas.Series` containing one column of
    labels to be extracted.

Returns:
  A numpy `ndarray` of labels from the DataFrame.

Raises:
  ValueError: if more than one column is found or type is not int, float or
    bool."
  [ labels ]
  (py/call-attr io "extract_pandas_labels"  labels ))

(defn extract-pandas-matrix 
  "Extracts numpy matrix from pandas DataFrame. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please access pandas data directly.

Args:
  data: `pandas.DataFrame` containing the data to be extracted.

Returns:
  A numpy `ndarray` of the DataFrame's values."
  [ data ]
  (py/call-attr io "extract_pandas_matrix"  data ))

(defn generator-input-fn 
  "Returns input function that returns dicts of numpy arrays (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please use tf.data.

   yielded from a generator.

It is assumed that every dict of numpy arrays yielded from the dictionary
represents a single sample. The generator should consume a single epoch of the
data.

This returns a function outputting `features` and `target` based on the dict
of numpy arrays. The dict `features` has the same keys as an element yielded
from x.

Example:
  ```python
  def generator():
    for index in range(10):
      yield {'height': np.random.randint(32,36),
            'age': np.random.randint(18, 80),
            'label': np.ones(1)}

  with tf.compat.v1.Session() as session:
    input_fn = generator_io.generator_input_fn(
        generator, target_key=\"label\", batch_size=2, shuffle=False,
        num_epochs=1)
  ```

Args:
  x: Generator Function, returns a `Generator` that will yield the data
    in `dict` of numpy arrays
  target_key: String or Container of Strings, the key or Container of keys of
    the numpy arrays in x dictionaries to use as target.
  batch_size: Integer, size of batches to return.
  num_epochs: Integer, number of epochs to iterate over data. If `None` will
    run forever.
  shuffle: Boolean, if True shuffles the queue. Avoid shuffle at prediction
    time.
  queue_capacity: Integer, size of queue to accumulate.
  num_threads: Integer, number of threads used for reading and enqueueing.
  pad_value: default value for dynamic padding of data samples, if provided.

Returns:
  Function, that returns a feature `dict` with `Tensors` and an optional
   label `dict` with `Tensors`, or if target_key is `str` label is a `Tensor`

Raises:
  TypeError: `x` is not `FunctionType`.
  TypeError: `x()` is not `GeneratorType`.
  TypeError: `next(x())` is not `dict`.
  TypeError: `target_key` is not `str` or `target_key` is not `Container`
     of `str`.
  KeyError:  `target_key` not a key or `target_key[index]` not in next(`x()`).
  KeyError: `key` mismatch between dicts emitted from `x()`"
  [x target_key & {:keys [batch_size num_epochs shuffle queue_capacity num_threads pad_value]
                       :or {pad_value None}} ]
    (py/call-attr-kw io "generator_input_fn" [x target_key] {:batch_size batch_size :num_epochs num_epochs :shuffle shuffle :queue_capacity queue_capacity :num_threads num_threads :pad_value pad_value }))
(defn numpy-input-fn 
  "This input_fn diffs from the core version with default `shuffle`. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use tf.compat.v1.estimator.inputs.numpy_input_fn."
  [x y  & {:keys [batch_size num_epochs shuffle queue_capacity num_threads]} ]
    (py/call-attr-kw io "numpy_input_fn" [x y] {:batch_size batch_size :num_epochs num_epochs :shuffle shuffle :queue_capacity queue_capacity :num_threads num_threads }))
(defn pandas-input-fn 
  "This input_fn diffs from the core version with default `shuffle`. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please use tf.compat.v1.estimator.inputs.pandas_input_fn"
  [x y  & {:keys [batch_size num_epochs shuffle queue_capacity num_threads target_column]} ]
    (py/call-attr-kw io "pandas_input_fn" [x y] {:batch_size batch_size :num_epochs num_epochs :shuffle shuffle :queue_capacity queue_capacity :num_threads num_threads :target_column target_column }))

(defn queue-parsed-features 
  "Speeds up parsing by using queues to do it asynchronously. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use tf.data.

This function adds the tensors in `parsed_features` to a queue, which allows
the parsing (or any other expensive op before this) to be asynchronous wrt the
rest of the training graph. This greatly improves read latency and speeds up
training since the data will already be parsed and ready when each step of
training needs it.

All queue runners are added to the queue runners collection, and may be
started via `start_queue_runners`.

All ops are added to the default graph.

Args:
  parsed_features: A dict of string key to `Tensor` or `SparseTensor` objects.
  keys: `Tensor` of string keys.
  feature_queue_capacity: Capacity of the parsed features queue.
  num_enqueue_threads: Number of threads to enqueue the parsed example queue.
    Using multiple threads to enqueue the parsed example queue helps maintain
    a full queue when the subsequent computations overall are cheaper than
    parsing. In order to have predictable and repeatable order of reading and
    enqueueing, such as in prediction and evaluation mode,
    `num_enqueue_threads` should be 1.
  name: Name of resulting op.

Returns:
  Returns tuple of:
  - `Tensor` corresponding to `keys` if provided, otherwise `None`.
  -  A dict of string key to `Tensor` or `SparseTensor` objects corresponding
     to `parsed_features`.
Raises:
  ValueError: for invalid inputs."
  [parsed_features keys & {:keys [feature_queue_capacity num_enqueue_threads name]
                       :or {name None}} ]
    (py/call-attr-kw io "queue_parsed_features" [parsed_features keys] {:feature_queue_capacity feature_queue_capacity :num_enqueue_threads num_enqueue_threads :name name }))

(defn read-batch-examples 
  "Adds operations to read, queue, batch `Example` protos. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use tf.data.

Given file pattern (or list of files), will setup a queue for file names,
read `Example` proto using provided `reader`, use batch queue to create
batches of examples of size `batch_size`.

All queue runners are added to the queue runners collection, and may be
started via `start_queue_runners`.

All ops are added to the default graph.

Use `parse_fn` if you need to do parsing / processing on single examples.

Args:
  file_pattern: List of files or patterns of file paths containing
      `Example` records. See `tf.io.gfile.glob` for pattern rules.
  batch_size: An int or scalar `Tensor` specifying the batch size to use.
  reader: A function or class that returns an object with
    `read` method, (filename tensor) -> (example tensor).
  randomize_input: Whether the input should be randomized.
  num_epochs: Integer specifying the number of times to read through the
    dataset. If `None`, cycles through the dataset forever.
    NOTE - If specified, creates a variable that must be initialized, so call
    `tf.compat.v1.local_variables_initializer()` and run the op in a session.
  queue_capacity: Capacity for input queue.
  num_threads: The number of threads enqueuing examples. In order to have
    predictable and repeatable order of reading and enqueueing, such as in
    prediction and evaluation mode, `num_threads` should be 1.
  read_batch_size: An int or scalar `Tensor` specifying the number of
    records to read at once.
  parse_fn: Parsing function, takes `Example` Tensor returns parsed
    representation. If `None`, no parsing is done.
  name: Name of resulting op.
  seed: An integer (optional). Seed used if randomize_input == True.

Returns:
  String `Tensor` of batched `Example` proto.

Raises:
  ValueError: for invalid inputs."
  [file_pattern batch_size reader & {:keys [randomize_input num_epochs queue_capacity num_threads read_batch_size parse_fn name seed]
                       :or {num_epochs None parse_fn None name None seed None}} ]
    (py/call-attr-kw io "read_batch_examples" [file_pattern batch_size reader] {:randomize_input randomize_input :num_epochs num_epochs :queue_capacity queue_capacity :num_threads num_threads :read_batch_size read_batch_size :parse_fn parse_fn :name name :seed seed }))

(defn read-batch-features 
  "Adds operations to read, queue, batch and parse `Example` protos. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use tf.data.

Given file pattern (or list of files), will setup a queue for file names,
read `Example` proto using provided `reader`, use batch queue to create
batches of examples of size `batch_size` and parse example given `features`
specification.

All queue runners are added to the queue runners collection, and may be
started via `start_queue_runners`.

All ops are added to the default graph.

Args:
  file_pattern: List of files or patterns of file paths containing
      `Example` records. See `tf.io.gfile.glob` for pattern rules.
  batch_size: An int or scalar `Tensor` specifying the batch size to use.
  features: A `dict` mapping feature keys to `FixedLenFeature` or
    `VarLenFeature` values.
  reader: A function or class that returns an object with
    `read` method, (filename tensor) -> (example tensor).
  randomize_input: Whether the input should be randomized.
  num_epochs: Integer specifying the number of times to read through the
    dataset. If None, cycles through the dataset forever. NOTE - If specified,
    creates a variable that must be initialized, so call
    tf.compat.v1.local_variables_initializer() and run the op in a session.
  queue_capacity: Capacity for input queue.
  feature_queue_capacity: Capacity of the parsed features queue. Set this
    value to a small number, for example 5 if the parsed features are large.
  reader_num_threads: The number of threads to read examples. In order to have
    predictable and repeatable order of reading and enqueueing, such as in
    prediction and evaluation mode, `reader_num_threads` should be 1.
  num_enqueue_threads: Number of threads to enqueue the parsed example queue.
    Using multiple threads to enqueue the parsed example queue helps maintain
    a full queue when the subsequent computations overall are cheaper than
    parsing. In order to have predictable and repeatable order of reading and
    enqueueing, such as in prediction and evaluation mode,
    `num_enqueue_threads` should be 1.
  parse_fn: Parsing function, takes `Example` Tensor returns parsed
    representation. If `None`, no parsing is done.
  name: Name of resulting op.
  read_batch_size: An int or scalar `Tensor` specifying the number of
    records to read at once. If `None`, defaults to `batch_size`.

Returns:
  A dict of `Tensor` or `SparseTensor` objects for each in `features`.

Raises:
  ValueError: for invalid inputs."
  [file_pattern batch_size features reader & {:keys [randomize_input num_epochs queue_capacity feature_queue_capacity reader_num_threads num_enqueue_threads parse_fn name read_batch_size]
                       :or {num_epochs None parse_fn None name None read_batch_size None}} ]
    (py/call-attr-kw io "read_batch_features" [file_pattern batch_size features reader] {:randomize_input randomize_input :num_epochs num_epochs :queue_capacity queue_capacity :feature_queue_capacity feature_queue_capacity :reader_num_threads reader_num_threads :num_enqueue_threads num_enqueue_threads :parse_fn parse_fn :name name :read_batch_size read_batch_size }))

(defn read-batch-record-features 
  "Reads TFRecord, queues, batches and parses `Example` proto. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use tf.data.

See more detailed description in `read_examples`.

Args:
  file_pattern: List of files or patterns of file paths containing
      `Example` records. See `tf.io.gfile.glob` for pattern rules.
  batch_size: An int or scalar `Tensor` specifying the batch size to use.
  features: A `dict` mapping feature keys to `FixedLenFeature` or
    `VarLenFeature` values.
  randomize_input: Whether the input should be randomized.
  num_epochs: Integer specifying the number of times to read through the
    dataset. If None, cycles through the dataset forever. NOTE - If specified,
    creates a variable that must be initialized, so call
    tf.compat.v1.local_variables_initializer() and run the op in a session.
  queue_capacity: Capacity for input queue.
  reader_num_threads: The number of threads to read examples. In order to have
    predictable and repeatable order of reading and enqueueing, such as in
    prediction and evaluation mode, `reader_num_threads` should be 1.
  name: Name of resulting op.

Returns:
  A dict of `Tensor` or `SparseTensor` objects for each in `features`.

Raises:
  ValueError: for invalid inputs."
  [file_pattern batch_size features & {:keys [randomize_input num_epochs queue_capacity reader_num_threads name]
                       :or {num_epochs None}} ]
    (py/call-attr-kw io "read_batch_record_features" [file_pattern batch_size features] {:randomize_input randomize_input :num_epochs num_epochs :queue_capacity queue_capacity :reader_num_threads reader_num_threads :name name }))

(defn read-keyed-batch-examples 
  "Adds operations to read, queue, batch `Example` protos. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use tf.data.

Given file pattern (or list of files), will setup a queue for file names,
read `Example` proto using provided `reader`, use batch queue to create
batches of examples of size `batch_size`.

All queue runners are added to the queue runners collection, and may be
started via `start_queue_runners`.

All ops are added to the default graph.

Use `parse_fn` if you need to do parsing / processing on single examples.

Args:
  file_pattern: List of files or patterns of file paths containing
      `Example` records. See `tf.io.gfile.glob` for pattern rules.
  batch_size: An int or scalar `Tensor` specifying the batch size to use.
  reader: A function or class that returns an object with
    `read` method, (filename tensor) -> (example tensor).
  randomize_input: Whether the input should be randomized.
  num_epochs: Integer specifying the number of times to read through the
    dataset. If `None`, cycles through the dataset forever.
    NOTE - If specified, creates a variable that must be initialized, so call
    `tf.compat.v1.local_variables_initializer()` and run the op in a session.
  queue_capacity: Capacity for input queue.
  num_threads: The number of threads enqueuing examples. In order to have
    predictable and repeatable order of reading and enqueueing, such as in
    prediction and evaluation mode, `num_threads` should be 1.
  read_batch_size: An int or scalar `Tensor` specifying the number of
    records to read at once.
  parse_fn: Parsing function, takes `Example` Tensor returns parsed
    representation. If `None`, no parsing is done.
  name: Name of resulting op.
  seed: An integer (optional). Seed used if randomize_input == True.

Returns:
  Returns tuple of:
  - `Tensor` of string keys.
  - String `Tensor` of batched `Example` proto.

Raises:
  ValueError: for invalid inputs."
  [file_pattern batch_size reader & {:keys [randomize_input num_epochs queue_capacity num_threads read_batch_size parse_fn name seed]
                       :or {num_epochs None parse_fn None name None seed None}} ]
    (py/call-attr-kw io "read_keyed_batch_examples" [file_pattern batch_size reader] {:randomize_input randomize_input :num_epochs num_epochs :queue_capacity queue_capacity :num_threads num_threads :read_batch_size read_batch_size :parse_fn parse_fn :name name :seed seed }))

(defn read-keyed-batch-examples-shared-queue 
  "Adds operations to read, queue, batch `Example` protos. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use tf.data.

Given file pattern (or list of files), will setup a shared queue for file
names, setup a worker queue that pulls from the shared queue, read `Example`
protos using provided `reader`, use batch queue to create batches of examples
of size `batch_size`. This provides at most once visit guarantees. Note that
this only works if the parameter servers are not pre-empted or restarted or
the session is not restored from a checkpoint since the state of a queue
is not checkpointed and we will end up restarting from the entire list of
files.

All queue runners are added to the queue runners collection, and may be
started via `start_queue_runners`.

All ops are added to the default graph.

Use `parse_fn` if you need to do parsing / processing on single examples.

Args:
  file_pattern: List of files or patterns of file paths containing
      `Example` records. See `tf.io.gfile.glob` for pattern rules.
  batch_size: An int or scalar `Tensor` specifying the batch size to use.
  reader: A function or class that returns an object with
    `read` method, (filename tensor) -> (example tensor).
  randomize_input: Whether the input should be randomized.
  num_epochs: Integer specifying the number of times to read through the
    dataset. If `None`, cycles through the dataset forever.
    NOTE - If specified, creates a variable that must be initialized, so call
    `tf.compat.v1.local_variables_initializer()` and run the op in a session.
  queue_capacity: Capacity for input queue.
  num_threads: The number of threads enqueuing examples.
  read_batch_size: An int or scalar `Tensor` specifying the number of
    records to read at once.
  parse_fn: Parsing function, takes `Example` Tensor returns parsed
    representation. If `None`, no parsing is done.
  name: Name of resulting op.
  seed: An integer (optional). Seed used if randomize_input == True.

Returns:
  Returns tuple of:
  - `Tensor` of string keys.
  - String `Tensor` of batched `Example` proto.

Raises:
  ValueError: for invalid inputs."
  [file_pattern batch_size reader & {:keys [randomize_input num_epochs queue_capacity num_threads read_batch_size parse_fn name seed]
                       :or {num_epochs None parse_fn None name None seed None}} ]
    (py/call-attr-kw io "read_keyed_batch_examples_shared_queue" [file_pattern batch_size reader] {:randomize_input randomize_input :num_epochs num_epochs :queue_capacity queue_capacity :num_threads num_threads :read_batch_size read_batch_size :parse_fn parse_fn :name name :seed seed }))

(defn read-keyed-batch-features 
  "Adds operations to read, queue, batch and parse `Example` protos. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use tf.data.

Given file pattern (or list of files), will setup a queue for file names,
read `Example` proto using provided `reader`, use batch queue to create
batches of examples of size `batch_size` and parse example given `features`
specification.

All queue runners are added to the queue runners collection, and may be
started via `start_queue_runners`.

All ops are added to the default graph.

Args:
  file_pattern: List of files or patterns of file paths containing
      `Example` records. See `tf.io.gfile.glob` for pattern rules.
  batch_size: An int or scalar `Tensor` specifying the batch size to use.
  features: A `dict` mapping feature keys to `FixedLenFeature` or
    `VarLenFeature` values.
  reader: A function or class that returns an object with
    `read` method, (filename tensor) -> (example tensor).
  randomize_input: Whether the input should be randomized.
  num_epochs: Integer specifying the number of times to read through the
    dataset. If None, cycles through the dataset forever. NOTE - If specified,
    creates a variable that must be initialized, so call
    tf.compat.v1.local_variables_initializer() and run the op in a session.
  queue_capacity: Capacity for input queue.
  reader_num_threads: The number of threads to read examples. In order to have
    predictable and repeatable order of reading and enqueueing, such as in
    prediction and evaluation mode, `reader_num_threads` should be 1.
  feature_queue_capacity: Capacity of the parsed features queue.
  num_enqueue_threads: Number of threads to enqueue the parsed example queue.
    Using multiple threads to enqueue the parsed example queue helps maintain
    a full queue when the subsequent computations overall are cheaper than
    parsing. In order to have predictable and repeatable order of reading and
    enqueueing, such as in prediction and evaluation mode,
    `num_enqueue_threads` should be 1.
  parse_fn: Parsing function, takes `Example` Tensor returns parsed
    representation. If `None`, no parsing is done.
  name: Name of resulting op.
  read_batch_size: An int or scalar `Tensor` specifying the number of
    records to read at once. If `None`, defaults to `batch_size`.

Returns:
  Returns tuple of:
  - `Tensor` of string keys.
  - A dict of `Tensor` or `SparseTensor` objects for each in `features`.

Raises:
  ValueError: for invalid inputs."
  [file_pattern batch_size features reader & {:keys [randomize_input num_epochs queue_capacity reader_num_threads feature_queue_capacity num_enqueue_threads parse_fn name read_batch_size]
                       :or {num_epochs None parse_fn None name None read_batch_size None}} ]
    (py/call-attr-kw io "read_keyed_batch_features" [file_pattern batch_size features reader] {:randomize_input randomize_input :num_epochs num_epochs :queue_capacity queue_capacity :reader_num_threads reader_num_threads :feature_queue_capacity feature_queue_capacity :num_enqueue_threads num_enqueue_threads :parse_fn parse_fn :name name :read_batch_size read_batch_size }))

(defn read-keyed-batch-features-shared-queue 
  "Adds operations to read, queue, batch and parse `Example` protos. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use tf.data.

Given file pattern (or list of files), will setup a shared queue for file
names, setup a worker queue that gets filenames from the shared queue,
read `Example` proto using provided `reader`, use batch queue to create
batches of examples of size `batch_size` and parse example given `features`
specification.

All queue runners are added to the queue runners collection, and may be
started via `start_queue_runners`.

All ops are added to the default graph.

Args:
  file_pattern: List of files or patterns of file paths containing
      `Example` records. See `tf.io.gfile.glob` for pattern rules.
  batch_size: An int or scalar `Tensor` specifying the batch size to use.
  features: A `dict` mapping feature keys to `FixedLenFeature` or
    `VarLenFeature` values.
  reader: A function or class that returns an object with
    `read` method, (filename tensor) -> (example tensor).
  randomize_input: Whether the input should be randomized.
  num_epochs: Integer specifying the number of times to read through the
    dataset. If None, cycles through the dataset forever. NOTE - If specified,
    creates a variable that must be initialized, so call
    tf.compat.v1.local_variables_initializer() and run the op in a session.
  queue_capacity: Capacity for input queue.
  reader_num_threads: The number of threads to read examples.
  feature_queue_capacity: Capacity of the parsed features queue.
  num_queue_runners: Number of threads to enqueue the parsed example queue.
    Using multiple threads to enqueue the parsed example queue helps maintain
    a full queue when the subsequent computations overall are cheaper than
    parsing.
  parse_fn: Parsing function, takes `Example` Tensor returns parsed
    representation. If `None`, no parsing is done.
  name: Name of resulting op.

Returns:
  Returns tuple of:
  - `Tensor` of string keys.
  - A dict of `Tensor` or `SparseTensor` objects for each in `features`.

Raises:
  ValueError: for invalid inputs."
  [file_pattern batch_size features reader & {:keys [randomize_input num_epochs queue_capacity reader_num_threads feature_queue_capacity num_queue_runners parse_fn name]
                       :or {num_epochs None parse_fn None name None}} ]
    (py/call-attr-kw io "read_keyed_batch_features_shared_queue" [file_pattern batch_size features reader] {:randomize_input randomize_input :num_epochs num_epochs :queue_capacity queue_capacity :reader_num_threads reader_num_threads :feature_queue_capacity feature_queue_capacity :num_queue_runners num_queue_runners :parse_fn parse_fn :name name }))
