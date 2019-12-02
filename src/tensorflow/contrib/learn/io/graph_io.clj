(ns tensorflow.contrib.learn.python.learn.learn-io.graph-io
  "Methods to read data in the graph (deprecated).

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
(defonce graph-io (import-module "tensorflow.contrib.learn.python.learn.learn_io.graph_io"))
(defn deprecated 
  "Decorator for marking functions or methods deprecated.

  This decorator logs a deprecation warning whenever the decorated function is
  called. It has the following format:

    <function> (from <module>) is deprecated and will be removed after <date>.
    Instructions for updating:
    <instructions>

  If `date` is None, 'after <date>' is replaced with 'in a future version'.
  <function> will include the class name if it is a method.

  It also edits the docstring of the function: ' (deprecated)' is appended
  to the first line of the docstring and a deprecation notice is prepended
  to the rest of the docstring.

  Args:
    date: String or None. The date the function is scheduled to be removed.
      Must be ISO 8601 (YYYY-MM-DD), or None.
    instructions: String. Instructions on how to update code using the
      deprecated function.
    warn_once: Boolean. Set to `True` to warn only the first time the decorated
      function is called. Otherwise, every call will log a warning.

  Returns:
    Decorated function or method.

  Raises:
    ValueError: If date is not None or in ISO 8601 format, or instructions are
      empty.
  "
  [date instructions  & {:keys [warn_once]} ]
    (py/call-attr-kw graph-io "deprecated" [date instructions] {:warn_once warn_once }))

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
    (py/call-attr-kw graph-io "queue_parsed_features" [parsed_features keys] {:feature_queue_capacity feature_queue_capacity :num_enqueue_threads num_enqueue_threads :name name }))

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
    (py/call-attr-kw graph-io "read_batch_examples" [file_pattern batch_size reader] {:randomize_input randomize_input :num_epochs num_epochs :queue_capacity queue_capacity :num_threads num_threads :read_batch_size read_batch_size :parse_fn parse_fn :name name :seed seed }))

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
    (py/call-attr-kw graph-io "read_batch_features" [file_pattern batch_size features reader] {:randomize_input randomize_input :num_epochs num_epochs :queue_capacity queue_capacity :feature_queue_capacity feature_queue_capacity :reader_num_threads reader_num_threads :num_enqueue_threads num_enqueue_threads :parse_fn parse_fn :name name :read_batch_size read_batch_size }))

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
    (py/call-attr-kw graph-io "read_batch_record_features" [file_pattern batch_size features] {:randomize_input randomize_input :num_epochs num_epochs :queue_capacity queue_capacity :reader_num_threads reader_num_threads :name name }))

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
    (py/call-attr-kw graph-io "read_keyed_batch_examples" [file_pattern batch_size reader] {:randomize_input randomize_input :num_epochs num_epochs :queue_capacity queue_capacity :num_threads num_threads :read_batch_size read_batch_size :parse_fn parse_fn :name name :seed seed }))

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
    (py/call-attr-kw graph-io "read_keyed_batch_examples_shared_queue" [file_pattern batch_size reader] {:randomize_input randomize_input :num_epochs num_epochs :queue_capacity queue_capacity :num_threads num_threads :read_batch_size read_batch_size :parse_fn parse_fn :name name :seed seed }))

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
    (py/call-attr-kw graph-io "read_keyed_batch_features" [file_pattern batch_size features reader] {:randomize_input randomize_input :num_epochs num_epochs :queue_capacity queue_capacity :reader_num_threads reader_num_threads :feature_queue_capacity feature_queue_capacity :num_enqueue_threads num_enqueue_threads :parse_fn parse_fn :name name :read_batch_size read_batch_size }))

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
    (py/call-attr-kw graph-io "read_keyed_batch_features_shared_queue" [file_pattern batch_size features reader] {:randomize_input randomize_input :num_epochs num_epochs :queue_capacity queue_capacity :reader_num_threads reader_num_threads :feature_queue_capacity feature_queue_capacity :num_queue_runners num_queue_runners :parse_fn parse_fn :name name }))
