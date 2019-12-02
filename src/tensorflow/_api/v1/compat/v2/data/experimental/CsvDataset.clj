(ns tensorflow.-api.v1.compat.v2.data.experimental.CsvDataset
  "A Dataset comprising lines from one or more CSV files."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce experimental (import-module "tensorflow._api.v1.compat.v2.data.experimental"))

(defn CsvDataset 
  "A Dataset comprising lines from one or more CSV files."
  [filenames record_defaults compression_type buffer_size & {:keys [header field_delim use_quote_delim na_value select_cols]
                       :or {select_cols None}} ]
    (py/call-attr-kw experimental "CsvDataset" [filenames record_defaults compression_type buffer_size] {:header header :field_delim field_delim :use_quote_delim use_quote_delim :na_value na_value :select_cols select_cols }))

(defn apply 
  "Applies a transformation function to this dataset.

    `apply` enables chaining of custom `Dataset` transformations, which are
    represented as functions that take one `Dataset` argument and return a
    transformed `Dataset`.

    For example:

    ```
    dataset = (dataset.map(lambda x: x ** 2)
               .apply(group_by_window(key_func, reduce_func, window_size))
               .map(lambda x: x ** 3))
    ```

    Args:
      transformation_func: A function that takes one `Dataset` argument and
        returns a `Dataset`.

    Returns:
      Dataset: The `Dataset` returned by applying `transformation_func` to this
          dataset.
    "
  [ self transformation_func ]
  (py/call-attr self "apply"  self transformation_func ))
(defn batch 
  "Combines consecutive elements of this dataset into batches.

    The components of the resulting element will have an additional outer
    dimension, which will be `batch_size` (or `N % batch_size` for the last
    element if `batch_size` does not divide the number of input elements `N`
    evenly and `drop_remainder` is `False`). If your program depends on the
    batches having the same outer dimension, you should set the `drop_remainder`
    argument to `True` to prevent the smaller batch from being produced.

    Args:
      batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
        consecutive elements of this dataset to combine in a single batch.
      drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
        whether the last batch should be dropped in the case it has fewer than
        `batch_size` elements; the default behavior is not to drop the smaller
        batch.

    Returns:
      Dataset: A `Dataset`.
    "
  [self batch_size  & {:keys [drop_remainder]} ]
    (py/call-attr-kw self "batch" [batch_size] {:drop_remainder drop_remainder }))
(defn cache 
  "Caches the elements in this dataset.

    Args:
      filename: A `tf.string` scalar `tf.Tensor`, representing the name of a
        directory on the filesystem to use for caching elements in this Dataset.
        If a filename is not provided, the dataset will be cached in memory.

    Returns:
      Dataset: A `Dataset`.
    "
  [self   & {:keys [filename]} ]
    (py/call-attr-kw self "cache" [] {:filename filename }))

(defn concatenate 
  "Creates a `Dataset` by concatenating the given dataset with this dataset.

    ```python
    a = Dataset.range(1, 4)  # ==> [ 1, 2, 3 ]
    b = Dataset.range(4, 8)  # ==> [ 4, 5, 6, 7 ]

    # The input dataset and dataset to be concatenated should have the same
    # nested structures and output types.
    # c = Dataset.range(8, 14).batch(2)  # ==> [ [8, 9], [10, 11], [12, 13] ]
    # d = Dataset.from_tensor_slices([14.0, 15.0, 16.0])
    # a.concatenate(c) and a.concatenate(d) would result in error.

    a.concatenate(b)  # ==> [ 1, 2, 3, 4, 5, 6, 7 ]
    ```

    Args:
      dataset: `Dataset` to be concatenated.

    Returns:
      Dataset: A `Dataset`.
    "
  [ self dataset ]
  (py/call-attr self "concatenate"  self dataset ))

(defn element-spec 
  ""
  [ self ]
    (py/call-attr self "element_spec"))
(defn enumerate 
  "Enumerates the elements of this dataset.

    It is similar to python's `enumerate`.

    For example:

    ```python
    # NOTE: The following examples use `{ ... }` to represent the
    # contents of a dataset.
    a = { 1, 2, 3 }
    b = { (7, 8), (9, 10) }

    # The nested structure of the `datasets` argument determines the
    # structure of elements in the resulting dataset.
    a.enumerate(start=5)) == { (5, 1), (6, 2), (7, 3) }
    b.enumerate() == { (0, (7, 8)), (1, (9, 10)) }
    ```

    Args:
      start: A `tf.int64` scalar `tf.Tensor`, representing the start value for
        enumeration.

    Returns:
      Dataset: A `Dataset`.
    "
  [self   & {:keys [start]} ]
    (py/call-attr-kw self "enumerate" [] {:start start }))

(defn filter 
  "Filters this dataset according to `predicate`.

    ```python
    d = tf.data.Dataset.from_tensor_slices([1, 2, 3])

    d = d.filter(lambda x: x < 3)  # ==> [1, 2]

    # `tf.math.equal(x, y)` is required for equality comparison
    def filter_fn(x):
      return tf.math.equal(x, 1)

    d = d.filter(filter_fn)  # ==> [1]
    ```

    Args:
      predicate: A function mapping a dataset element to a boolean.

    Returns:
      Dataset: The `Dataset` containing the elements of this dataset for which
          `predicate` is `True`.
    "
  [ self predicate ]
  (py/call-attr self "filter"  self predicate ))

(defn flat-map 
  "Maps `map_func` across this dataset and flattens the result.

    Use `flat_map` if you want to make sure that the order of your dataset
    stays the same. For example, to flatten a dataset of batches into a
    dataset of their elements:

    ```python
    a = Dataset.from_tensor_slices([ [1, 2, 3], [4, 5, 6], [7, 8, 9] ])

    a.flat_map(lambda x: Dataset.from_tensor_slices(x + 1)) # ==>
    #  [ 2, 3, 4, 5, 6, 7, 8, 9, 10 ]
    ```

    `tf.data.Dataset.interleave()` is a generalization of `flat_map`, since
    `flat_map` produces the same output as
    `tf.data.Dataset.interleave(cycle_length=1)`

    Args:
      map_func: A function mapping a dataset element to a dataset.

    Returns:
      Dataset: A `Dataset`.
    "
  [ self map_func ]
  (py/call-attr self "flat_map"  self map_func ))

(defn from-generator 
  "Creates a `Dataset` whose elements are generated by `generator`.

    The `generator` argument must be a callable object that returns
    an object that supports the `iter()` protocol (e.g. a generator function).
    The elements generated by `generator` must be compatible with the given
    `output_types` and (optional) `output_shapes` arguments.

    For example:

    ```python
    import itertools
    tf.compat.v1.enable_eager_execution()

    def gen():
      for i in itertools.count(1):
        yield (i, [1] * i)

    ds = tf.data.Dataset.from_generator(
        gen, (tf.int64, tf.int64), (tf.TensorShape([]), tf.TensorShape([None])))

    for value in ds.take(2):
      print value
    # (1, array([1]))
    # (2, array([1, 1]))
    ```

    NOTE: The current implementation of `Dataset.from_generator()` uses
    `tf.numpy_function` and inherits the same constraints. In particular, it
    requires the `Dataset`- and `Iterator`-related operations to be placed
    on a device in the same process as the Python program that called
    `Dataset.from_generator()`. The body of `generator` will not be
    serialized in a `GraphDef`, and you should not use this method if you
    need to serialize your model and restore it in a different environment.

    NOTE: If `generator` depends on mutable global variables or other external
    state, be aware that the runtime may invoke `generator` multiple times
    (in order to support repeating the `Dataset`) and at any time
    between the call to `Dataset.from_generator()` and the production of the
    first element from the generator. Mutating global variables or external
    state can cause undefined behavior, and we recommend that you explicitly
    cache any external state in `generator` before calling
    `Dataset.from_generator()`.

    Args:
      generator: A callable object that returns an object that supports the
        `iter()` protocol. If `args` is not specified, `generator` must take no
        arguments; otherwise it must take as many arguments as there are values
        in `args`.
      output_types: A nested structure of `tf.DType` objects corresponding to
        each component of an element yielded by `generator`.
      output_shapes: (Optional.) A nested structure of `tf.TensorShape` objects
        corresponding to each component of an element yielded by `generator`.
      args: (Optional.) A tuple of `tf.Tensor` objects that will be evaluated
        and passed to `generator` as NumPy-array arguments.

    Returns:
      Dataset: A `Dataset`.
    "
  [ self generator output_types output_shapes args ]
  (py/call-attr self "from_generator"  self generator output_types output_shapes args ))

(defn from-tensor-slices 
  "Creates a `Dataset` whose elements are slices of the given tensors.

    Note that if `tensors` contains a NumPy array, and eager execution is not
    enabled, the values will be embedded in the graph as one or more
    `tf.constant` operations. For large datasets (> 1 GB), this can waste
    memory and run into byte limits of graph serialization. If `tensors`
    contains one or more large NumPy arrays, consider the alternative described
    in [this guide](
    https://tensorflow.org/guide/datasets#consuming_numpy_arrays).

    Args:
      tensors: A dataset element, with each component having the same size in
        the 0th dimension.

    Returns:
      Dataset: A `Dataset`.
    "
  [ self tensors ]
  (py/call-attr self "from_tensor_slices"  self tensors ))

(defn from-tensors 
  "Creates a `Dataset` with a single element, comprising the given tensors.

    Note that if `tensors` contains a NumPy array, and eager execution is not
    enabled, the values will be embedded in the graph as one or more
    `tf.constant` operations. For large datasets (> 1 GB), this can waste
    memory and run into byte limits of graph serialization. If `tensors`
    contains one or more large NumPy arrays, consider the alternative described
    in [this
    guide](https://tensorflow.org/guide/datasets#consuming_numpy_arrays).

    Args:
      tensors: A dataset element.

    Returns:
      Dataset: A `Dataset`.
    "
  [ self tensors ]
  (py/call-attr self "from_tensors"  self tensors ))

(defn interleave 
  "Maps `map_func` across this dataset, and interleaves the results.

    For example, you can use `Dataset.interleave()` to process many input files
    concurrently:

    ```python
    # Preprocess 4 files concurrently, and interleave blocks of 16 records from
    # each file.
    filenames = [\"/var/data/file1.txt\", \"/var/data/file2.txt\", ...]
    dataset = (Dataset.from_tensor_slices(filenames)
               .interleave(lambda x:
                   TextLineDataset(x).map(parse_fn, num_parallel_calls=1),
                   cycle_length=4, block_length=16))
    ```

    The `cycle_length` and `block_length` arguments control the order in which
    elements are produced. `cycle_length` controls the number of input elements
    that are processed concurrently. If you set `cycle_length` to 1, this
    transformation will handle one input element at a time, and will produce
    identical results to `tf.data.Dataset.flat_map`. In general,
    this transformation will apply `map_func` to `cycle_length` input elements,
    open iterators on the returned `Dataset` objects, and cycle through them
    producing `block_length` consecutive elements from each iterator, and
    consuming the next input element each time it reaches the end of an
    iterator.

    For example:

    ```python
    a = Dataset.range(1, 6)  # ==> [ 1, 2, 3, 4, 5 ]

    # NOTE: New lines indicate \"block\" boundaries.
    a.interleave(lambda x: Dataset.from_tensors(x).repeat(6),
                cycle_length=2, block_length=4)  # ==> [1, 1, 1, 1,
                                                 #      2, 2, 2, 2,
                                                 #      1, 1,
                                                 #      2, 2,
                                                 #      3, 3, 3, 3,
                                                 #      4, 4, 4, 4,
                                                 #      3, 3,
                                                 #      4, 4,
                                                 #      5, 5, 5, 5,
                                                 #      5, 5]
    ```

    NOTE: The order of elements yielded by this transformation is
    deterministic, as long as `map_func` is a pure function. If
    `map_func` contains any stateful operations, the order in which
    that state is accessed is undefined.

    Args:
      map_func: A function mapping a dataset element to a dataset.
      cycle_length: (Optional.) The number of input elements that will be
        processed concurrently. If not specified, the value will be derived from
        the number of available CPU cores. If the `num_parallel_calls` argument
        is set to `tf.data.experimental.AUTOTUNE`, the `cycle_length` argument
        also identifies the maximum degree of parallelism.
      block_length: (Optional.) The number of consecutive elements to produce
        from each input element before cycling to another input element.
      num_parallel_calls: (Optional.) If specified, the implementation creates a
        threadpool, which is used to fetch inputs from cycle elements
        asynchronously and in parallel. The default behavior is to fetch inputs
        from cycle elements synchronously with no parallelism. If the value
        `tf.data.experimental.AUTOTUNE` is used, then the number of parallel
        calls is set dynamically based on available CPU.

    Returns:
      Dataset: A `Dataset`.
    "
  [self map_func & {:keys [cycle_length block_length num_parallel_calls]
                       :or {num_parallel_calls None}} ]
    (py/call-attr-kw self "interleave" [map_func] {:cycle_length cycle_length :block_length block_length :num_parallel_calls num_parallel_calls }))

(defn list-files 
  "A dataset of all files matching one or more glob patterns.

    NOTE: The default behavior of this method is to return filenames in
    a non-deterministic random shuffled order. Pass a `seed` or `shuffle=False`
    to get results in a deterministic order.

    Example:
      If we had the following files on our filesystem:
        - /path/to/dir/a.txt
        - /path/to/dir/b.py
        - /path/to/dir/c.py
      If we pass \"/path/to/dir/*.py\" as the directory, the dataset
      would produce:
        - /path/to/dir/b.py
        - /path/to/dir/c.py

    Args:
      file_pattern: A string, a list of strings, or a `tf.Tensor` of string type
        (scalar or vector), representing the filename glob (i.e. shell wildcard)
        pattern(s) that will be matched.
      shuffle: (Optional.) If `True`, the file names will be shuffled randomly.
        Defaults to `True`.
      seed: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the random
        seed that will be used to create the distribution. See
        `tf.compat.v1.set_random_seed` for behavior.

    Returns:
     Dataset: A `Dataset` of strings corresponding to file names.
    "
  [ self file_pattern shuffle seed ]
  (py/call-attr self "list_files"  self file_pattern shuffle seed ))

(defn map 
  "Maps `map_func` across the elements of this dataset.

    This transformation applies `map_func` to each element of this dataset, and
    returns a new dataset containing the transformed elements, in the same
    order as they appeared in the input.

    For example:

    ```python
    a = Dataset.range(1, 6)  # ==> [ 1, 2, 3, 4, 5 ]

    a.map(lambda x: x + 1)  # ==> [ 2, 3, 4, 5, 6 ]
    ```

    The input signature of `map_func` is determined by the structure of each
    element in this dataset. For example:

    ```python
    # NOTE: The following examples use `{ ... }` to represent the
    # contents of a dataset.
    # Each element is a `tf.Tensor` object.
    a = { 1, 2, 3, 4, 5 }
    # `map_func` takes a single argument of type `tf.Tensor` with the same
    # shape and dtype.
    result = a.map(lambda x: ...)

    # Each element is a tuple containing two `tf.Tensor` objects.
    b = { (1, \"foo\"), (2, \"bar\"), (3, \"baz\") }
    # `map_func` takes two arguments of type `tf.Tensor`.
    result = b.map(lambda x_int, y_str: ...)

    # Each element is a dictionary mapping strings to `tf.Tensor` objects.
    c = { {\"a\": 1, \"b\": \"foo\"}, {\"a\": 2, \"b\": \"bar\"}, {\"a\": 3, \"b\": \"baz\"} }
    # `map_func` takes a single argument of type `dict` with the same keys as
    # the elements.
    result = c.map(lambda d: ...)
    ```

    The value or values returned by `map_func` determine the structure of each
    element in the returned dataset.

    ```python
    # `map_func` returns a scalar `tf.Tensor` of type `tf.float32`.
    def f(...):
      return tf.constant(37.0)
    result = dataset.map(f)
    result.output_classes == tf.Tensor
    result.output_types == tf.float32
    result.output_shapes == []  # scalar

    # `map_func` returns two `tf.Tensor` objects.
    def g(...):
      return tf.constant(37.0), tf.constant([\"Foo\", \"Bar\", \"Baz\"])
    result = dataset.map(g)
    result.output_classes == (tf.Tensor, tf.Tensor)
    result.output_types == (tf.float32, tf.string)
    result.output_shapes == ([], [3])

    # Python primitives, lists, and NumPy arrays are implicitly converted to
    # `tf.Tensor`.
    def h(...):
      return 37.0, [\"Foo\", \"Bar\", \"Baz\"], np.array([1.0, 2.0] dtype=np.float64)
    result = dataset.map(h)
    result.output_classes == (tf.Tensor, tf.Tensor, tf.Tensor)
    result.output_types == (tf.float32, tf.string, tf.float64)
    result.output_shapes == ([], [3], [2])

    # `map_func` can return nested structures.
    def i(...):
      return {\"a\": 37.0, \"b\": [42, 16]}, \"foo\"
    result.output_classes == ({\"a\": tf.Tensor, \"b\": tf.Tensor}, tf.Tensor)
    result.output_types == ({\"a\": tf.float32, \"b\": tf.int32}, tf.string)
    result.output_shapes == ({\"a\": [], \"b\": [2]}, [])
    ```

    `map_func` can accept as arguments and return any type of dataset element.

    Note that irrespective of the context in which `map_func` is defined (eager
    vs. graph), tf.data traces the function and executes it as a graph. To use
    Python code inside of the function you have two options:

    1) Rely on AutoGraph to convert Python code into an equivalent graph
    computation. The downside of this approach is that AutoGraph can convert
    some but not all Python code.

    2) Use `tf.py_function`, which allows you to write arbitrary Python code but
    will generally result in worse performance than 1). For example:

    ```python
    d = tf.data.Dataset.from_tensor_slices(['hello', 'world'])

    # transform a string tensor to upper case string using a Python function
    def upper_case_fn(t: tf.Tensor) -> str:
        return t.numpy().decode('utf-8').upper()

    d.map(lambda x: tf.py_function(func=upper_case_fn,
          inp=[x], Tout=tf.string))  # ==> [ \"HELLO\", \"WORLD\" ]
    ```

    Args:
      map_func: A function mapping a dataset element to another dataset element.
      num_parallel_calls: (Optional.) A `tf.int32` scalar `tf.Tensor`,
        representing the number elements to process asynchronously in parallel.
        If not specified, elements will be processed sequentially. If the value
        `tf.data.experimental.AUTOTUNE` is used, then the number of parallel
        calls is set dynamically based on available CPU.

    Returns:
      Dataset: A `Dataset`.
    "
  [ self map_func num_parallel_calls ]
  (py/call-attr self "map"  self map_func num_parallel_calls ))

(defn options 
  "Returns the options for this dataset and its inputs.

    Returns:
      A `tf.data.Options` object representing the dataset options.
    "
  [ self  ]
  (py/call-attr self "options"  self  ))
(defn padded-batch 
  "Combines consecutive elements of this dataset into padded batches.

    This transformation combines multiple consecutive elements of the input
    dataset into a single element.

    Like `tf.data.Dataset.batch`, the components of the resulting element will
    have an additional outer dimension, which will be `batch_size` (or
    `N % batch_size` for the last element if `batch_size` does not divide the
    number of input elements `N` evenly and `drop_remainder` is `False`). If
    your program depends on the batches having the same outer dimension, you
    should set the `drop_remainder` argument to `True` to prevent the smaller
    batch from being produced.

    Unlike `tf.data.Dataset.batch`, the input elements to be batched may have
    different shapes, and this transformation will pad each component to the
    respective shape in `padding_shapes`. The `padding_shapes` argument
    determines the resulting shape for each dimension of each component in an
    output element:

    * If the dimension is a constant (e.g. `tf.compat.v1.Dimension(37)`), the
    component
      will be padded out to that length in that dimension.
    * If the dimension is unknown (e.g. `tf.compat.v1.Dimension(None)`), the
    component
      will be padded out to the maximum length of all elements in that
      dimension.

    See also `tf.data.experimental.dense_to_sparse_batch`, which combines
    elements that may have different shapes into a `tf.SparseTensor`.

    Args:
      batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
        consecutive elements of this dataset to combine in a single batch.
      padded_shapes: A nested structure of `tf.TensorShape` or `tf.int64` vector
        tensor-like objects representing the shape to which the respective
        component of each input element should be padded prior to batching. Any
        unknown dimensions (e.g. `tf.compat.v1.Dimension(None)` in a
        `tf.TensorShape` or `-1` in a tensor-like object) will be padded to the
        maximum size of that dimension in each batch.
      padding_values: (Optional.) A nested structure of scalar-shaped
        `tf.Tensor`, representing the padding values to use for the respective
        components.  Defaults are `0` for numeric types and the empty string for
        string types.
      drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
        whether the last batch should be dropped in the case it has fewer than
        `batch_size` elements; the default behavior is not to drop the smaller
        batch.

    Returns:
      Dataset: A `Dataset`.
    "
  [self batch_size padded_shapes padding_values  & {:keys [drop_remainder]} ]
    (py/call-attr-kw self "padded_batch" [batch_size padded_shapes padding_values] {:drop_remainder drop_remainder }))

(defn prefetch 
  "Creates a `Dataset` that prefetches elements from this dataset.

    Note: Like other `Dataset` methods, prefetch operates on the
    elements of the input dataset. It has no concept of examples vs. batches.
    `examples.prefetch(2)` will prefetch two elements (2 examples),
    while `examples.batch(20).prefetch(2)` will prefetch 2 elements
    (2 batches, of 20 examples each).

    Args:
      buffer_size: A `tf.int64` scalar `tf.Tensor`, representing the maximum
        number of elements that will be buffered when prefetching.

    Returns:
      Dataset: A `Dataset`.
    "
  [ self buffer_size ]
  (py/call-attr self "prefetch"  self buffer_size ))

(defn range 
  "Creates a `Dataset` of a step-separated range of values.

    For example:

    ```python
    Dataset.range(5) == [0, 1, 2, 3, 4]
    Dataset.range(2, 5) == [2, 3, 4]
    Dataset.range(1, 5, 2) == [1, 3]
    Dataset.range(1, 5, -2) == []
    Dataset.range(5, 1) == []
    Dataset.range(5, 1, -2) == [5, 3]
    ```

    Args:
      *args: follows the same semantics as python's xrange.
        len(args) == 1 -> start = 0, stop = args[0], step = 1
        len(args) == 2 -> start = args[0], stop = args[1], step = 1
        len(args) == 3 -> start = args[0], stop = args[1, stop = args[2]

    Returns:
      Dataset: A `RangeDataset`.

    Raises:
      ValueError: if len(args) == 0.
    "
  [ self  ]
  (py/call-attr self "range"  self  ))

(defn reduce 
  "Reduces the input dataset to a single element.

    The transformation calls `reduce_func` successively on every element of
    the input dataset until the dataset is exhausted, aggregating information in
    its internal state. The `initial_state` argument is used for the initial
    state and the final state is returned as the result.

    For example:
    - `tf.data.Dataset.range(5).reduce(np.int64(0), lambda x, _: x + 1)`
      produces `5`
    - `tf.data.Dataset.range(5).reduce(np.int64(0), lambda x, y: x + y)`
      produces `10`

    Args:
      initial_state: An element representing the initial state of the
        transformation.
      reduce_func: A function that maps `(old_state, input_element)` to
        `new_state`. It must take two arguments and return a new element
        The structure of `new_state` must match the structure of
        `initial_state`.

    Returns:
      A dataset element corresponding to the final state of the transformation.

    "
  [ self initial_state reduce_func ]
  (py/call-attr self "reduce"  self initial_state reduce_func ))

(defn repeat 
  "Repeats this dataset `count` times.

    NOTE: If this dataset is a function of global state (e.g. a random number
    generator), then different repetitions may produce different elements.

    Args:
      count: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the
        number of times the dataset should be repeated. The default behavior (if
        `count` is `None` or `-1`) is for the dataset be repeated indefinitely.

    Returns:
      Dataset: A `Dataset`.
    "
  [ self count ]
  (py/call-attr self "repeat"  self count ))

(defn shard 
  "Creates a `Dataset` that includes only 1/`num_shards` of this dataset.

    This dataset operator is very useful when running distributed training, as
    it allows each worker to read a unique subset.

    When reading a single input file, you can skip elements as follows:

    ```python
    d = tf.data.TFRecordDataset(input_file)
    d = d.shard(num_workers, worker_index)
    d = d.repeat(num_epochs)
    d = d.shuffle(shuffle_buffer_size)
    d = d.map(parser_fn, num_parallel_calls=num_map_threads)
    ```

    Important caveats:

    - Be sure to shard before you use any randomizing operator (such as
      shuffle).
    - Generally it is best if the shard operator is used early in the dataset
      pipeline. For example, when reading from a set of TFRecord files, shard
      before converting the dataset to input samples. This avoids reading every
      file on every worker. The following is an example of an efficient
      sharding strategy within a complete pipeline:

    ```python
    d = Dataset.list_files(pattern)
    d = d.shard(num_workers, worker_index)
    d = d.repeat(num_epochs)
    d = d.shuffle(shuffle_buffer_size)
    d = d.interleave(tf.data.TFRecordDataset,
                     cycle_length=num_readers, block_length=1)
    d = d.map(parser_fn, num_parallel_calls=num_map_threads)
    ```

    Args:
      num_shards: A `tf.int64` scalar `tf.Tensor`, representing the number of
        shards operating in parallel.
      index: A `tf.int64` scalar `tf.Tensor`, representing the worker index.

    Returns:
      Dataset: A `Dataset`.

    Raises:
      InvalidArgumentError: if `num_shards` or `index` are illegal values.
        Note: error checking is done on a best-effort basis, and errors aren't
        guaranteed to be caught upon dataset creation. (e.g. providing in a
        placeholder tensor bypasses the early checking, and will instead result
        in an error during a session.run call.)
    "
  [ self num_shards index ]
  (py/call-attr self "shard"  self num_shards index ))

(defn shuffle 
  "Randomly shuffles the elements of this dataset.

    This dataset fills a buffer with `buffer_size` elements, then randomly
    samples elements from this buffer, replacing the selected elements with new
    elements. For perfect shuffling, a buffer size greater than or equal to the
    full size of the dataset is required.

    For instance, if your dataset contains 10,000 elements but `buffer_size` is
    set to 1,000, then `shuffle` will initially select a random element from
    only the first 1,000 elements in the buffer. Once an element is selected,
    its space in the buffer is replaced by the next (i.e. 1,001-st) element,
    maintaining the 1,000 element buffer.

    Args:
      buffer_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
        elements from this dataset from which the new dataset will sample.
      seed: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the random
        seed that will be used to create the distribution. See
        `tf.compat.v1.set_random_seed` for behavior.
      reshuffle_each_iteration: (Optional.) A boolean, which if true indicates
        that the dataset should be pseudorandomly reshuffled each time it is
        iterated over. (Defaults to `True`.)

    Returns:
      Dataset: A `Dataset`.
    "
  [ self buffer_size seed reshuffle_each_iteration ]
  (py/call-attr self "shuffle"  self buffer_size seed reshuffle_each_iteration ))

(defn skip 
  "Creates a `Dataset` that skips `count` elements from this dataset.

    Args:
      count: A `tf.int64` scalar `tf.Tensor`, representing the number of
        elements of this dataset that should be skipped to form the new dataset.
        If `count` is greater than the size of this dataset, the new dataset
        will contain no elements.  If `count` is -1, skips the entire dataset.

    Returns:
      Dataset: A `Dataset`.
    "
  [ self count ]
  (py/call-attr self "skip"  self count ))

(defn take 
  "Creates a `Dataset` with at most `count` elements from this dataset.

    Args:
      count: A `tf.int64` scalar `tf.Tensor`, representing the number of
        elements of this dataset that should be taken to form the new dataset.
        If `count` is -1, or if `count` is greater than the size of this
        dataset, the new dataset will contain all elements of this dataset.

    Returns:
      Dataset: A `Dataset`.
    "
  [ self count ]
  (py/call-attr self "take"  self count ))

(defn unbatch 
  "Splits elements of a dataset into multiple elements.

    For example, if elements of the dataset are shaped `[B, a0, a1, ...]`,
    where `B` may vary for each input element, then for each element in the
    dataset, the unbatched dataset will contain `B` consecutive elements
    of shape `[a0, a1, ...]`.

    ```python
    # NOTE: The following example uses `{ ... }` to represent the contents
    # of a dataset.
    ds = { ['a', 'b', 'c'], ['a', 'b'], ['a', 'b', 'c', 'd'] }

    ds.unbatch() == {'a', 'b', 'c', 'a', 'b', 'a', 'b', 'c', 'd'}
    ```

    Returns:
      A `Dataset` transformation function, which can be passed to
      `tf.data.Dataset.apply`.
    "
  [ self  ]
  (py/call-attr self "unbatch"  self  ))
(defn window 
  "Combines (nests of) input elements into a dataset of (nests of) windows.

    A \"window\" is a finite dataset of flat elements of size `size` (or possibly
    fewer if there are not enough input elements to fill the window and
    `drop_remainder` evaluates to false).

    The `stride` argument determines the stride of the input elements, and the
    `shift` argument determines the shift of the window.

    For example, letting {...} to represent a Dataset:

    - `tf.data.Dataset.range(7).window(2)` produces
      `{{0, 1}, {2, 3}, {4, 5}, {6}}`
    - `tf.data.Dataset.range(7).window(3, 2, 1, True)` produces
      `{{0, 1, 2}, {2, 3, 4}, {4, 5, 6}}`
    - `tf.data.Dataset.range(7).window(3, 1, 2, True)` produces
      `{{0, 2, 4}, {1, 3, 5}, {2, 4, 6}}`

    Note that when the `window` transformation is applied to a dataset of
    nested elements, it produces a dataset of nested windows.

    For example:

    - `tf.data.Dataset.from_tensor_slices((range(4), range(4))).window(2)`
      produces `{({0, 1}, {0, 1}), ({2, 3}, {2, 3})}`
    - `tf.data.Dataset.from_tensor_slices({\"a\": range(4)}).window(2)`
      produces `{{\"a\": {0, 1}}, {\"a\": {2, 3}}}`

    Args:
      size: A `tf.int64` scalar `tf.Tensor`, representing the number of elements
        of the input dataset to combine into a window.
      shift: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the
        forward shift of the sliding window in each iteration. Defaults to
        `size`.
      stride: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the
        stride of the input elements in the sliding window.
      drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
        whether a window should be dropped in case its size is smaller than
        `window_size`.

    Returns:
      Dataset: A `Dataset` of (nests of) windows -- a finite datasets of flat
        elements created from the (nests of) input elements.

    "
  [self size shift  & {:keys [stride drop_remainder]} ]
    (py/call-attr-kw self "window" [size shift] {:stride stride :drop_remainder drop_remainder }))

(defn with-options 
  "Returns a new `tf.data.Dataset` with the given options set.

    The options are \"global\" in the sense they apply to the entire dataset.
    If options are set multiple times, they are merged as long as different
    options do not use different non-default values.

    Args:
      options: A `tf.data.Options` that identifies the options the use.

    Returns:
      Dataset: A `Dataset` with the given options.

    Raises:
      ValueError: when an option is set more than once to a non-default value
    "
  [ self options ]
  (py/call-attr self "with_options"  self options ))

(defn zip 
  "Creates a `Dataset` by zipping together the given datasets.

    This method has similar semantics to the built-in `zip()` function
    in Python, with the main difference being that the `datasets`
    argument can be an arbitrary nested structure of `Dataset` objects.
    For example:

    ```python
    a = Dataset.range(1, 4)  # ==> [ 1, 2, 3 ]
    b = Dataset.range(4, 7)  # ==> [ 4, 5, 6 ]
    c = Dataset.range(7, 13).batch(2)  # ==> [ [7, 8], [9, 10], [11, 12] ]
    d = Dataset.range(13, 15)  # ==> [ 13, 14 ]

    # The nested structure of the `datasets` argument determines the
    # structure of elements in the resulting dataset.
    Dataset.zip((a, b))  # ==> [ (1, 4), (2, 5), (3, 6) ]
    Dataset.zip((b, a))  # ==> [ (4, 1), (5, 2), (6, 3) ]

    # The `datasets` argument may contain an arbitrary number of
    # datasets.
    Dataset.zip((a, b, c))  # ==> [ (1, 4, [7, 8]),
                            #       (2, 5, [9, 10]),
                            #       (3, 6, [11, 12]) ]

    # The number of elements in the resulting dataset is the same as
    # the size of the smallest dataset in `datasets`.
    Dataset.zip((a, d))  # ==> [ (1, 13), (2, 14) ]
    ```

    Args:
      datasets: A nested structure of datasets.

    Returns:
      Dataset: A `Dataset`.
    "
  [ self datasets ]
  (py/call-attr self "zip"  self datasets ))
