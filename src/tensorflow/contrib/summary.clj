(ns tensorflow.contrib.summary.summary
  "TensorFlow Summary API v2.

The operations in this package are safe to use with eager execution turned on or
off. It has a more flexible API that allows summaries to be written directly
from ops to places other than event log files, rather than propagating protos
from `tf.summary.merge_all` to `tf.summary.FileWriter`.

To use with eager execution enabled, write your code as follows:

```python
global_step = tf.train.get_or_create_global_step()
summary_writer = tf.contrib.summary.create_file_writer(
    train_dir, flush_millis=10000)
with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
  # model code goes here
  # and in it call
  tf.contrib.summary.scalar(\"loss\", my_loss)
  # In this case every call to tf.contrib.summary.scalar will generate a record
  # ...
```

To use it with graph execution, write your code as follows:

```python
global_step = tf.train.get_or_create_global_step()
summary_writer = tf.contrib.summary.create_file_writer(
    train_dir, flush_millis=10000)
with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
  # model definition code goes here
  # and in it call
  tf.contrib.summary.scalar(\"loss\", my_loss)
  # In this case every call to tf.contrib.summary.scalar will generate an op,
  # note the need to run tf.contrib.summary.all_summary_ops() to make sure these
  # ops get executed.
  # ...
  train_op = ....

with tf.Session(...) as sess:
  tf.global_variables_initializer().run()
  tf.contrib.summary.initialize(graph=tf.get_default_graph())
  # ...
  while not_done_training:
    sess.run([train_op, tf.contrib.summary.all_summary_ops()])
    # ...
```
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce summary (import-module "tensorflow.contrib.summary.summary"))

(defn all-summary-ops 
  "Returns all V2-style summary ops defined in the current default graph.

  This includes ops from TF 2.0 tf.summary and TF 1.x tf.contrib.summary (except
  for `tf.contrib.summary.graph` and `tf.contrib.summary.import_event`), but
  does *not* include TF 1.x tf.summary ops.

  Returns:
    List of summary ops, or None if called under eager execution.
  "
  [  ]
  (py/call-attr summary "all_summary_ops"  ))

(defn always-record-summaries 
  "Sets the should_record_summaries Tensor to always true."
  [  ]
  (py/call-attr summary "always_record_summaries"  ))

(defn audio 
  "Writes an audio summary if possible."
  [ name tensor sample_rate max_outputs family step ]
  (py/call-attr summary "audio"  name tensor sample_rate max_outputs family step ))

(defn create-db-writer 
  "Creates a summary database writer in the current context.

  This can be used to write tensors from the execution graph directly
  to a database. Only SQLite is supported right now. This function
  will create the schema if it doesn't exist. Entries in the Users,
  Experiments, and Runs tables will be created automatically if they
  don't already exist.

  Args:
    db_uri: For example \"file:/tmp/foo.sqlite\".
    experiment_name: Defaults to YYYY-MM-DD in local time if None.
      Empty string means the Run will not be associated with an
      Experiment. Can't contain ASCII control characters or <>. Case
      sensitive.
    run_name: Defaults to HH:MM:SS in local time if None. Empty string
      means a Tag will not be associated with any Run. Can't contain
      ASCII control characters or <>. Case sensitive.
    user_name: Defaults to system username if None. Empty means the
      Experiment will not be associated with a User. Must be valid as
      both a DNS label and Linux username.
    name: Shared name for this SummaryWriter resource stored to default
      `tf.Graph`.

  Returns:
    A `tf.summary.SummaryWriter` instance.
  "
  [ db_uri experiment_name run_name user_name name ]
  (py/call-attr summary "create_db_writer"  db_uri experiment_name run_name user_name name ))

(defn create-file-writer 
  "Creates a summary file writer in the current context under the given name.

  Args:
    logdir: a string, or None. If a string, creates a summary file writer
     which writes to the directory named by the string. If None, returns
     a mock object which acts like a summary writer but does nothing,
     useful to use as a context manager.
    max_queue: the largest number of summaries to keep in a queue; will
     flush once the queue gets bigger than this. Defaults to 10.
    flush_millis: the largest interval between flushes. Defaults to 120,000.
    filename_suffix: optional suffix for the event file name. Defaults to `.v2`.
    name: Shared name for this SummaryWriter resource stored to default
      Graph. Defaults to the provided logdir prefixed with `logdir:`. Note: if a
      summary writer resource with this shared name already exists, the returned
      SummaryWriter wraps that resource and the other arguments have no effect.

  Returns:
    Either a summary writer or an empty object which can be used as a
    summary writer.
  "
  [ logdir max_queue flush_millis filename_suffix name ]
  (py/call-attr summary "create_file_writer"  logdir max_queue flush_millis filename_suffix name ))

(defn create-summary-file-writer 
  "Please use `tf.contrib.summary.create_file_writer`. (deprecated)

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Renamed to create_file_writer()."
  [  ]
  (py/call-attr summary "create_summary_file_writer"  ))

(defn eval-dir 
  "Construct a logdir for an eval summary writer."
  [ model_dir name ]
  (py/call-attr summary "eval_dir"  model_dir name ))

(defn flush 
  "Forces summary writer to send any buffered data to storage.

  This operation blocks until that finishes.

  Args:
    writer: The `tf.summary.SummaryWriter` resource to flush.
      The thread default will be used if this parameter is None.
      Otherwise a `tf.no_op` is returned.
    name: A name for the operation (optional).

  Returns:
    The created `tf.Operation`.
  "
  [ writer name ]
  (py/call-attr summary "flush"  writer name ))

(defn generic 
  "Writes a tensor summary if possible."
  [ name tensor metadata family step ]
  (py/call-attr summary "generic"  name tensor metadata family step ))

(defn graph 
  "Writes a TensorFlow graph to the summary interface.

  The graph summary is, strictly speaking, not a summary. Conditions
  like `tf.summary.should_record_summaries` do not apply. Only
  a single graph can be associated with a particular run. If multiple
  graphs are written, then only the last one will be considered by
  TensorBoard.

  When not using eager execution mode, the user should consider passing
  the `graph` parameter to `tf.compat.v1.summary.initialize` instead of
  calling this function. Otherwise special care needs to be taken when
  using the graph to record the graph.

  Args:
    param: A `tf.Tensor` containing a serialized graph proto. When
      eager execution is enabled, this function will automatically
      coerce `tf.Graph`, `tf.compat.v1.GraphDef`, and string types.
    step: The global step variable. This doesn't have useful semantics
      for graph summaries, but is used anyway, due to the structure of
      event log files. This defaults to the global step.
    name: A name for the operation (optional).

  Returns:
    The created `tf.Operation` or a `tf.no_op` if summary writing has
    not been enabled for this context.

  Raises:
    TypeError: If `param` isn't already a `tf.Tensor` in graph mode.
  "
  [ param step name ]
  (py/call-attr summary "graph"  param step name ))

(defn histogram 
  "Writes a histogram summary if possible."
  [ name tensor family step ]
  (py/call-attr summary "histogram"  name tensor family step ))

(defn image 
  "Writes an image summary if possible."
  [name tensor bad_color & {:keys [max_images family step]
                       :or {family None step None}} ]
    (py/call-attr-kw summary "image" [name tensor bad_color] {:max_images max_images :family family :step step }))

(defn import-event 
  "Writes a `tf.compat.v1.Event` binary proto.

  This can be used to import existing event logs into a new summary writer sink.
  Please note that this is lower level than the other summary functions and
  will ignore the `tf.summary.should_record_summaries` setting.

  Args:
    tensor: A `tf.Tensor` of type `string` containing a serialized
      `tf.compat.v1.Event` proto.
    name: A name for the operation (optional).

  Returns:
    The created `tf.Operation`.
  "
  [ tensor name ]
  (py/call-attr summary "import_event"  tensor name ))

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

(defn never-record-summaries 
  "Sets the should_record_summaries Tensor to always false."
  [  ]
  (py/call-attr summary "never_record_summaries"  ))

(defn record-summaries-every-n-global-steps 
  "Sets the should_record_summaries Tensor to true if global_step % n == 0."
  [ n global_step ]
  (py/call-attr summary "record_summaries_every_n_global_steps"  n global_step ))

(defn scalar 
  "Writes a scalar summary if possible.

  Unlike `tf.contrib.summary.generic` this op may change the dtype
  depending on the writer, for both practical and efficiency concerns.

  Args:
    name: An arbitrary name for this summary.
    tensor: A `tf.Tensor` Must be one of the following types:
      `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`,
      `int8`, `uint16`, `half`, `uint32`, `uint64`.
    family: Optional, the summary's family.
    step: The `int64` monotonic step variable, which defaults
      to `tf.compat.v1.train.get_global_step`.

  Returns:
    The created `tf.Operation` or a `tf.no_op` if summary writing has
    not been enabled for this context.
  "
  [ name tensor family step ]
  (py/call-attr summary "scalar"  name tensor family step ))

(defn should-record-summaries 
  "Returns boolean Tensor which is true if summaries should be recorded."
  [  ]
  (py/call-attr summary "should_record_summaries"  ))

(defn summary-writer-initializer-op 
  "Graph-mode only. Returns the list of ops to create all summary writers.

  Returns:
    The initializer ops.

  Raises:
    RuntimeError: If in Eager mode.
  "
  [  ]
  (py/call-attr summary "summary_writer_initializer_op"  ))
