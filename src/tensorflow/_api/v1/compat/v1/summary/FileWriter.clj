(ns tensorflow.-api.v1.compat.v1.summary.FileWriter
  "Writes `Summary` protocol buffers to event files.

  The `FileWriter` class provides a mechanism to create an event file in a
  given directory and add summaries and events to it. The class updates the
  file contents asynchronously. This allows a training program to call methods
  to add data to the file directly from the training loop, without slowing down
  training.

  When constructed with a `tf.compat.v1.Session` parameter, a `FileWriter`
  instead forms a compatibility layer over new graph-based summaries
  (`tf.contrib.summary`) to facilitate the use of new summary writing with
  pre-existing code that expects a `FileWriter` instance.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce summary (import-module "tensorflow._api.v1.compat.v1.summary"))

(defn FileWriter 
  "Writes `Summary` protocol buffers to event files.

  The `FileWriter` class provides a mechanism to create an event file in a
  given directory and add summaries and events to it. The class updates the
  file contents asynchronously. This allows a training program to call methods
  to add data to the file directly from the training loop, without slowing down
  training.

  When constructed with a `tf.compat.v1.Session` parameter, a `FileWriter`
  instead forms a compatibility layer over new graph-based summaries
  (`tf.contrib.summary`) to facilitate the use of new summary writing with
  pre-existing code that expects a `FileWriter` instance.
  "
  [logdir graph & {:keys [max_queue flush_secs graph_def filename_suffix session]
                       :or {graph_def None filename_suffix None session None}} ]
    (py/call-attr-kw summary "FileWriter" [logdir graph] {:max_queue max_queue :flush_secs flush_secs :graph_def graph_def :filename_suffix filename_suffix :session session }))

(defn add-event 
  "Adds an event to the event file.

    Args:
      event: An `Event` protocol buffer.
    "
  [ self event ]
  (py/call-attr self "add_event"  self event ))

(defn add-graph 
  "Adds a `Graph` to the event file.

    The graph described by the protocol buffer will be displayed by
    TensorBoard. Most users pass a graph in the constructor instead.

    Args:
      graph: A `Graph` object, such as `sess.graph`.
      global_step: Number. Optional global step counter to record with the
        graph.
      graph_def: DEPRECATED. Use the `graph` parameter instead.

    Raises:
      ValueError: If both graph and graph_def are passed to the method.
    "
  [ self graph global_step graph_def ]
  (py/call-attr self "add_graph"  self graph global_step graph_def ))

(defn add-meta-graph 
  "Adds a `MetaGraphDef` to the event file.

    The `MetaGraphDef` allows running the given graph via
    `saver.import_meta_graph()`.

    Args:
      meta_graph_def: A `MetaGraphDef` object, often as returned by
        `saver.export_meta_graph()`.
      global_step: Number. Optional global step counter to record with the
        graph.

    Raises:
      TypeError: If both `meta_graph_def` is not an instance of `MetaGraphDef`.
    "
  [ self meta_graph_def global_step ]
  (py/call-attr self "add_meta_graph"  self meta_graph_def global_step ))

(defn add-run-metadata 
  "Adds a metadata information for a single session.run() call.

    Args:
      run_metadata: A `RunMetadata` protobuf object.
      tag: The tag name for this metadata.
      global_step: Number. Optional global step counter to record with the
        StepStats.

    Raises:
      ValueError: If the provided tag was already used for this type of event.
    "
  [ self run_metadata tag global_step ]
  (py/call-attr self "add_run_metadata"  self run_metadata tag global_step ))

(defn add-session-log 
  "Adds a `SessionLog` protocol buffer to the event file.

    This method wraps the provided session in an `Event` protocol buffer
    and adds it to the event file.

    Args:
      session_log: A `SessionLog` protocol buffer.
      global_step: Number. Optional global step value to record with the
        summary.
    "
  [ self session_log global_step ]
  (py/call-attr self "add_session_log"  self session_log global_step ))

(defn add-summary 
  "Adds a `Summary` protocol buffer to the event file.

    This method wraps the provided summary in an `Event` protocol buffer
    and adds it to the event file.

    You can pass the result of evaluating any summary op, using
    `tf.Session.run` or
    `tf.Tensor.eval`, to this
    function. Alternatively, you can pass a `tf.compat.v1.Summary` protocol
    buffer that you populate with your own data. The latter is
    commonly done to report evaluation results in event files.

    Args:
      summary: A `Summary` protocol buffer, optionally serialized as a string.
      global_step: Number. Optional global step value to record with the
        summary.
    "
  [ self summary global_step ]
  (py/call-attr self "add_summary"  self summary global_step ))

(defn close 
  "Flushes the event file to disk and close the file.

    Call this method when you do not need the summary writer anymore.
    "
  [ self  ]
  (py/call-attr self "close"  self  ))

(defn flush 
  "Flushes the event file to disk.

    Call this method to make sure that all pending events have been written to
    disk.
    "
  [ self  ]
  (py/call-attr self "flush"  self  ))

(defn get-logdir 
  "Returns the directory where event file will be written."
  [ self  ]
  (py/call-attr self "get_logdir"  self  ))

(defn reopen 
  "Reopens the EventFileWriter.

    Can be called after `close()` to add more events in the same directory.
    The events will go into a new events file.

    Does nothing if the EventFileWriter was not closed.
    "
  [ self  ]
  (py/call-attr self "reopen"  self  ))
