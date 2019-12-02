(ns tensorflow.contrib.keras.api.keras.callbacks.TensorBoard
  "Enable visualizations for TensorBoard.

  TensorBoard is a visualization tool provided with TensorFlow.

  This callback logs events for TensorBoard, including:
  * Metrics summary plots
  * Training graph visualization
  * Activation histograms
  * Sampled profiling

  If you have installed TensorFlow with pip, you should be able
  to launch TensorBoard from the command line:

  ```sh
  tensorboard --logdir=path_to_your_logs
  ```

  You can find more information about TensorBoard
  [here](https://www.tensorflow.org/get_started/summaries_and_tensorboard).

  Arguments:
      log_dir: the path of the directory where to save the log files to be
        parsed by TensorBoard.
      histogram_freq: frequency (in epochs) at which to compute activation and
        weight histograms for the layers of the model. If set to 0, histograms
        won't be computed. Validation data (or split) must be specified for
        histogram visualizations.
      write_graph: whether to visualize the graph in TensorBoard. The log file
        can become quite large when write_graph is set to True.
      write_images: whether to write model weights to visualize as image in
        TensorBoard.
      update_freq: `'batch'` or `'epoch'` or integer. When using `'batch'`,
        writes the losses and metrics to TensorBoard after each batch. The same
        applies for `'epoch'`. If using an integer, let's say `1000`, the
        callback will write the metrics and losses to TensorBoard every 1000
        batches. Note that writing too frequently to TensorBoard can slow down
        your training.
      profile_batch: Profile the batch to sample compute characteristics. By
        default, it will profile the second batch. Set profile_batch=0 to
        disable profiling. Must run in TensorFlow eager mode.
      embeddings_freq: frequency (in epochs) at which embedding layers will
        be visualized. If set to 0, embeddings won't be visualized.
      embeddings_metadata: a dictionary which maps layer name to a file name in
        which metadata for this embedding layer is saved. See the
        [details](
          https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
        about metadata files format. In case if the same metadata file is
        used for all embedding layers, string can be passed.

  Raises:
      ValueError: If histogram_freq is set and no validation data is provided.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce callbacks (import-module "tensorflow.contrib.keras.api.keras.callbacks"))

(defn TensorBoard 
  "Enable visualizations for TensorBoard.

  TensorBoard is a visualization tool provided with TensorFlow.

  This callback logs events for TensorBoard, including:
  * Metrics summary plots
  * Training graph visualization
  * Activation histograms
  * Sampled profiling

  If you have installed TensorFlow with pip, you should be able
  to launch TensorBoard from the command line:

  ```sh
  tensorboard --logdir=path_to_your_logs
  ```

  You can find more information about TensorBoard
  [here](https://www.tensorflow.org/get_started/summaries_and_tensorboard).

  Arguments:
      log_dir: the path of the directory where to save the log files to be
        parsed by TensorBoard.
      histogram_freq: frequency (in epochs) at which to compute activation and
        weight histograms for the layers of the model. If set to 0, histograms
        won't be computed. Validation data (or split) must be specified for
        histogram visualizations.
      write_graph: whether to visualize the graph in TensorBoard. The log file
        can become quite large when write_graph is set to True.
      write_images: whether to write model weights to visualize as image in
        TensorBoard.
      update_freq: `'batch'` or `'epoch'` or integer. When using `'batch'`,
        writes the losses and metrics to TensorBoard after each batch. The same
        applies for `'epoch'`. If using an integer, let's say `1000`, the
        callback will write the metrics and losses to TensorBoard every 1000
        batches. Note that writing too frequently to TensorBoard can slow down
        your training.
      profile_batch: Profile the batch to sample compute characteristics. By
        default, it will profile the second batch. Set profile_batch=0 to
        disable profiling. Must run in TensorFlow eager mode.
      embeddings_freq: frequency (in epochs) at which embedding layers will
        be visualized. If set to 0, embeddings won't be visualized.
      embeddings_metadata: a dictionary which maps layer name to a file name in
        which metadata for this embedding layer is saved. See the
        [details](
          https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
        about metadata files format. In case if the same metadata file is
        used for all embedding layers, string can be passed.

  Raises:
      ValueError: If histogram_freq is set and no validation data is provided.
  "
  [ & {:keys [log_dir histogram_freq write_graph write_images update_freq profile_batch embeddings_freq embeddings_metadata]
       :or {embeddings_metadata None}} ]
  
   (py/call-attr-kw callbacks "TensorBoard" [] {:log_dir log_dir :histogram_freq histogram_freq :write_graph write_graph :write_images write_images :update_freq update_freq :profile_batch profile_batch :embeddings_freq embeddings_freq :embeddings_metadata embeddings_metadata }))

(defn on-batch-begin 
  "A backwards compatibility alias for `on_train_batch_begin`."
  [ self batch logs ]
  (py/call-attr self "on_batch_begin"  self batch logs ))

(defn on-batch-end 
  "A backwards compatibility alias for `on_train_batch_end`."
  [ self batch logs ]
  (py/call-attr self "on_batch_end"  self batch logs ))

(defn on-epoch-begin 
  ""
  [ self epoch logs ]
  (py/call-attr self "on_epoch_begin"  self epoch logs ))

(defn on-epoch-end 
  "Runs metrics and histogram summaries at epoch end."
  [ self epoch logs ]
  (py/call-attr self "on_epoch_end"  self epoch logs ))

(defn on-predict-batch-begin 
  "Called at the beginning of a batch in `predict` methods.

    Subclasses should override for any actions to run.

    Arguments:
        batch: integer, index of batch within the current epoch.
        logs: dict. Has keys `batch` and `size` representing the current batch
          number and the size of the batch.
    "
  [ self batch logs ]
  (py/call-attr self "on_predict_batch_begin"  self batch logs ))

(defn on-predict-batch-end 
  "Called at the end of a batch in `predict` methods.

    Subclasses should override for any actions to run.

    Arguments:
        batch: integer, index of batch within the current epoch.
        logs: dict. Metric results for this batch.
    "
  [ self batch logs ]
  (py/call-attr self "on_predict_batch_end"  self batch logs ))

(defn on-predict-begin 
  "Called at the beginning of prediction.

    Subclasses should override for any actions to run.

    Arguments:
        logs: dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    "
  [ self logs ]
  (py/call-attr self "on_predict_begin"  self logs ))

(defn on-predict-end 
  "Called at the end of prediction.

    Subclasses should override for any actions to run.

    Arguments:
        logs: dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    "
  [ self logs ]
  (py/call-attr self "on_predict_end"  self logs ))

(defn on-test-batch-begin 
  "Called at the beginning of a batch in `evaluate` methods.

    Also called at the beginning of a validation batch in the `fit`
    methods, if validation data is provided.

    Subclasses should override for any actions to run.

    Arguments:
        batch: integer, index of batch within the current epoch.
        logs: dict. Has keys `batch` and `size` representing the current batch
          number and the size of the batch.
    "
  [ self batch logs ]
  (py/call-attr self "on_test_batch_begin"  self batch logs ))

(defn on-test-batch-end 
  ""
  [ self batch logs ]
  (py/call-attr self "on_test_batch_end"  self batch logs ))

(defn on-test-begin 
  ""
  [ self logs ]
  (py/call-attr self "on_test_begin"  self logs ))

(defn on-test-end 
  "Called at the end of evaluation or validation.

    Subclasses should override for any actions to run.

    Arguments:
        logs: dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    "
  [ self logs ]
  (py/call-attr self "on_test_end"  self logs ))

(defn on-train-batch-begin 
  "Called at the beginning of a training batch in `fit` methods.

    Subclasses should override for any actions to run.

    Arguments:
        batch: integer, index of batch within the current epoch.
        logs: dict. Has keys `batch` and `size` representing the current batch
          number and the size of the batch.
    "
  [ self batch logs ]
  (py/call-attr self "on_train_batch_begin"  self batch logs ))

(defn on-train-batch-end 
  "Writes scalar summaries for metrics on every training batch.

    Performs profiling if current batch is in profiler_batches.

    Arguments:
      batch: Integer, index of batch within the current epoch.
      logs: Dict. Metric results for this batch.
    "
  [ self batch logs ]
  (py/call-attr self "on_train_batch_end"  self batch logs ))

(defn on-train-begin 
  ""
  [ self logs ]
  (py/call-attr self "on_train_begin"  self logs ))

(defn on-train-end 
  ""
  [ self logs ]
  (py/call-attr self "on_train_end"  self logs ))

(defn set-model 
  "Sets Keras model and writes graph if specified."
  [ self model ]
  (py/call-attr self "set_model"  self model ))

(defn set-params 
  ""
  [ self params ]
  (py/call-attr self "set_params"  self params ))
