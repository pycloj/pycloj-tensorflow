(ns tensorflow.python.keras.api.-v1.keras.callbacks.TensorBoard
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
      write_grads: whether to visualize gradient histograms in TensorBoard.
        `histogram_freq` must be greater than 0.
      batch_size: size of batch of inputs to feed to the network for histograms
        computation.
      write_images: whether to write model weights to visualize as image in
        TensorBoard.
      embeddings_freq: frequency (in epochs) at which selected embedding layers
        will be saved. If set to 0, embeddings won't be computed. Data to be
        visualized in TensorBoard's Embedding tab must be passed as
        `embeddings_data`.
      embeddings_layer_names: a list of names of layers to keep eye on. If None
        or empty list all the embedding layer will be watched.
      embeddings_metadata: a dictionary which maps layer name to a file name in
        which metadata for this embedding layer is saved. See the
          [details](https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
            about metadata files format. In case if the same metadata file is
            used for all embedding layers, string can be passed.
      embeddings_data: data to be embedded at layers specified in
        `embeddings_layer_names`. Numpy array (if the model has a single input)
        or list of Numpy arrays (if the model has multiple inputs). Learn [more
        about
            embeddings](https://www.tensorflow.org/programmers_guide/embedding)
      update_freq: `'batch'` or `'epoch'` or integer. When using `'batch'`,
        writes the losses and metrics to TensorBoard after each batch. The same
        applies for `'epoch'`. If using an integer, let's say `1000`, the
        callback will write the metrics and losses to TensorBoard every 1000
        samples. Note that writing too frequently to TensorBoard can slow down
        your training.
      profile_batch: Profile the batch to sample compute characteristics. By
        default, it will profile the second batch. Set profile_batch=0 to
        disable profiling.

  Raises:
      ValueError: If histogram_freq is set and no validation data is provided.

  @compatibility(eager)
  Using the `TensorBoard` callback will work when eager execution is enabled,
  with the restriction that outputting histogram summaries of weights and
  gradients is not supported. Consequently, `histogram_freq` will be ignored.
  @end_compatibility
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce callbacks (import-module "tensorflow.python.keras.api._v1.keras.callbacks"))

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
      write_grads: whether to visualize gradient histograms in TensorBoard.
        `histogram_freq` must be greater than 0.
      batch_size: size of batch of inputs to feed to the network for histograms
        computation.
      write_images: whether to write model weights to visualize as image in
        TensorBoard.
      embeddings_freq: frequency (in epochs) at which selected embedding layers
        will be saved. If set to 0, embeddings won't be computed. Data to be
        visualized in TensorBoard's Embedding tab must be passed as
        `embeddings_data`.
      embeddings_layer_names: a list of names of layers to keep eye on. If None
        or empty list all the embedding layer will be watched.
      embeddings_metadata: a dictionary which maps layer name to a file name in
        which metadata for this embedding layer is saved. See the
          [details](https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
            about metadata files format. In case if the same metadata file is
            used for all embedding layers, string can be passed.
      embeddings_data: data to be embedded at layers specified in
        `embeddings_layer_names`. Numpy array (if the model has a single input)
        or list of Numpy arrays (if the model has multiple inputs). Learn [more
        about
            embeddings](https://www.tensorflow.org/programmers_guide/embedding)
      update_freq: `'batch'` or `'epoch'` or integer. When using `'batch'`,
        writes the losses and metrics to TensorBoard after each batch. The same
        applies for `'epoch'`. If using an integer, let's say `1000`, the
        callback will write the metrics and losses to TensorBoard every 1000
        samples. Note that writing too frequently to TensorBoard can slow down
        your training.
      profile_batch: Profile the batch to sample compute characteristics. By
        default, it will profile the second batch. Set profile_batch=0 to
        disable profiling.

  Raises:
      ValueError: If histogram_freq is set and no validation data is provided.

  @compatibility(eager)
  Using the `TensorBoard` callback will work when eager execution is enabled,
  with the restriction that outputting histogram summaries of weights and
  gradients is not supported. Consequently, `histogram_freq` will be ignored.
  @end_compatibility
  "
  [ & {:keys [log_dir histogram_freq batch_size write_graph write_grads write_images embeddings_freq embeddings_layer_names embeddings_metadata embeddings_data update_freq profile_batch]
       :or {embeddings_layer_names None embeddings_metadata None embeddings_data None}} ]
  
   (py/call-attr-kw callbacks "TensorBoard" [] {:log_dir log_dir :histogram_freq histogram_freq :batch_size batch_size :write_graph write_graph :write_grads write_grads :write_images write_images :embeddings_freq embeddings_freq :embeddings_layer_names embeddings_layer_names :embeddings_metadata embeddings_metadata :embeddings_data embeddings_data :update_freq update_freq :profile_batch profile_batch }))

(defn on-batch-begin 
  "A backwards compatibility alias for `on_train_batch_begin`."
  [ self batch logs ]
  (py/call-attr self "on_batch_begin"  self batch logs ))

(defn on-batch-end 
  "Writes scalar summaries for metrics on every training batch.

    Performs profiling if current batch is in profiler_batches.
    "
  [ self batch logs ]
  (py/call-attr self "on_batch_end"  self batch logs ))

(defn on-epoch-begin 
  "Add histogram op to Model eval_function callbacks, reset batch count."
  [ self epoch logs ]
  (py/call-attr self "on_epoch_begin"  self epoch logs ))

(defn on-epoch-end 
  "Checks if summary ops should run next epoch, logs scalar summaries."
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
  "Called at the end of a batch in `evaluate` methods.

    Also called at the end of a validation batch in the `fit`
    methods, if validation data is provided.

    Subclasses should override for any actions to run.

    Arguments:
        batch: integer, index of batch within the current epoch.
        logs: dict. Metric results for this batch.
    "
  [ self batch logs ]
  (py/call-attr self "on_test_batch_end"  self batch logs ))

(defn on-test-begin 
  "Called at the beginning of evaluation or validation.

    Subclasses should override for any actions to run.

    Arguments:
        logs: dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    "
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
  "Called at the end of a training batch in `fit` methods.

    Subclasses should override for any actions to run.

    Arguments:
        batch: integer, index of batch within the current epoch.
        logs: dict. Metric results for this batch.
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
  "Sets Keras model and creates summary ops."
  [ self model ]
  (py/call-attr self "set_model"  self model ))

(defn set-params 
  ""
  [ self params ]
  (py/call-attr self "set_params"  self params ))
