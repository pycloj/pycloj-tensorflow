(ns tensorflow.contrib.keras.api.keras.callbacks.LambdaCallback
  "Callback for creating simple, custom callbacks on-the-fly.

  This callback is constructed with anonymous functions that will be called
  at the appropriate time. Note that the callbacks expects positional
  arguments, as:

   - `on_epoch_begin` and `on_epoch_end` expect two positional arguments:
      `epoch`, `logs`
   - `on_batch_begin` and `on_batch_end` expect two positional arguments:
      `batch`, `logs`
   - `on_train_begin` and `on_train_end` expect one positional argument:
      `logs`

  Arguments:
      on_epoch_begin: called at the beginning of every epoch.
      on_epoch_end: called at the end of every epoch.
      on_batch_begin: called at the beginning of every batch.
      on_batch_end: called at the end of every batch.
      on_train_begin: called at the beginning of model training.
      on_train_end: called at the end of model training.

  Example:

  ```python
  # Print the batch number at the beginning of every batch.
  batch_print_callback = LambdaCallback(
      on_batch_begin=lambda batch,logs: print(batch))

  # Stream the epoch loss to a file in JSON format. The file content
  # is not well-formed JSON but rather has a JSON object per line.
  import json
  json_log = open('loss_log.json', mode='wt', buffering=1)
  json_logging_callback = LambdaCallback(
      on_epoch_end=lambda epoch, logs: json_log.write(
          json.dumps({'epoch': epoch, 'loss': logs['loss']}) + '\n'),
      on_train_end=lambda logs: json_log.close()
  )

  # Terminate some processes after having finished model training.
  processes = ...
  cleanup_callback = LambdaCallback(
      on_train_end=lambda logs: [
          p.terminate() for p in processes if p.is_alive()])

  model.fit(...,
            callbacks=[batch_print_callback,
                       json_logging_callback,
                       cleanup_callback])
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
(defonce callbacks (import-module "tensorflow.contrib.keras.api.keras.callbacks"))

(defn LambdaCallback 
  "Callback for creating simple, custom callbacks on-the-fly.

  This callback is constructed with anonymous functions that will be called
  at the appropriate time. Note that the callbacks expects positional
  arguments, as:

   - `on_epoch_begin` and `on_epoch_end` expect two positional arguments:
      `epoch`, `logs`
   - `on_batch_begin` and `on_batch_end` expect two positional arguments:
      `batch`, `logs`
   - `on_train_begin` and `on_train_end` expect one positional argument:
      `logs`

  Arguments:
      on_epoch_begin: called at the beginning of every epoch.
      on_epoch_end: called at the end of every epoch.
      on_batch_begin: called at the beginning of every batch.
      on_batch_end: called at the end of every batch.
      on_train_begin: called at the beginning of model training.
      on_train_end: called at the end of model training.

  Example:

  ```python
  # Print the batch number at the beginning of every batch.
  batch_print_callback = LambdaCallback(
      on_batch_begin=lambda batch,logs: print(batch))

  # Stream the epoch loss to a file in JSON format. The file content
  # is not well-formed JSON but rather has a JSON object per line.
  import json
  json_log = open('loss_log.json', mode='wt', buffering=1)
  json_logging_callback = LambdaCallback(
      on_epoch_end=lambda epoch, logs: json_log.write(
          json.dumps({'epoch': epoch, 'loss': logs['loss']}) + '\n'),
      on_train_end=lambda logs: json_log.close()
  )

  # Terminate some processes after having finished model training.
  processes = ...
  cleanup_callback = LambdaCallback(
      on_train_end=lambda logs: [
          p.terminate() for p in processes if p.is_alive()])

  model.fit(...,
            callbacks=[batch_print_callback,
                       json_logging_callback,
                       cleanup_callback])
  ```
  "
  [ on_epoch_begin on_epoch_end on_batch_begin on_batch_end on_train_begin on_train_end ]
  (py/call-attr callbacks "LambdaCallback"  on_epoch_begin on_epoch_end on_batch_begin on_batch_end on_train_begin on_train_end ))

(defn on-batch-begin 
  "A backwards compatibility alias for `on_train_batch_begin`."
  [ self batch logs ]
  (py/call-attr self "on_batch_begin"  self batch logs ))

(defn on-batch-end 
  "A backwards compatibility alias for `on_train_batch_end`."
  [ self batch logs ]
  (py/call-attr self "on_batch_end"  self batch logs ))

(defn on-epoch-begin 
  "Called at the start of an epoch.

    Subclasses should override for any actions to run. This function should only
    be called during TRAIN mode.

    Arguments:
        epoch: integer, index of epoch.
        logs: dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    "
  [ self epoch logs ]
  (py/call-attr self "on_epoch_begin"  self epoch logs ))

(defn on-epoch-end 
  "Called at the end of an epoch.

    Subclasses should override for any actions to run. This function should only
    be called during TRAIN mode.

    Arguments:
        epoch: integer, index of epoch.
        logs: dict, metric results for this training epoch, and for the
          validation epoch if validation is performed. Validation result keys
          are prefixed with `val_`.
    "
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
  "Called at the beginning of training.

    Subclasses should override for any actions to run.

    Arguments:
        logs: dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    "
  [ self logs ]
  (py/call-attr self "on_train_begin"  self logs ))

(defn on-train-end 
  "Called at the end of training.

    Subclasses should override for any actions to run.

    Arguments:
        logs: dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    "
  [ self logs ]
  (py/call-attr self "on_train_end"  self logs ))

(defn set-model 
  ""
  [ self model ]
  (py/call-attr self "set_model"  self model ))

(defn set-params 
  ""
  [ self params ]
  (py/call-attr self "set_params"  self params ))
