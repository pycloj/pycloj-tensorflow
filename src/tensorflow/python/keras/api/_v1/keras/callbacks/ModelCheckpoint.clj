(ns tensorflow.python.keras.api.-v1.keras.callbacks.ModelCheckpoint
  "Save the model after every epoch.

  `filepath` can contain named formatting options,
  which will be filled the value of `epoch` and
  keys in `logs` (passed in `on_epoch_end`).

  For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
  then the model checkpoints will be saved with the epoch number and
  the validation loss in the filename.

  Arguments:
      filepath: string, path to save the model file.
      monitor: quantity to monitor.
      verbose: verbosity mode, 0 or 1.
      save_best_only: if `save_best_only=True`, the latest best model according
        to the quantity monitored will not be overwritten.
      mode: one of {auto, min, max}. If `save_best_only=True`, the decision to
        overwrite the current save file is made based on either the maximization
        or the minimization of the monitored quantity. For `val_acc`, this
        should be `max`, for `val_loss` this should be `min`, etc. In `auto`
        mode, the direction is automatically inferred from the name of the
        monitored quantity.
      save_weights_only: if True, then only the model's weights will be saved
        (`model.save_weights(filepath)`), else the full model is saved
        (`model.save(filepath)`).
      save_freq: `'epoch'` or integer. When using `'epoch'`, the callback saves
        the model after each epoch. When using integer, the callback saves the
        model at end of a batch at which this many samples have been seen since
        last saving. Note that if the saving isn't aligned to epochs, the
        monitored metric may potentially be less reliable (it could reflect as
        little as 1 batch, since the metrics get reset every epoch). Defaults to
        `'epoch'`
      **kwargs: Additional arguments for backwards compatibility. Possible key
        is `period`.
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
(defn ModelCheckpoint 
  "Save the model after every epoch.

  `filepath` can contain named formatting options,
  which will be filled the value of `epoch` and
  keys in `logs` (passed in `on_epoch_end`).

  For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
  then the model checkpoints will be saved with the epoch number and
  the validation loss in the filename.

  Arguments:
      filepath: string, path to save the model file.
      monitor: quantity to monitor.
      verbose: verbosity mode, 0 or 1.
      save_best_only: if `save_best_only=True`, the latest best model according
        to the quantity monitored will not be overwritten.
      mode: one of {auto, min, max}. If `save_best_only=True`, the decision to
        overwrite the current save file is made based on either the maximization
        or the minimization of the monitored quantity. For `val_acc`, this
        should be `max`, for `val_loss` this should be `min`, etc. In `auto`
        mode, the direction is automatically inferred from the name of the
        monitored quantity.
      save_weights_only: if True, then only the model's weights will be saved
        (`model.save_weights(filepath)`), else the full model is saved
        (`model.save(filepath)`).
      save_freq: `'epoch'` or integer. When using `'epoch'`, the callback saves
        the model after each epoch. When using integer, the callback saves the
        model at end of a batch at which this many samples have been seen since
        last saving. Note that if the saving isn't aligned to epochs, the
        monitored metric may potentially be less reliable (it could reflect as
        little as 1 batch, since the metrics get reset every epoch). Defaults to
        `'epoch'`
      **kwargs: Additional arguments for backwards compatibility. Possible key
        is `period`.
  "
  [filepath  & {:keys [monitor verbose save_best_only save_weights_only mode save_freq]} ]
    (py/call-attr-kw callbacks "ModelCheckpoint" [filepath] {:monitor monitor :verbose verbose :save_best_only save_best_only :save_weights_only save_weights_only :mode mode :save_freq save_freq }))

(defn on-batch-begin 
  "A backwards compatibility alias for `on_train_batch_begin`."
  [ self batch logs ]
  (py/call-attr self "on_batch_begin"  self batch logs ))

(defn on-batch-end 
  ""
  [ self batch logs ]
  (py/call-attr self "on_batch_end"  self batch logs ))

(defn on-epoch-begin 
  ""
  [ self epoch logs ]
  (py/call-attr self "on_epoch_begin"  self epoch logs ))

(defn on-epoch-end 
  ""
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
  ""
  [ self model ]
  (py/call-attr self "set_model"  self model ))

(defn set-params 
  ""
  [ self params ]
  (py/call-attr self "set_params"  self params ))
