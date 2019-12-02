(ns tensorflow.python.keras.api.-v1.keras.callbacks.CSVLogger
  "Callback that streams epoch results to a csv file.

  Supports all values that can be represented as a string,
  including 1D iterables such as np.ndarray.

  Example:

  ```python
  csv_logger = CSVLogger('training.log')
  model.fit(X_train, Y_train, callbacks=[csv_logger])
  ```

  Arguments:
      filename: filename of the csv file, e.g. 'run/log.csv'.
      separator: string used to separate elements in the csv file.
      append: True: append if file exists (useful for continuing
          training). False: overwrite existing file,
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
(defn CSVLogger 
  "Callback that streams epoch results to a csv file.

  Supports all values that can be represented as a string,
  including 1D iterables such as np.ndarray.

  Example:

  ```python
  csv_logger = CSVLogger('training.log')
  model.fit(X_train, Y_train, callbacks=[csv_logger])
  ```

  Arguments:
      filename: filename of the csv file, e.g. 'run/log.csv'.
      separator: string used to separate elements in the csv file.
      append: True: append if file exists (useful for continuing
          training). False: overwrite existing file,
  "
  [filename  & {:keys [separator append]} ]
    (py/call-attr-kw callbacks "CSVLogger" [filename] {:separator separator :append append }))

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
