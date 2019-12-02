(ns tensorflow.-api.v1.compat.v1.data.experimental.DistributeOptions
  "Represents options for distributed data processing.

  You can set the distribution options of a dataset through the
  `experimental_distribute` property of `tf.data.Options`; the property is
  an instance of `tf.data.experimental.DistributeOptions`.

  ```python
  options = tf.data.Options()
  options.experimental_distribute.auto_shard = False
  dataset = dataset.with_options(options)
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
(defonce experimental (import-module "tensorflow._api.v1.compat.v1.data.experimental"))

(defn DistributeOptions 
  "Represents options for distributed data processing.

  You can set the distribution options of a dataset through the
  `experimental_distribute` property of `tf.data.Options`; the property is
  an instance of `tf.data.experimental.DistributeOptions`.

  ```python
  options = tf.data.Options()
  options.experimental_distribute.auto_shard = False
  dataset = dataset.with_options(options)
  ```
  "
  [  ]
  (py/call-attr experimental "DistributeOptions"  ))

(defn auto-shard 
  "Whether the dataset should be automatically sharded when processedin a distributed fashion. This is applicable when using Keras with multi-worker/TPU distribution strategy, and by using strategy.experimental_distribute_dataset(). In other cases, this option does nothing. If None, defaults to True."
  [ self ]
    (py/call-attr self "auto_shard"))

(defn num-devices 
  "The number of devices attached to this input pipeline. This will be automatically set by MultiDeviceIterator."
  [ self ]
    (py/call-attr self "num_devices"))
