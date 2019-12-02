(ns tensorflow.-api.v1.compat.v1.data.experimental.ThreadingOptions
  "Represents options for dataset threading.

  You can set the threading options of a dataset through the
  `experimental_threading` property of `tf.data.Options`; the property is
  an instance of `tf.data.experimental.ThreadingOptions`.

  ```python
  options = tf.data.Options()
  options.experimental_threading.private_threadpool_size = 10
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

(defn ThreadingOptions 
  "Represents options for dataset threading.

  You can set the threading options of a dataset through the
  `experimental_threading` property of `tf.data.Options`; the property is
  an instance of `tf.data.experimental.ThreadingOptions`.

  ```python
  options = tf.data.Options()
  options.experimental_threading.private_threadpool_size = 10
  dataset = dataset.with_options(options)
  ```
  "
  [  ]
  (py/call-attr experimental "ThreadingOptions"  ))

(defn max-intra-op-parallelism 
  "If set, it overrides the maximum degree of intra-op parallelism."
  [ self ]
    (py/call-attr self "max_intra_op_parallelism"))

(defn private-threadpool-size 
  "If set, the dataset will use a private threadpool of the given size."
  [ self ]
    (py/call-attr self "private_threadpool_size"))
