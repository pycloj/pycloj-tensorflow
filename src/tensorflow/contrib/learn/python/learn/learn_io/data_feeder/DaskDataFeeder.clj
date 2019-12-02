(ns tensorflow.contrib.learn.python.learn.learn-io.data-feeder.DaskDataFeeder
  "Data feeder for that reads data from dask.Series and dask.DataFrame.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.

  Numpy arrays can be serialized to disk and it's possible to do random seeks
  into them. DaskDataFeeder will remove requirement to have full dataset in the
  memory and still do random seeks for sampling of batches.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce data-feeder (import-module "tensorflow.contrib.learn.python.learn.learn_io.data_feeder"))

(defn DaskDataFeeder 
  "Data feeder for that reads data from dask.Series and dask.DataFrame.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.

  Numpy arrays can be serialized to disk and it's possible to do random seeks
  into them. DaskDataFeeder will remove requirement to have full dataset in the
  memory and still do random seeks for sampling of batches.
  "
  [x y n_classes batch_size & {:keys [shuffle random_state epochs]
                       :or {random_state None epochs None}} ]
    (py/call-attr-kw data-feeder "DaskDataFeeder" [x y n_classes batch_size] {:shuffle shuffle :random_state random_state :epochs epochs }))

(defn get-feed-dict-fn 
  "Returns a function, that will sample data and provide it to placeholders.

    Args:
      input_placeholder: tf.compat.v1.placeholder for input features mini batch.
      output_placeholder: tf.compat.v1.placeholder for output labels.

    Returns:
      A function that when called samples a random subset of batch size
      from x and y.
    "
  [ self input_placeholder output_placeholder ]
  (py/call-attr self "get_feed_dict_fn"  self input_placeholder output_placeholder ))

(defn get-feed-params 
  "Function returns a `dict` with data feed params while training.

    Returns:
      A `dict` with data feed params while training.
    "
  [ self  ]
  (py/call-attr self "get_feed_params"  self  ))
