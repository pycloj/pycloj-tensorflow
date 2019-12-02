(ns tensorflow.contrib.learn.python.learn.learn-io.data-feeder.DataFeeder
  "Data feeder is an example class to sample data for TF trainer.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.
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

(defn DataFeeder 
  "Data feeder is an example class to sample data for TF trainer.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.
  "
  [x y n_classes batch_size & {:keys [shuffle random_state epochs]
                       :or {random_state None epochs None}} ]
    (py/call-attr-kw data-feeder "DataFeeder" [x y n_classes batch_size] {:shuffle shuffle :random_state random_state :epochs epochs }))

(defn batch-size 
  ""
  [ self ]
    (py/call-attr self "batch_size"))

(defn get-feed-dict-fn 
  "Returns a function that samples data into given placeholders.

    Returns:
      A function that when called samples a random subset of batch size
      from `x` and `y`.
    "
  [ self  ]
  (py/call-attr self "get_feed_dict_fn"  self  ))

(defn get-feed-params 
  "Function returns a `dict` with data feed params while training.

    Returns:
      A `dict` with data feed params while training.
    "
  [ self  ]
  (py/call-attr self "get_feed_params"  self  ))

(defn input-builder 
  "Builds inputs in the graph.

    Returns:
      Two placeholders for inputs and outputs.
    "
  [ self  ]
  (py/call-attr self "input_builder"  self  ))

(defn input-dtype 
  ""
  [ self ]
    (py/call-attr self "input_dtype"))

(defn make-epoch-variable 
  "Adds a placeholder variable for the epoch to the graph.

    Returns:
      The epoch placeholder.
    "
  [ self  ]
  (py/call-attr self "make_epoch_variable"  self  ))

(defn output-dtype 
  ""
  [ self ]
    (py/call-attr self "output_dtype"))

(defn set-placeholders 
  "Sets placeholders for this data feeder.

    Args:
      input_placeholder: Placeholder for `x` variable. Should match shape
        of the examples in the x dataset.
      output_placeholder: Placeholder for `y` variable. Should match
        shape of the examples in the y dataset. Can be `None`.
    "
  [ self input_placeholder output_placeholder ]
  (py/call-attr self "set_placeholders"  self input_placeholder output_placeholder ))

(defn shuffle 
  ""
  [ self ]
    (py/call-attr self "shuffle"))

(defn x 
  ""
  [ self ]
    (py/call-attr self "x"))

(defn y 
  ""
  [ self ]
    (py/call-attr self "y"))
