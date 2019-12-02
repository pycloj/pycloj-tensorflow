(ns tensorflow.contrib.learn.python.learn.learn-io.data-feeder.StreamingDataFeeder
  "Data feeder for TF trainer that reads data from iterator.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.

  Streaming data feeder allows to read data as it comes it from disk or
  somewhere else. It's custom to have this iterators rotate infinetly over
  the dataset, to allow control of how much to learn on the trainer side.
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

(defn StreamingDataFeeder 
  "Data feeder for TF trainer that reads data from iterator.

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.

  Streaming data feeder allows to read data as it comes it from disk or
  somewhere else. It's custom to have this iterators rotate infinetly over
  the dataset, to allow control of how much to learn on the trainer side.
  "
  [ x y n_classes batch_size ]
  (py/call-attr data-feeder "StreamingDataFeeder"  x y n_classes batch_size ))

(defn batch-size 
  ""
  [ self ]
    (py/call-attr self "batch_size"))

(defn get-feed-dict-fn 
  "Returns a function, that will sample data and provide it to placeholders.

    Returns:
      A function that when called samples a random subset of batch size
      from x and y.
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
