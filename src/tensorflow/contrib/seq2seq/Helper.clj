(ns tensorflow.contrib.seq2seq.Helper
  "Interface for implementing sampling in seq2seq decoders.

  Helper instances are used by `BasicDecoder`.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce seq2seq (import-module "tensorflow.contrib.seq2seq"))

(defn Helper 
  "Interface for implementing sampling in seq2seq decoders.

  Helper instances are used by `BasicDecoder`.
  "
  [  ]
  (py/call-attr seq2seq "Helper"  ))

(defn batch-size 
  "Batch size of tensor returned by `sample`.

    Returns a scalar int32 tensor.
    "
  [ self ]
    (py/call-attr self "batch_size"))

(defn initialize 
  "Returns `(initial_finished, initial_inputs)`."
  [ self name ]
  (py/call-attr self "initialize"  self name ))

(defn next-inputs 
  "Returns `(finished, next_inputs, next_state)`."
  [ self time outputs state sample_ids name ]
  (py/call-attr self "next_inputs"  self time outputs state sample_ids name ))

(defn sample 
  "Returns `sample_ids`."
  [ self time outputs state name ]
  (py/call-attr self "sample"  self time outputs state name ))

(defn sample-ids-dtype 
  "DType of tensor returned by `sample`.

    Returns a DType.
    "
  [ self ]
    (py/call-attr self "sample_ids_dtype"))

(defn sample-ids-shape 
  "Shape of tensor returned by `sample`, excluding the batch dimension.

    Returns a `TensorShape`.
    "
  [ self ]
    (py/call-attr self "sample_ids_shape"))
