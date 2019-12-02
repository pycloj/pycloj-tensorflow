(ns tensorflow.contrib.seq2seq.CustomHelper
  "Base abstract class that allows the user to customize sampling."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce seq2seq (import-module "tensorflow.contrib.seq2seq"))

(defn CustomHelper 
  "Base abstract class that allows the user to customize sampling."
  [ initialize_fn sample_fn next_inputs_fn sample_ids_shape sample_ids_dtype ]
  (py/call-attr seq2seq "CustomHelper"  initialize_fn sample_fn next_inputs_fn sample_ids_shape sample_ids_dtype ))

(defn batch-size 
  ""
  [ self ]
    (py/call-attr self "batch_size"))

(defn initialize 
  ""
  [ self name ]
  (py/call-attr self "initialize"  self name ))

(defn next-inputs 
  ""
  [ self time outputs state sample_ids name ]
  (py/call-attr self "next_inputs"  self time outputs state sample_ids name ))

(defn sample 
  ""
  [ self time outputs state name ]
  (py/call-attr self "sample"  self time outputs state name ))

(defn sample-ids-dtype 
  ""
  [ self ]
    (py/call-attr self "sample_ids_dtype"))

(defn sample-ids-shape 
  ""
  [ self ]
    (py/call-attr self "sample_ids_shape"))
