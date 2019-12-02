(ns tensorflow.contrib.seq2seq.InferenceHelper
  "A helper to use during inference with a custom sampling function."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce seq2seq (import-module "tensorflow.contrib.seq2seq"))

(defn InferenceHelper 
  "A helper to use during inference with a custom sampling function."
  [ sample_fn sample_shape sample_dtype start_inputs end_fn next_inputs_fn ]
  (py/call-attr seq2seq "InferenceHelper"  sample_fn sample_shape sample_dtype start_inputs end_fn next_inputs_fn ))

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
