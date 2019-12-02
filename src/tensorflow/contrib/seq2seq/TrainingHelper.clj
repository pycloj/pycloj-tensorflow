(ns tensorflow.contrib.seq2seq.TrainingHelper
  "A helper for use during training.  Only reads inputs.

  Returned sample_ids are the argmax of the RNN output logits.
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

(defn TrainingHelper 
  "A helper for use during training.  Only reads inputs.

  Returned sample_ids are the argmax of the RNN output logits.
  "
  [inputs sequence_length & {:keys [time_major name]
                       :or {name None}} ]
    (py/call-attr-kw seq2seq "TrainingHelper" [inputs sequence_length] {:time_major time_major :name name }))

(defn batch-size 
  ""
  [ self ]
    (py/call-attr self "batch_size"))

(defn initialize 
  ""
  [ self name ]
  (py/call-attr self "initialize"  self name ))

(defn inputs 
  ""
  [ self ]
    (py/call-attr self "inputs"))

(defn next-inputs 
  "next_inputs_fn for TrainingHelper."
  [ self time outputs state name ]
  (py/call-attr self "next_inputs"  self time outputs state name ))

(defn sample 
  ""
  [ self time outputs name ]
  (py/call-attr self "sample"  self time outputs name ))

(defn sample-ids-dtype 
  ""
  [ self ]
    (py/call-attr self "sample_ids_dtype"))

(defn sample-ids-shape 
  ""
  [ self ]
    (py/call-attr self "sample_ids_shape"))

(defn sequence-length 
  ""
  [ self ]
    (py/call-attr self "sequence_length"))
