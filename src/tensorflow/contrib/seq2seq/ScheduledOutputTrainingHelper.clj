(ns tensorflow.contrib.seq2seq.ScheduledOutputTrainingHelper
  "A training helper that adds scheduled sampling directly to outputs.

  Returns False for sample_ids where no sampling took place; True elsewhere.
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

(defn ScheduledOutputTrainingHelper 
  "A training helper that adds scheduled sampling directly to outputs.

  Returns False for sample_ids where no sampling took place; True elsewhere.
  "
  [inputs sequence_length sampling_probability & {:keys [time_major seed next_inputs_fn auxiliary_inputs name]
                       :or {seed None next_inputs_fn None auxiliary_inputs None name None}} ]
    (py/call-attr-kw seq2seq "ScheduledOutputTrainingHelper" [inputs sequence_length sampling_probability] {:time_major time_major :seed seed :next_inputs_fn next_inputs_fn :auxiliary_inputs auxiliary_inputs :name name }))

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

(defn sequence-length 
  ""
  [ self ]
    (py/call-attr self "sequence_length"))
