(ns tensorflow.contrib.seq2seq.AttentionMechanism
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce seq2seq (import-module "tensorflow.contrib.seq2seq"))

(defn AttentionMechanism 
  ""
  [  ]
  (py/call-attr seq2seq "AttentionMechanism"  ))

(defn alignments-size 
  ""
  [ self ]
    (py/call-attr self "alignments_size"))

(defn state-size 
  ""
  [ self ]
    (py/call-attr self "state_size"))
