(ns tensorflow.contrib.seq2seq.BeamSearchDecoderState
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

(defn BeamSearchDecoderState 
  ""
  [ cell_state log_probs finished lengths accumulated_attention_probs ]
  (py/call-attr seq2seq "BeamSearchDecoderState"  cell_state log_probs finished lengths accumulated_attention_probs ))

(defn accumulated-attention-probs 
  "Alias for field number 4"
  [ self ]
    (py/call-attr self "accumulated_attention_probs"))

(defn cell-state 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "cell_state"))

(defn finished 
  "Alias for field number 2"
  [ self ]
    (py/call-attr self "finished"))

(defn lengths 
  "Alias for field number 3"
  [ self ]
    (py/call-attr self "lengths"))

(defn log-probs 
  "Alias for field number 1"
  [ self ]
    (py/call-attr self "log_probs"))
