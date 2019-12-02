(ns tensorflow.contrib.seq2seq.BasicDecoderOutput
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

(defn BasicDecoderOutput 
  ""
  [ rnn_output sample_id ]
  (py/call-attr seq2seq "BasicDecoderOutput"  rnn_output sample_id ))

(defn rnn-output 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "rnn_output"))

(defn sample-id 
  "Alias for field number 1"
  [ self ]
    (py/call-attr self "sample_id"))
