(ns tensorflow.contrib.seq2seq.BeamSearchDecoderOutput
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

(defn BeamSearchDecoderOutput 
  ""
  [ scores predicted_ids parent_ids ]
  (py/call-attr seq2seq "BeamSearchDecoderOutput"  scores predicted_ids parent_ids ))

(defn parent-ids 
  "Alias for field number 2"
  [ self ]
    (py/call-attr self "parent_ids"))

(defn predicted-ids 
  "Alias for field number 1"
  [ self ]
    (py/call-attr self "predicted_ids"))

(defn scores 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "scores"))
