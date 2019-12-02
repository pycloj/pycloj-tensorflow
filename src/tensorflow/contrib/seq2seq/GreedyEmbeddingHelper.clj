(ns tensorflow.contrib.seq2seq.GreedyEmbeddingHelper
  "A helper for use during inference.

  Uses the argmax of the output (treated as logits) and passes the
  result through an embedding layer to get the next input.
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

(defn GreedyEmbeddingHelper 
  "A helper for use during inference.

  Uses the argmax of the output (treated as logits) and passes the
  result through an embedding layer to get the next input.
  "
  [ embedding start_tokens end_token ]
  (py/call-attr seq2seq "GreedyEmbeddingHelper"  embedding start_tokens end_token ))

(defn batch-size 
  ""
  [ self ]
    (py/call-attr self "batch_size"))

(defn initialize 
  ""
  [ self name ]
  (py/call-attr self "initialize"  self name ))

(defn next-inputs 
  "next_inputs_fn for GreedyEmbeddingHelper."
  [ self time outputs state sample_ids name ]
  (py/call-attr self "next_inputs"  self time outputs state sample_ids name ))

(defn sample 
  "sample for GreedyEmbeddingHelper."
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
