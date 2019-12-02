(ns tensorflow.-api.v1.nn.rnn-cell.LSTMStateTuple
  "Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.

  Stores two elements: `(c, h)`, in that order. Where `c` is the hidden state
  and `h` is the output.

  Only used when `state_is_tuple=True`.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce rnn-cell (import-module "tensorflow._api.v1.nn.rnn_cell"))

(defn LSTMStateTuple 
  "Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.

  Stores two elements: `(c, h)`, in that order. Where `c` is the hidden state
  and `h` is the output.

  Only used when `state_is_tuple=True`.
  "
  [ c h ]
  (py/call-attr rnn-cell "LSTMStateTuple"  c h ))

(defn c 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "c"))

(defn dtype 
  ""
  [ self ]
    (py/call-attr self "dtype"))

(defn h 
  "Alias for field number 1"
  [ self ]
    (py/call-attr self "h"))
