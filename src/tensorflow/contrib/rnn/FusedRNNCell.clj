(ns tensorflow.contrib.rnn.FusedRNNCell
  "Abstract object representing a fused RNN cell.

  A fused RNN cell represents the entire RNN expanded over the time
  dimension. In effect, this represents an entire recurrent network.

  Unlike RNN cells which are subclasses of `rnn_cell.RNNCell`, a `FusedRNNCell`
  operates on the entire time sequence at once, by putting the loop over time
  inside the cell. This usually leads to much more efficient, but more complex
  and less flexible implementations.

  Every `FusedRNNCell` must implement `__call__` with the following signature.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce rnn (import-module "tensorflow.contrib.rnn"))

(defn FusedRNNCell 
  "Abstract object representing a fused RNN cell.

  A fused RNN cell represents the entire RNN expanded over the time
  dimension. In effect, this represents an entire recurrent network.

  Unlike RNN cells which are subclasses of `rnn_cell.RNNCell`, a `FusedRNNCell`
  operates on the entire time sequence at once, by putting the loop over time
  inside the cell. This usually leads to much more efficient, but more complex
  and less flexible implementations.

  Every `FusedRNNCell` must implement `__call__` with the following signature.
  "
  [  ]
  (py/call-attr rnn "FusedRNNCell"  ))
