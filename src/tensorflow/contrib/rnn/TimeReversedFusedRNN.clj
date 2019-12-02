(ns tensorflow.contrib.rnn.TimeReversedFusedRNN
  "This is an adaptor to time-reverse a FusedRNNCell.

  For example,

  ```python
  cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell(10)
  fw_lstm = tf.contrib.rnn.FusedRNNCellAdaptor(cell, use_dynamic_rnn=True)
  bw_lstm = tf.contrib.rnn.TimeReversedFusedRNN(fw_lstm)
  fw_out, fw_state = fw_lstm(inputs)
  bw_out, bw_state = bw_lstm(inputs)
  ```
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

(defn TimeReversedFusedRNN 
  "This is an adaptor to time-reverse a FusedRNNCell.

  For example,

  ```python
  cell = tf.compat.v1.nn.rnn_cell.BasicRNNCell(10)
  fw_lstm = tf.contrib.rnn.FusedRNNCellAdaptor(cell, use_dynamic_rnn=True)
  bw_lstm = tf.contrib.rnn.TimeReversedFusedRNN(fw_lstm)
  fw_out, fw_state = fw_lstm(inputs)
  bw_out, bw_state = bw_lstm(inputs)
  ```
  "
  [ cell ]
  (py/call-attr rnn "TimeReversedFusedRNN"  cell ))
