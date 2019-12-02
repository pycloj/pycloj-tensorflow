(ns tensorflow.contrib.rnn.FusedRNNCellAdaptor
  "This is an adaptor for RNNCell classes to be used with `FusedRNNCell`."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce rnn (import-module "tensorflow.contrib.rnn"))
(defn FusedRNNCellAdaptor 
  "This is an adaptor for RNNCell classes to be used with `FusedRNNCell`."
  [cell  & {:keys [use_dynamic_rnn]} ]
    (py/call-attr-kw rnn "FusedRNNCellAdaptor" [cell] {:use_dynamic_rnn use_dynamic_rnn }))
