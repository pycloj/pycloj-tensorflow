(ns tensorflow.contrib.summary.summary.SummaryWriter
  "Interface representing a stateful summary writer object."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce summary (import-module "tensorflow.contrib.summary.summary"))

(defn SummaryWriter 
  "Interface representing a stateful summary writer object."
  [  ]
  (py/call-attr summary "SummaryWriter"  ))

(defn as-default 
  "Returns a context manager that enables summary writing."
  [ self  ]
  (py/call-attr self "as_default"  self  ))

(defn close 
  "Flushes and closes the summary writer."
  [ self  ]
  (py/call-attr self "close"  self  ))

(defn flush 
  "Flushes any buffered data."
  [ self  ]
  (py/call-attr self "flush"  self  ))

(defn init 
  "Initializes the summary writer."
  [ self  ]
  (py/call-attr self "init"  self  ))

(defn set-as-default 
  "Enables this summary writer for the current thread."
  [ self  ]
  (py/call-attr self "set_as_default"  self  ))
