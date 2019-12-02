(ns tensorflow.train.SessionCreator
  "A factory for tf.Session."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce train (import-module "tensorflow.train"))

(defn SessionCreator 
  "A factory for tf.Session."
  [  ]
  (py/call-attr train "SessionCreator"  ))

(defn create-session 
  ""
  [ self  ]
  (py/call-attr self "create_session"  self  ))
