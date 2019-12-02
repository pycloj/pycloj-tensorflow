(ns tensorflow.train.WorkerSessionCreator
  "Creates a tf.compat.v1.Session for a worker."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce train (import-module "tensorflow.train"))

(defn WorkerSessionCreator 
  "Creates a tf.compat.v1.Session for a worker."
  [scaffold & {:keys [master config max_wait_secs]
                       :or {config None}} ]
    (py/call-attr-kw train "WorkerSessionCreator" [scaffold] {:master master :config config :max_wait_secs max_wait_secs }))

(defn create-session 
  ""
  [ self  ]
  (py/call-attr self "create_session"  self  ))
