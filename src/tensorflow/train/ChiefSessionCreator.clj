(ns tensorflow.train.ChiefSessionCreator
  "Creates a tf.compat.v1.Session for a chief."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce train (import-module "tensorflow.train"))

(defn ChiefSessionCreator 
  "Creates a tf.compat.v1.Session for a chief."
  [scaffold & {:keys [master config checkpoint_dir checkpoint_filename_with_path]
                       :or {config None checkpoint_dir None checkpoint_filename_with_path None}} ]
    (py/call-attr-kw train "ChiefSessionCreator" [scaffold] {:master master :config config :checkpoint_dir checkpoint_dir :checkpoint_filename_with_path checkpoint_filename_with_path }))

(defn create-session 
  ""
  [ self  ]
  (py/call-attr self "create_session"  self  ))
