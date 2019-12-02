(ns tensorflow.-api.v1.compat.v1.train.CheckpointSaverHook
  "Saves checkpoints every N steps or seconds."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce train (import-module "tensorflow._api.v1.compat.v1.train"))

(defn CheckpointSaverHook 
  "Saves checkpoints every N steps or seconds."
  [checkpoint_dir save_secs save_steps saver & {:keys [checkpoint_basename scaffold listeners]
                       :or {scaffold None listeners None}} ]
    (py/call-attr-kw train "CheckpointSaverHook" [checkpoint_dir save_secs save_steps saver] {:checkpoint_basename checkpoint_basename :scaffold scaffold :listeners listeners }))

(defn after-create-session 
  ""
  [ self session coord ]
  (py/call-attr self "after_create_session"  self session coord ))

(defn after-run 
  ""
  [ self run_context run_values ]
  (py/call-attr self "after_run"  self run_context run_values ))

(defn before-run 
  ""
  [ self run_context ]
  (py/call-attr self "before_run"  self run_context ))

(defn begin 
  ""
  [ self  ]
  (py/call-attr self "begin"  self  ))

(defn end 
  ""
  [ self session ]
  (py/call-attr self "end"  self session ))
