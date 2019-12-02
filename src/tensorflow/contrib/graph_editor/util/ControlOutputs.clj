(ns tensorflow.contrib.graph-editor.util.ControlOutputs
  "The control outputs topology."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce util (import-module "tensorflow.contrib.graph_editor.util"))

(defn ControlOutputs 
  "The control outputs topology."
  [ graph ]
  (py/call-attr util "ControlOutputs"  graph ))

(defn get 
  "return the control outputs of op."
  [ self op ]
  (py/call-attr self "get"  self op ))

(defn get-all 
  ""
  [ self  ]
  (py/call-attr self "get_all"  self  ))

(defn graph 
  ""
  [ self ]
    (py/call-attr self "graph"))

(defn update 
  "Update the control outputs if the graph has changed."
  [ self  ]
  (py/call-attr self "update"  self  ))
