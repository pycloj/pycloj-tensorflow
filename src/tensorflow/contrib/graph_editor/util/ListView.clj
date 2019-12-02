(ns tensorflow.contrib.graph-editor.util.ListView
  "Immutable list wrapper.

  This class is strongly inspired by the one in tf.Operation.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce util (import-module "tensorflow.contrib.graph_editor.util"))

(defn ListView 
  "Immutable list wrapper.

  This class is strongly inspired by the one in tf.Operation.
  "
  [ list_ ]
  (py/call-attr util "ListView"  list_ ))
