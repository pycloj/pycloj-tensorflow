(ns tensorflow.contrib.graph-editor.transform.Transformer
  "Transform a subgraph into another one.

  By default, the constructor create a transform which copy a subgraph and
  replaces inputs with placeholders. This behavior can be modified by changing
  the handlers.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce transform (import-module "tensorflow.contrib.graph_editor.transform"))

(defn Transformer 
  "Transform a subgraph into another one.

  By default, the constructor create a transform which copy a subgraph and
  replaces inputs with placeholders. This behavior can be modified by changing
  the handlers.
  "
  [  ]
  (py/call-attr transform "Transformer"  ))
