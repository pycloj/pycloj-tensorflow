(ns tensorflow.contrib.graph-editor.transform.StringIO
  "Text I/O implementation using an in-memory buffer.

The initial_value argument sets the value of object.  The newline
argument is like the one of TextIOWrapper's constructor."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce transform (import-module "tensorflow.contrib.graph_editor.transform"))

(defn StringIO 
  "Text I/O implementation using an in-memory buffer.

The initial_value argument sets the value of object.  The newline
argument is like the one of TextIOWrapper's constructor."
  [ & {:keys [initial_value newline]} ]
   (py/call-attr-kw transform "StringIO" [] {:initial_value initial_value :newline newline }))
