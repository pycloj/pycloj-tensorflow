(ns tensorflow-core.contrib.autograph.AutoGraphError
  "Base class for all AutoGraph exceptions."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce autograph (import-module "tensorflow_core.contrib.autograph"))
