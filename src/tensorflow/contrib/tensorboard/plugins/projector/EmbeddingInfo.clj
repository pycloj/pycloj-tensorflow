(ns tensorflow.contrib.tensorboard.plugins.projector.EmbeddingInfo
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce projector (import-module "tensorflow.contrib.tensorboard.plugins.projector"))
