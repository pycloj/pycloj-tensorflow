(ns tensorflow.contrib.tensorboard.plugins.projector.projector-config-pb2.EmbeddingInfo
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce projector-config-pb2 (import-module "tensorflow.contrib.tensorboard.plugins.projector.projector_config_pb2"))
