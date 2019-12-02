(ns tensorflow.contrib.tpu.InputPipelineConfig
  "Please see the definition of these values in TPUConfig."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce tpu (import-module "tensorflow.contrib.tpu"))

(defn InputPipelineConfig 
  "Please see the definition of these values in TPUConfig."
  [  ]
  (py/call-attr tpu "InputPipelineConfig"  ))
