(ns tensorflow.contrib.tpu.TPUDistributionStrategy
  "The strategy to run Keras model on TPU."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce tpu (import-module "tensorflow.contrib.tpu"))
(defn TPUDistributionStrategy 
  "The strategy to run Keras model on TPU."
  [tpu_cluster_resolver  & {:keys [using_single_core]} ]
    (py/call-attr-kw tpu "TPUDistributionStrategy" [tpu_cluster_resolver] {:using_single_core using_single_core }))
