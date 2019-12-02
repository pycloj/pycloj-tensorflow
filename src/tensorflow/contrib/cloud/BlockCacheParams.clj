(ns tensorflow.contrib.cloud.BlockCacheParams
  "BlockCacheParams is a struct used for configuring the GCS Block Cache."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce cloud (import-module "tensorflow.contrib.cloud"))

(defn BlockCacheParams 
  "BlockCacheParams is a struct used for configuring the GCS Block Cache."
  [ block_size max_bytes max_staleness ]
  (py/call-attr cloud "BlockCacheParams"  block_size max_bytes max_staleness ))

(defn block-size 
  ""
  [ self ]
    (py/call-attr self "block_size"))

(defn max-bytes 
  ""
  [ self ]
    (py/call-attr self "max_bytes"))

(defn max-staleness 
  ""
  [ self ]
    (py/call-attr self "max_staleness"))
