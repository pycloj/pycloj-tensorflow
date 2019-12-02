(ns tensorflow.-api.v1.data.experimental.MapVectorizationOptions
  "Represents options for the MapVectorization optimization."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce experimental (import-module "tensorflow._api.v1.data.experimental"))

(defn MapVectorizationOptions 
  "Represents options for the MapVectorization optimization."
  [  ]
  (py/call-attr experimental "MapVectorizationOptions"  ))

(defn enabled 
  "Whether to vectorize map transformations. If None, defaults to False."
  [ self ]
    (py/call-attr self "enabled"))

(defn use-choose-fastest 
  "Whether to use ChooseFastestBranchDataset with this transformation. If True, the pipeline picks between the vectorized and original segment at runtime based on their iterations speed. If None, defaults to False."
  [ self ]
    (py/call-attr self "use_choose_fastest"))
