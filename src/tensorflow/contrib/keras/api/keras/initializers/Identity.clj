(ns tensorflow.contrib.keras.api.keras.initializers.Identity
  "Initializer that generates the identity matrix.

  Only use for 2D matrices.

  Args:
    gain: Multiplicative factor to apply to the identity matrix.
    dtype: Default data type, used if no `dtype` argument is provided when
      calling the initializer. Only floating point types are supported.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce initializers (import-module "tensorflow.contrib.keras.api.keras.initializers"))

(defn Identity 
  "Initializer that generates the identity matrix.

  Only use for 2D matrices.

  Args:
    gain: Multiplicative factor to apply to the identity matrix.
    dtype: Default data type, used if no `dtype` argument is provided when
      calling the initializer. Only floating point types are supported.
  "
  [ & {:keys [gain dtype]} ]
   (py/call-attr-kw initializers "Identity" [] {:gain gain :dtype dtype }))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))
