(ns tensorflow.-api.v1.compat.v1.lite.TocoConverter
  "Convert a TensorFlow model into `output_format` using TOCO.

  This class has been deprecated. Please use `lite.TFLiteConverter` instead.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce lite (import-module "tensorflow._api.v1.compat.v1.lite"))

(defn TocoConverter 
  "Convert a TensorFlow model into `output_format` using TOCO.

  This class has been deprecated. Please use `lite.TFLiteConverter` instead.
  "
  [  ]
  (py/call-attr lite "TocoConverter"  ))
