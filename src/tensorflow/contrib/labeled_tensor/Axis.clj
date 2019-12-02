(ns tensorflow.contrib.labeled-tensor.Axis
  "Size and label information for an axis.

  Axis contains either a tf.compat.v1.Dimension indicating the size of an axis,
  or a tuple of tick labels for the axis.

  If tick labels are provided, they must be unique.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce labeled-tensor (import-module "tensorflow.contrib.labeled_tensor"))

(defn Axis 
  "Size and label information for an axis.

  Axis contains either a tf.compat.v1.Dimension indicating the size of an axis,
  or a tuple of tick labels for the axis.

  If tick labels are provided, they must be unique.
  "
  [ name value ]
  (py/call-attr labeled-tensor "Axis"  name value ))

(defn dimension 
  ""
  [ self ]
    (py/call-attr self "dimension"))

(defn index 
  "Returns the integer position of the given tick label."
  [ self value ]
  (py/call-attr self "index"  self value ))

(defn labels 
  "Returns the tuple containing coordinate labels, else None."
  [ self ]
    (py/call-attr self "labels"))

(defn name 
  ""
  [ self ]
    (py/call-attr self "name"))

(defn size 
  ""
  [ self ]
    (py/call-attr self "size"))

(defn value 
  "Returns the tf.compat.v1.Dimension or tuple specifying axis ticks."
  [ self ]
    (py/call-attr self "value"))
