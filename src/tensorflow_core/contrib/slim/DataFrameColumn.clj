(ns tensorflow-core.contrib.slim.DataFrameColumn
  "Represents a feature column produced from a `DataFrame`.

  Instances of this class are immutable.  A `DataFrame` column may be dense or
  sparse, and may have any shape, with the constraint that dimension 0 is
  batch_size.

  Args:
    column_name: a name for this column
    series: a `Series` to be wrapped, which has already had its base features
      substituted with `PredefinedSeries`.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce slim (import-module "tensorflow_core.contrib.slim"))

(defn DataFrameColumn 
  "Represents a feature column produced from a `DataFrame`.

  Instances of this class are immutable.  A `DataFrame` column may be dense or
  sparse, and may have any shape, with the constraint that dimension 0 is
  batch_size.

  Args:
    column_name: a name for this column
    series: a `Series` to be wrapped, which has already had its base features
      substituted with `PredefinedSeries`.
  "
  [ column_name series ]
  (py/call-attr slim "DataFrameColumn"  column_name series ))

(defn column-name 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "column_name"))

(defn config 
  ""
  [ self ]
    (py/call-attr self "config"))

(defn insert-transformed-feature 
  ""
  [ self columns_to_tensors ]
  (py/call-attr self "insert_transformed_feature"  self columns_to_tensors ))

(defn key 
  "Returns a string which will be used as a key when we do sorting."
  [ self ]
    (py/call-attr self "key"))

(defn name 
  ""
  [ self ]
    (py/call-attr self "name"))

(defn series 
  "Alias for field number 1"
  [ self ]
    (py/call-attr self "series"))
