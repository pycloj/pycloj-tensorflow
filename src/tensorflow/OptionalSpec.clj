(ns tensorflow.OptionalSpec
  "Represents an optional potentially containing a structured value."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce tensorflow (import-module "tensorflow"))

(defn OptionalSpec 
  "Represents an optional potentially containing a structured value."
  [ value_structure ]
  (py/call-attr tensorflow "OptionalSpec"  value_structure ))

(defn from-value 
  ""
  [ self value ]
  (py/call-attr self "from_value"  self value ))

(defn is-compatible-with 
  "Returns true if `spec_or_value` is compatible with this TypeSpec."
  [ self spec_or_value ]
  (py/call-attr self "is_compatible_with"  self spec_or_value ))

(defn most-specific-compatible-type 
  "Returns the most specific TypeSpec compatible with `self` and `other`.

    Args:
      other: A `TypeSpec`.

    Raises:
      ValueError: If there is no TypeSpec that is compatible with both `self`
        and `other`.
    "
  [ self other ]
  (py/call-attr self "most_specific_compatible_type"  self other ))

(defn value-type 
  ""
  [ self ]
    (py/call-attr self "value_type"))
