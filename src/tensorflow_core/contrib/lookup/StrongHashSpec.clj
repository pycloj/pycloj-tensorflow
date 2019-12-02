(ns tensorflow-core.contrib.lookup.StrongHashSpec
  "A structure to specify a key of the strong keyed hash spec.

  The strong hash requires a `key`, which is a list of 2 unsigned integer
  numbers. These should be non-zero; random numbers generated from random.org
  would be a fine choice.

  Fields:
    key: The key to be used by the keyed hashing function.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce lookup (import-module "tensorflow_core.contrib.lookup"))

(defn StrongHashSpec 
  "A structure to specify a key of the strong keyed hash spec.

  The strong hash requires a `key`, which is a list of 2 unsigned integer
  numbers. These should be non-zero; random numbers generated from random.org
  would be a fine choice.

  Fields:
    key: The key to be used by the keyed hashing function.
  "
  [ key ]
  (py/call-attr lookup "StrongHashSpec"  key ))

(defn hasher 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "hasher"))

(defn key 
  "Alias for field number 1"
  [ self ]
    (py/call-attr self "key"))
