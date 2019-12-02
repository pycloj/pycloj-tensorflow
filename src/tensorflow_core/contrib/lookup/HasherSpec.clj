(ns tensorflow-core.contrib.lookup.HasherSpec
  "A structure for the spec of the hashing function to use for hash buckets.

  `hasher` is the name of the hashing function to use (eg. \"fasthash\",
  \"stronghash\").
  `key` is optional and specify the key to use for the hash function if
  supported, currently only used by a strong hash.

  Fields:
    hasher: The hasher name to use.
    key: The key to be used by the hashing function, if required.
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

(defn HasherSpec 
  "A structure for the spec of the hashing function to use for hash buckets.

  `hasher` is the name of the hashing function to use (eg. \"fasthash\",
  \"stronghash\").
  `key` is optional and specify the key to use for the hash function if
  supported, currently only used by a strong hash.

  Fields:
    hasher: The hasher name to use.
    key: The key to be used by the hashing function, if required.
  "
  [ hasher key ]
  (py/call-attr lookup "HasherSpec"  hasher key ))

(defn hasher 
  "Alias for field number 0"
  [ self ]
    (py/call-attr self "hasher"))

(defn key 
  "Alias for field number 1"
  [ self ]
    (py/call-attr self "key"))
