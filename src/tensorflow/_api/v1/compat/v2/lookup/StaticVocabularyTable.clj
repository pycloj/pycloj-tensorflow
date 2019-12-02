(ns tensorflow.-api.v1.compat.v2.lookup.StaticVocabularyTable
  "String to Id table wrapper that assigns out-of-vocabulary keys to buckets.

  For example, if an instance of `StaticVocabularyTable` is initialized with a
  string-to-id initializer that maps:

  * `emerson -> 0`
  * `lake -> 1`
  * `palmer -> 2`

  The `Vocabulary` object will performs the following mapping:

  * `emerson -> 0`
  * `lake -> 1`
  * `palmer -> 2`
  * `<other term> -> bucket_id`, where bucket_id will be between `3` and
  `3 + num_oov_buckets - 1`, calculated by:
  `hash(<term>) % num_oov_buckets + vocab_size`

  If input_tensor is `[\"emerson\", \"lake\", \"palmer\", \"king\", \"crimson\"]`,
  the lookup result is `[0, 1, 2, 4, 7]`.

  If `initializer` is None, only out-of-vocabulary buckets are used.

  Example usage:

  ```python
  num_oov_buckets = 3
  input_tensor = tf.constant([\"emerson\", \"lake\", \"palmer\", \"king\", \"crimnson\"])
  table = tf.lookup.StaticVocabularyTable(
      tf.TextFileIdTableInitializer(filename), num_oov_buckets)
  out = table.lookup(input_tensor).
  table.init.run()
  print(out.eval())
  ```

  The hash function used for generating out-of-vocabulary buckets ID is
  Fingerprint64.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce lookup (import-module "tensorflow._api.v1.compat.v2.lookup"))

(defn StaticVocabularyTable 
  "String to Id table wrapper that assigns out-of-vocabulary keys to buckets.

  For example, if an instance of `StaticVocabularyTable` is initialized with a
  string-to-id initializer that maps:

  * `emerson -> 0`
  * `lake -> 1`
  * `palmer -> 2`

  The `Vocabulary` object will performs the following mapping:

  * `emerson -> 0`
  * `lake -> 1`
  * `palmer -> 2`
  * `<other term> -> bucket_id`, where bucket_id will be between `3` and
  `3 + num_oov_buckets - 1`, calculated by:
  `hash(<term>) % num_oov_buckets + vocab_size`

  If input_tensor is `[\"emerson\", \"lake\", \"palmer\", \"king\", \"crimson\"]`,
  the lookup result is `[0, 1, 2, 4, 7]`.

  If `initializer` is None, only out-of-vocabulary buckets are used.

  Example usage:

  ```python
  num_oov_buckets = 3
  input_tensor = tf.constant([\"emerson\", \"lake\", \"palmer\", \"king\", \"crimnson\"])
  table = tf.lookup.StaticVocabularyTable(
      tf.TextFileIdTableInitializer(filename), num_oov_buckets)
  out = table.lookup(input_tensor).
  table.init.run()
  print(out.eval())
  ```

  The hash function used for generating out-of-vocabulary buckets ID is
  Fingerprint64.
  "
  [ initializer num_oov_buckets lookup_key_dtype name ]
  (py/call-attr lookup "StaticVocabularyTable"  initializer num_oov_buckets lookup_key_dtype name ))

(defn key-dtype 
  "The table key dtype."
  [ self ]
    (py/call-attr self "key_dtype"))

(defn lookup 
  "Looks up `keys` in the table, outputs the corresponding values.

    It assigns out-of-vocabulary keys to buckets based in their hashes.

    Args:
      keys: Keys to look up. May be either a `SparseTensor` or dense `Tensor`.
      name: Optional name for the op.

    Returns:
      A `SparseTensor` if keys are sparse, otherwise a dense `Tensor`.

    Raises:
      TypeError: when `keys` doesn't match the table key data type.
    "
  [ self keys name ]
  (py/call-attr self "lookup"  self keys name ))

(defn name 
  ""
  [ self ]
    (py/call-attr self "name"))

(defn resource-handle 
  ""
  [ self ]
    (py/call-attr self "resource_handle"))

(defn size 
  "Compute the number of elements in this table."
  [ self name ]
  (py/call-attr self "size"  self name ))

(defn value-dtype 
  "The table value dtype."
  [ self ]
    (py/call-attr self "value_dtype"))
