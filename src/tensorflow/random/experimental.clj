(ns tensorflow.-api.v1.random.experimental
  "Public API for tf.random.experimental namespace.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce experimental (import-module "tensorflow._api.v1.random.experimental"))

(defn create-rng-state 
  "Creates a RNG state.

  Args:
    seed: an integer or 1-D tensor.
    algorithm: an integer representing the RNG algorithm.

  Returns:
    a 1-D tensor whose size depends on the algorithm.
  "
  [ seed algorithm ]
  (py/call-attr experimental "create_rng_state"  seed algorithm ))

(defn get-global-generator 
  ""
  [  ]
  (py/call-attr experimental "get_global_generator"  ))

(defn set-global-generator 
  "Replaces the global generator with another `Generator` object.

  This function creates a new Generator object (and the Variable object within),
  which does not work well with tf.function because (1) tf.function puts
  restrictions on Variable creation thus reset_global_generator can't be freely
  used inside tf.function; (2) redirecting a global variable to
  a new object is problematic with tf.function because the old object may be
  captured by a 'tf.function'ed function and still be used by it.
  A 'tf.function'ed function only keeps weak references to variables,
  so deleting a variable and then calling that function again may raise an
  error, as demonstrated by
  random_test.py/RandomTest.testResetGlobalGeneratorBadWithDefun .

  Args:
    generator: the new `Generator` object.
  "
  [ generator ]
  (py/call-attr experimental "set_global_generator"  generator ))
