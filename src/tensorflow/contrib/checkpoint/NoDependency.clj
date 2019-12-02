(ns tensorflow.contrib.checkpoint.NoDependency
  "Allows attribute assignment to `Trackable` objects with no dependency.

  Example usage:
  ```python
  obj = Trackable()
  obj.has_dependency = tf.Variable(0., name=\"dep\")
  obj.no_dependency = NoDependency(tf.Variable(1., name=\"nodep\"))
  assert obj.no_dependency.name == \"nodep:0\"
  ```

  `obj` in this example has a dependency on the variable \"dep\", and both
  attributes contain un-wrapped `Variable` objects.

  `NoDependency` also works with `tf.keras.Model`, but only for checkpoint
  dependencies: wrapping a `Layer` in `NoDependency` will assign the (unwrapped)
  `Layer` to the attribute without a checkpoint dependency, but the `Model` will
  still track the `Layer` (so it will appear in `Model.layers`, and its
  variables will appear in `Model.variables`).
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce checkpoint (import-module "tensorflow.contrib.checkpoint"))

(defn NoDependency 
  "Allows attribute assignment to `Trackable` objects with no dependency.

  Example usage:
  ```python
  obj = Trackable()
  obj.has_dependency = tf.Variable(0., name=\"dep\")
  obj.no_dependency = NoDependency(tf.Variable(1., name=\"nodep\"))
  assert obj.no_dependency.name == \"nodep:0\"
  ```

  `obj` in this example has a dependency on the variable \"dep\", and both
  attributes contain un-wrapped `Variable` objects.

  `NoDependency` also works with `tf.keras.Model`, but only for checkpoint
  dependencies: wrapping a `Layer` in `NoDependency` will assign the (unwrapped)
  `Layer` to the attribute without a checkpoint dependency, but the `Model` will
  still track the `Layer` (so it will appear in `Model.layers`, and its
  variables will appear in `Model.variables`).
  "
  [ value ]
  (py/call-attr checkpoint "NoDependency"  value ))
