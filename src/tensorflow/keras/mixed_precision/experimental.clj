(ns tensorflow.python.keras.api.-v1.keras.mixed-precision.experimental
  "Mixed precision API.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce experimental (import-module "tensorflow.python.keras.api._v1.keras.mixed_precision.experimental"))

(defn global-policy 
  "Returns the global Policy.

  The global policy is the default policy used for layers, if no policy is
  passed to the layer constructor. If no policy has been set with
  `keras.mixed_precision.experimental.set_policy`, this will return a policy
  constructed from `tf.keras.backend.floatx()` in TensorFlow 2, or an \"infer\"
  policy in TensorFlow 1.

  See `keras.mixed_precision.experimental.Policy` for more information.

  Returns:
    The global Policy.
  "
  [  ]
  (py/call-attr experimental "global_policy"  ))

(defn set-policy 
  "Sets the global Policy.

  The global policy is the default policy used for layers, if no policy is
  passed to the layer constructor. If no global policy is set, layers will
  instead default to a Policy constructed from `tf.keras.backend.floatx()` in
  TensorFlow 2. In TensorFlow 1, layers default to an \"infer\" policy.

  See `keras.mixed_precision.experimental.Policy` for more information.

  Args:
    policy: A Policy, or a string that will be converted to a Policy..
  "
  [ policy ]
  (py/call-attr experimental "set_policy"  policy ))
