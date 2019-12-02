(ns tensorflow.-api.v1.compat.v1.lite.TargetSpec
  "Specification of target device.

  Details about target device. Converter optimizes the generated model for
  specific device.

  Attributes:
    supported_ops: Experimental flag, subject to change. Set of OpsSet options
      supported by the device. (default set([OpsSet.TFLITE_BUILTINS]))
    supported_types: List of types for constant values on the target device.
      Supported values are types exported by lite.constants. Frequently, an
      optimization choice is driven by the most compact (i.e. smallest)
      type in this list (default [constants.FLOAT])
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

(defn TargetSpec 
  "Specification of target device.

  Details about target device. Converter optimizes the generated model for
  specific device.

  Attributes:
    supported_ops: Experimental flag, subject to change. Set of OpsSet options
      supported by the device. (default set([OpsSet.TFLITE_BUILTINS]))
    supported_types: List of types for constant values on the target device.
      Supported values are types exported by lite.constants. Frequently, an
      optimization choice is driven by the most compact (i.e. smallest)
      type in this list (default [constants.FLOAT])
  "
  [ supported_ops supported_types ]
  (py/call-attr lite "TargetSpec"  supported_ops supported_types ))
