(ns tensorflow.-api.v1.compat.v2.lite.experimental
  "Public API for tf.lite.experimental namespace.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce experimental (import-module "tensorflow._api.v1.compat.v2.lite.experimental"))

(defn load-delegate 
  "Returns loaded Delegate object.

  Args:
    library: Name of shared library containing the
      [TfLiteDelegate](https://www.tensorflow.org/lite/performance/delegates).
    options: Dictionary of options that are required to load the delegate. All
      keys and values in the dictionary should be convertible to str. Consult
      the documentation of the specific delegate for required and legal options.
      (default None)

  Returns:
    Delegate object.

  Raises:
    ValueError: Delegate failed to load.
    RuntimeError: If delegate loading is used on unsupported platform.
  "
  [ library options ]
  (py/call-attr experimental "load_delegate"  library options ))
