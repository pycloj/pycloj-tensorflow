(ns tensorflow.-api.v1.compat.v1.lite.experimental
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
(defonce experimental (import-module "tensorflow._api.v1.compat.v1.lite.experimental"))
(defn convert-op-hints-to-stubs 
  "Converts a graphdef with LiteOp hints into stub operations.

  This is used to prepare for toco conversion of complex intrinsic usages.
  Note: only one of session or graph_def should be used, not both.

  Args:
    session: A TensorFlow session that contains the graph to convert.
    graph_def: A graph def that we should convert.
    write_callback: A function pointer that can be used to write intermediate
      steps of graph transformation (optional).
  Returns:
    A new graphdef with all ops contained in OpHints being replaced by
    a single op call with the right parameters.
  Raises:
    ValueError: If both session and graph_def are provided.
  "
  [session graph_def  & {:keys [write_callback]} ]
    (py/call-attr-kw experimental "convert_op_hints_to_stubs" [session graph_def] {:write_callback write_callback }))

(defn get-potentially-supported-ops 
  "Returns operations potentially supported by TensorFlow Lite.

  The potentially support list contains a list of ops that are partially or
  fully supported, which is derived by simply scanning op names to check whether
  they can be handled without real conversion and specific parameters.

  Given that some ops may be partially supported, the optimal way to determine
  if a model's operations are supported is by converting using the TensorFlow
  Lite converter.

  Returns:
    A list of SupportedOp.
  "
  [  ]
  (py/call-attr experimental "get_potentially_supported_ops"  ))

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
