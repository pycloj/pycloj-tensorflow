(ns tensorflow.-api.v1.compat.v2.lite.TFLiteConverter
  "Converts a TensorFlow model into TensorFlow Lite model.

  Attributes:
    allow_custom_ops: Boolean indicating whether to allow custom operations.
      When false any unknown operation is an error. When true, custom ops are
      created for any op that is unknown. The developer will need to provide
      these to the TensorFlow Lite runtime with a custom resolver.
      (default False)
    target_spec: Experimental flag, subject to change. Specification of target
      device.
    optimizations: Experimental flag, subject to change. A list of optimizations
      to apply when converting the model. E.g. `[Optimize.DEFAULT]
    representative_dataset: A representative dataset that can be used to
      generate input and output samples for the model. The converter can use the
      dataset to evaluate different optimizations.
    experimental_enable_mlir_converter: Experimental flag, subject to change.
      Enables the MLIR converter instead of the TOCO converter.

  Example usage:

    ```python
    # Converting a SavedModel to a TensorFlow Lite model.
    converter = lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()

    # Converting a tf.Keras model to a TensorFlow Lite model.
    converter = lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Converting ConcreteFunctions to a TensorFlow Lite model.
    converter = lite.TFLiteConverter.from_concrete_functions([func])
    tflite_model = converter.convert()
    ```
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce lite (import-module "tensorflow._api.v1.compat.v2.lite"))

(defn TFLiteConverter 
  "Converts a TensorFlow model into TensorFlow Lite model.

  Attributes:
    allow_custom_ops: Boolean indicating whether to allow custom operations.
      When false any unknown operation is an error. When true, custom ops are
      created for any op that is unknown. The developer will need to provide
      these to the TensorFlow Lite runtime with a custom resolver.
      (default False)
    target_spec: Experimental flag, subject to change. Specification of target
      device.
    optimizations: Experimental flag, subject to change. A list of optimizations
      to apply when converting the model. E.g. `[Optimize.DEFAULT]
    representative_dataset: A representative dataset that can be used to
      generate input and output samples for the model. The converter can use the
      dataset to evaluate different optimizations.
    experimental_enable_mlir_converter: Experimental flag, subject to change.
      Enables the MLIR converter instead of the TOCO converter.

  Example usage:

    ```python
    # Converting a SavedModel to a TensorFlow Lite model.
    converter = lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()

    # Converting a tf.Keras model to a TensorFlow Lite model.
    converter = lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Converting ConcreteFunctions to a TensorFlow Lite model.
    converter = lite.TFLiteConverter.from_concrete_functions([func])
    tflite_model = converter.convert()
    ```
  "
  [ funcs trackable_obj ]
  (py/call-attr lite "TFLiteConverter"  funcs trackable_obj ))

(defn convert 
  "Converts a TensorFlow GraphDef based on instance variables.

    Returns:
      The converted data in serialized format.

    Raises:
      ValueError:
        Multiple concrete functions are specified.
        Input shape is not specified.
        Invalid quantization parameters.
    "
  [ self  ]
  (py/call-attr self "convert"  self  ))
