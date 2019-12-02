(ns tensorflow.-api.v1.compat.v1.lite.TFLiteConverter
  "Convert a TensorFlow model into `output_format`.

  This is used to convert from a TensorFlow GraphDef, SavedModel or tf.keras
  model into either a TFLite FlatBuffer or graph visualization.

  Attributes:
    inference_type: Target data type of real-number arrays in the output file.
      Must be `{tf.float32, tf.uint8}`. If `optimzations` are provided, this
      parameter is ignored. (default tf.float32)
    inference_input_type: Target data type of real-number input arrays. Allows
      for a different type for input arrays.
      If an integer type is provided and `optimizations` are not used,
      `quantized_inputs_stats` must be provided.
      If `inference_type` is tf.uint8, signaling conversion to a fully quantized
      model from a quantization-aware trained input model, then
      `inference_input_type` defaults to tf.uint8.
      In all other cases, `inference_input_type` defaults to tf.float32.
      Must be `{tf.float32, tf.uint8, tf.int8}`
    inference_output_type: Target data type of real-number output arrays. Allows
      for a different type for output arrays.
      If `inference_type` is tf.uint8, signaling conversion to a fully quantized
      model from a quantization-aware trained output model, then
      `inference_output_type` defaults to tf.uint8.
      In all other cases, `inference_output_type` must be tf.float32, an error
      will be thrown otherwise.
      Must be `{tf.float32, tf.uint8, tf.int8}`
    output_format: Output file format. Currently must be `{TFLITE,
      GRAPHVIZ_DOT}`. (default TFLITE)
    quantized_input_stats: Dict of strings representing input tensor names
      mapped to tuple of floats representing the mean and standard deviation
      of the training data (e.g., {\"foo\" : (0., 1.)}). Only need if
      `inference_input_type` is `QUANTIZED_UINT8`.
      real_input_value = (quantized_input_value - mean_value) / std_dev_value.
      (default {})
    default_ranges_stats: Tuple of integers representing (min, max) range values
      for all arrays without a specified range. Intended for experimenting with
      quantization via \"dummy quantization\". (default None)
    drop_control_dependency: Boolean indicating whether to drop control
      dependencies silently. This is due to TFLite not supporting control
      dependencies. (default True)
    reorder_across_fake_quant: Boolean indicating whether to reorder FakeQuant
      nodes in unexpected locations. Used when the location of the FakeQuant
      nodes is preventing graph transformations necessary to convert the graph.
      Results in a graph that differs from the quantized training graph,
      potentially causing differing arithmetic behavior. (default False)
    change_concat_input_ranges: Boolean to change behavior of min/max ranges for
      inputs and outputs of the concat operator for quantized models. Changes
      the ranges of concat operator overlap when true. (default False)
    allow_custom_ops: Boolean indicating whether to allow custom operations.
      When false any unknown operation is an error. When true, custom ops are
      created for any op that is unknown. The developer will need to provide
      these to the TensorFlow Lite runtime with a custom resolver.
      (default False)
    post_training_quantize: Deprecated. Please specify `[Optimize.DEFAULT]` for
      `optimizations` instead. Boolean indicating whether to quantize the
      weights of the converted float model.  Model size will be reduced and
      there will be latency improvements (at the cost of accuracy).
      (default False)
    dump_graphviz_dir: Full filepath of folder to dump the graphs at various
      stages of processing GraphViz .dot files. Preferred over
      --output_format=GRAPHVIZ_DOT in order to keep the requirements of the
      output file. (default None)
    dump_graphviz_video: Boolean indicating whether to dump the graph after
      every graph transformation. (default False)
    target_ops: Deprecated. Please specify `target_spec.supported_ops` instead.
      Set of OpsSet options indicating which converter to use.
      (default set([OpsSet.TFLITE_BUILTINS]))
    target_spec: Experimental flag, subject to change. Specification of target
      device.
    optimizations: Experimental flag, subject to change. A list of optimizations
      to apply when converting the model. E.g. `[Optimize.DEFAULT]`
    representative_dataset: A representative dataset that can be used to
      generate input and output samples for the model. The converter can use
      the dataset to evaluate different optimizations.
    experimental_enable_mlir_converter: Experimental flag, subject to change.
      Enables the MLIR converter instead of the TOCO converter.

  Example usage:

    ```python
    # Converting a GraphDef from session.
    converter = lite.TFLiteConverter.from_session(sess, in_tensors, out_tensors)
    tflite_model = converter.convert()
    open(\"converted_model.tflite\", \"wb\").write(tflite_model)

    # Converting a GraphDef from file.
    converter = lite.TFLiteConverter.from_frozen_graph(
      graph_def_file, input_arrays, output_arrays)
    tflite_model = converter.convert()
    open(\"converted_model.tflite\", \"wb\").write(tflite_model)

    # Converting a SavedModel.
    converter = lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    open(\"converted_model.tflite\", \"wb\").write(tflite_model)

    # Converting a tf.keras model.
    converter = lite.TFLiteConverter.from_keras_model_file(keras_model)
    tflite_model = converter.convert()
    open(\"converted_model.tflite\", \"wb\").write(tflite_model)
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
(defonce lite (import-module "tensorflow._api.v1.compat.v1.lite"))

(defn TFLiteConverter 
  "Convert a TensorFlow model into `output_format`.

  This is used to convert from a TensorFlow GraphDef, SavedModel or tf.keras
  model into either a TFLite FlatBuffer or graph visualization.

  Attributes:
    inference_type: Target data type of real-number arrays in the output file.
      Must be `{tf.float32, tf.uint8}`. If `optimzations` are provided, this
      parameter is ignored. (default tf.float32)
    inference_input_type: Target data type of real-number input arrays. Allows
      for a different type for input arrays.
      If an integer type is provided and `optimizations` are not used,
      `quantized_inputs_stats` must be provided.
      If `inference_type` is tf.uint8, signaling conversion to a fully quantized
      model from a quantization-aware trained input model, then
      `inference_input_type` defaults to tf.uint8.
      In all other cases, `inference_input_type` defaults to tf.float32.
      Must be `{tf.float32, tf.uint8, tf.int8}`
    inference_output_type: Target data type of real-number output arrays. Allows
      for a different type for output arrays.
      If `inference_type` is tf.uint8, signaling conversion to a fully quantized
      model from a quantization-aware trained output model, then
      `inference_output_type` defaults to tf.uint8.
      In all other cases, `inference_output_type` must be tf.float32, an error
      will be thrown otherwise.
      Must be `{tf.float32, tf.uint8, tf.int8}`
    output_format: Output file format. Currently must be `{TFLITE,
      GRAPHVIZ_DOT}`. (default TFLITE)
    quantized_input_stats: Dict of strings representing input tensor names
      mapped to tuple of floats representing the mean and standard deviation
      of the training data (e.g., {\"foo\" : (0., 1.)}). Only need if
      `inference_input_type` is `QUANTIZED_UINT8`.
      real_input_value = (quantized_input_value - mean_value) / std_dev_value.
      (default {})
    default_ranges_stats: Tuple of integers representing (min, max) range values
      for all arrays without a specified range. Intended for experimenting with
      quantization via \"dummy quantization\". (default None)
    drop_control_dependency: Boolean indicating whether to drop control
      dependencies silently. This is due to TFLite not supporting control
      dependencies. (default True)
    reorder_across_fake_quant: Boolean indicating whether to reorder FakeQuant
      nodes in unexpected locations. Used when the location of the FakeQuant
      nodes is preventing graph transformations necessary to convert the graph.
      Results in a graph that differs from the quantized training graph,
      potentially causing differing arithmetic behavior. (default False)
    change_concat_input_ranges: Boolean to change behavior of min/max ranges for
      inputs and outputs of the concat operator for quantized models. Changes
      the ranges of concat operator overlap when true. (default False)
    allow_custom_ops: Boolean indicating whether to allow custom operations.
      When false any unknown operation is an error. When true, custom ops are
      created for any op that is unknown. The developer will need to provide
      these to the TensorFlow Lite runtime with a custom resolver.
      (default False)
    post_training_quantize: Deprecated. Please specify `[Optimize.DEFAULT]` for
      `optimizations` instead. Boolean indicating whether to quantize the
      weights of the converted float model.  Model size will be reduced and
      there will be latency improvements (at the cost of accuracy).
      (default False)
    dump_graphviz_dir: Full filepath of folder to dump the graphs at various
      stages of processing GraphViz .dot files. Preferred over
      --output_format=GRAPHVIZ_DOT in order to keep the requirements of the
      output file. (default None)
    dump_graphviz_video: Boolean indicating whether to dump the graph after
      every graph transformation. (default False)
    target_ops: Deprecated. Please specify `target_spec.supported_ops` instead.
      Set of OpsSet options indicating which converter to use.
      (default set([OpsSet.TFLITE_BUILTINS]))
    target_spec: Experimental flag, subject to change. Specification of target
      device.
    optimizations: Experimental flag, subject to change. A list of optimizations
      to apply when converting the model. E.g. `[Optimize.DEFAULT]`
    representative_dataset: A representative dataset that can be used to
      generate input and output samples for the model. The converter can use
      the dataset to evaluate different optimizations.
    experimental_enable_mlir_converter: Experimental flag, subject to change.
      Enables the MLIR converter instead of the TOCO converter.

  Example usage:

    ```python
    # Converting a GraphDef from session.
    converter = lite.TFLiteConverter.from_session(sess, in_tensors, out_tensors)
    tflite_model = converter.convert()
    open(\"converted_model.tflite\", \"wb\").write(tflite_model)

    # Converting a GraphDef from file.
    converter = lite.TFLiteConverter.from_frozen_graph(
      graph_def_file, input_arrays, output_arrays)
    tflite_model = converter.convert()
    open(\"converted_model.tflite\", \"wb\").write(tflite_model)

    # Converting a SavedModel.
    converter = lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    open(\"converted_model.tflite\", \"wb\").write(tflite_model)

    # Converting a tf.keras model.
    converter = lite.TFLiteConverter.from_keras_model_file(keras_model)
    tflite_model = converter.convert()
    open(\"converted_model.tflite\", \"wb\").write(tflite_model)
    ```
  "
  [ graph_def input_tensors output_tensors input_arrays_with_shape output_arrays experimental_debug_info_func ]
  (py/call-attr lite "TFLiteConverter"  graph_def input_tensors output_tensors input_arrays_with_shape output_arrays experimental_debug_info_func ))

(defn convert 
  "Converts a TensorFlow GraphDef based on instance variables.

    Returns:
      The converted data in serialized format. Either a TFLite Flatbuffer or a
      Graphviz graph depending on value in `output_format`.

    Raises:
      ValueError:
        Input shape is not specified.
        None value for dimension in input_tensor.
    "
  [ self  ]
  (py/call-attr self "convert"  self  ))

(defn get-input-arrays 
  "Returns a list of the names of the input tensors.

    Returns:
      List of strings.
    "
  [ self  ]
  (py/call-attr self "get_input_arrays"  self  ))
