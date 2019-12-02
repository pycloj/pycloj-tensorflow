(ns tensorflow.contrib.quantize
  "Functions for rewriting graphs for quantized training."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce quantize (import-module "tensorflow.contrib.quantize"))

(defn create-eval-graph 
  "Rewrites an eval input_graph in place for simulated quantization.

  Variables added by the rewrite get added to the global variables collection.

  The graph has fake quantization ops inserted to simulate the error
  introduced by quantization. Since the graph is transformed in place,
  the expected behavior of previously held references to nodes and tensors may
  change.

  Args:
    input_graph: The tf.Graph to be transformed, if None then defaults to the
      default graph.

  Raises:
    ValueError: If elements contains an element that isn't a tf.Tensor or
      tf.Operation.
  "
  [ input_graph ]
  (py/call-attr quantize "create_eval_graph"  input_graph ))
(defn create-training-graph 
  "Rewrites a training input_graph in place for simulated quantization.

  Variables added by the rewrite get added to the global variables collection.

  This function must be invoked prior to insertion of gradient ops in a graph
  as quantization should be modeled in both forward and backward passes.

  The graph has fake quantization ops inserted to simulate the error
  introduced by quantization. Since the graph is transformed in place,
  the expected behavior of previously held references to nodes and tensors may
  change.

  The default value of quant_delay is suitable for finetuning an already trained
  floating point model (recommended).
  If one wants to train a quantized model from scratch, quant_delay should be
  set to the number of steps it take the floating point model to converge.
  Quantization will be activated at this point and effectively finetune the
  model. If quant_delay is not provided when training from scratch, training can
  often fail.

  Args:
    input_graph: The tf.Graph to be transformed.
    quant_delay: Number of steps after which weights and activations are
      quantized during training.

  Raises:
    ValueError: If elements contains an element that isn't a tf.Tensor or
      tf.Operation.
  "
  [input_graph  & {:keys [quant_delay]} ]
    (py/call-attr-kw quantize "create_training_graph" [input_graph] {:quant_delay quant_delay }))

(defn experimental-create-eval-graph 
  "Rewrites an eval input_graph in place for simulated quantization.

  Variables added by the rewrite get added to the global variables collection.

  This function has additional experimental options not (yet) available to
  create_eval_graph. The resulting behavior may be undefined.

  The graph has fake quantization ops inserted to simulate the error
  introduced by quantization. Since the graph is transformed in place,
  the expected behavior of previously held references to nodes and tensors may
  change.

  Args:
    input_graph: The tf.Graph to be transformed, if None then defaults to the
      default graph.
    weight_bits: Number of bits to use for quantizing weights.
    activation_bits: Number of bits to use for quantizing activations.
    symmetric: If true, use symmetric quantization limits instead of training
      the minimum and maximum of each quantization range separately.
    quant_delay: Number of steps after which weights and activations are
      quantized during eval.
    scope: The scope to be transformed. If it's not None, only the ops which
      are in this scope will be transformed.

  Raises:
    ValueError: If elements contains an element that isn't a tf.Tensor or
      tf.Operation.
  "
  [input_graph & {:keys [weight_bits activation_bits symmetric quant_delay scope]
                       :or {quant_delay None scope None}} ]
    (py/call-attr-kw quantize "experimental_create_eval_graph" [input_graph] {:weight_bits weight_bits :activation_bits activation_bits :symmetric symmetric :quant_delay quant_delay :scope scope }))

(defn experimental-create-training-graph 
  "Rewrites a training input_graph in place for simulated quantization.

  This function must be invoked prior to insertion of gradient ops in a graph
  as quantization should be modeled in both forward and backward passes.

  Variables added by the rewrite get added to the global variables collection.

  This function has additional experimental options not (yet) available to
  create_training_graph. The resulting behavior may be undefined.

  The graph has fake quantization ops inserted to simulate the error
  introduced by quantization. Since the graph is transformed in place,
  the expected behavior of previously held references to nodes and tensors may
  change.

  The default value of quant_delay is suitable for finetuning an already trained
  floating point model (recommended).
  If one wants to train a quantized model from scratch, quant_delay should be
  set to the number of steps it take the floating point model to converge.
  Quantization will be activated at this point and effectively finetune the
  model. If quant_delay is not provided when training from scratch, training can
  often fail.

  Args:
    input_graph: The tf.Graph to be transformed, if None then defaults to the
      default graph.
    weight_bits: Number of bits to use for quantizing weights.
    activation_bits: Number of bits to use for quantizing activations.
    symmetric: If true, use symmetric quantization limits instead of training
      the minimum and maximum of each quantization range separately.
    quant_delay: Number of steps after which weights and activations are
      quantized during training.
    freeze_bn_delay: Number of steps after which moving mean and variance are
      frozen and used instead of batch statistics during training.
      freeze_bn_delay should be greater than quant_delay and should correspond
      to when training has almost converged
    scope: The scope to be transformed. If it's not None, only the ops which
      are in this scope will be transformed.

  Raises:
    ValueError: If elements contains an element that isn't a tf.Tensor or
        tf.Operation.
  "
  [input_graph & {:keys [weight_bits activation_bits symmetric quant_delay freeze_bn_delay scope]
                       :or {freeze_bn_delay None scope None}} ]
    (py/call-attr-kw quantize "experimental_create_training_graph" [input_graph] {:weight_bits weight_bits :activation_bits activation_bits :symmetric symmetric :quant_delay quant_delay :freeze_bn_delay freeze_bn_delay :scope scope }))
