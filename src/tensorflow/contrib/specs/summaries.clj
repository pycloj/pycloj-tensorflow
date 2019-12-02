(ns tensorflow.contrib.specs.python.summaries
  "Functions for summarizing and describing TensorFlow graphs.

This contains functions that generate string descriptions from
TensorFlow graphs, for debugging, testing, and model size
estimation.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce summaries (import-module "tensorflow.contrib.specs.python.summaries"))

(defn tf-left-split 
  "Split the parameters of op for left recursion.

  Args:
    op: tf.Operation

  Returns:
    A tuple of the leftmost input tensor and a list of the
    remaining arguments.
  "
  [ op ]
  (py/call-attr summaries "tf_left_split"  op ))

(defn tf-num-params 
  "Number of parameters in a TensorFlow subgraph.

  Args:
      x: root of the subgraph (Tensor, Operation)

  Returns:
      Total number of elements found in all Variables
      in the subgraph.
  "
  [ x ]
  (py/call-attr summaries "tf_num_params"  x ))

(defn tf-parameter-iter 
  "Iterate over the left branches of a graph and yield sizes.

  Args:
      x: root of the subgraph (Tensor, Operation)

  Yields:
      A triple of name, number of params, and shape.
  "
  [ x ]
  (py/call-attr summaries "tf_parameter_iter"  x ))
(defn tf-parameter-summary 
  "Summarize parameters by depth.

  Args:
      x: root of the subgraph (Tensor, Operation)
      printer: print function for output
      combine: combine layers by top-level scope
  "
  [x  & {:keys [printer combine]} ]
    (py/call-attr-kw summaries "tf_parameter_summary" [x] {:printer printer :combine combine }))

(defn tf-print 
  "A simple print function for a TensorFlow graph.

  Args:
      x: a tf.Tensor or tf.Operation
      depth: current printing depth
      finished: set of nodes already output
      printer: print function to use

  Returns:
      Total number of parameters found in the
      subtree.
  "
  [x & {:keys [depth finished printer]
                       :or {finished None}} ]
    (py/call-attr-kw summaries "tf_print" [x] {:depth depth :finished finished :printer printer }))
(defn tf-spec-print 
  "Print a tree representing the spec.

  Args:
      spec: specification
      inputs: input to the spec construction (usually a Tensor)
      input_shape: optional shape of input
      input_type: type of the input tensor
  "
  [spec inputs input_shape  & {:keys [input_type]} ]
    (py/call-attr-kw summaries "tf_spec_print" [spec inputs input_shape] {:input_type input_type }))
(defn tf-spec-structure 
  "Return a postfix representation of the specification.

  This is intended to be used as part of test cases to
  check for gross differences in the structure of the graph.
  The resulting string is not invertible or unabiguous
  and cannot be used to reconstruct the graph accurately.

  Args:
      spec: specification
      inputs: input to the spec construction (usually a Tensor)
      input_shape: tensor shape (in lieu of inputs)
      input_type: type of the input tensor

  Returns:
      A string with a postfix representation of the
      specification.
  "
  [spec inputs input_shape  & {:keys [input_type]} ]
    (py/call-attr-kw summaries "tf_spec_structure" [spec inputs input_shape] {:input_type input_type }))
(defn tf-spec-summary 
  "Output a summary of the specification.

  This prints a list of left-most tensor operations and summarized the
  variables found in the right branches. This kind of representation
  is particularly useful for networks that are generally structured
  like pipelines.

  Args:
      spec: specification
      inputs: input to the spec construction (usually a Tensor)
      input_shape: optional shape of input
      input_type: type of the input tensor
  "
  [spec inputs input_shape  & {:keys [input_type]} ]
    (py/call-attr-kw summaries "tf_spec_summary" [spec inputs input_shape] {:input_type input_type }))

(defn tf-structure 
  "A postfix expression summarizing the TF graph.

  This is intended to be used as part of test cases to
  check for gross differences in the structure of the graph.
  The resulting string is not invertible or unabiguous
  and cannot be used to reconstruct the graph accurately.

  Args:
      x: a tf.Tensor or tf.Operation
      include_shapes: include shapes in the output string
      finished: a set of ops that have already been output

  Returns:
      A string representing the structure as a string of
      postfix operations.
  "
  [x & {:keys [include_shapes finished]
                       :or {finished None}} ]
    (py/call-attr-kw summaries "tf_structure" [x] {:include_shapes include_shapes :finished finished }))
