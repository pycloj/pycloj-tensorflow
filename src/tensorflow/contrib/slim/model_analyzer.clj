(ns tensorflow.contrib.slim.python.slim.model-analyzer
  "Tools for analyzing the operations and variables in a TensorFlow graph.

To analyze the operations in a graph:

  images, labels = LoadData(...)
  predictions = MyModel(images)

  slim.model_analyzer.analyze_ops(tf.compat.v1.get_default_graph(),
  print_info=True)

To analyze the model variables in a graph:

  variables = tf.compat.v1.model_variables()
  slim.model_analyzer.analyze_vars(variables, print_info=False)
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce model-analyzer (import-module "tensorflow.contrib.slim.python.slim.model_analyzer"))
(defn analyze-ops 
  "Compute the estimated size of the ops.outputs in the graph.

  Args:
    graph: the graph containing the operations.
    print_info: Optional, if true print ops and their outputs.

  Returns:
    total size of the ops.outputs
  "
  [graph  & {:keys [print_info]} ]
    (py/call-attr-kw model-analyzer "analyze_ops" [graph] {:print_info print_info }))
(defn analyze-vars 
  "Prints the names and shapes of the variables.

  Args:
    variables: list of variables, for example tf.compat.v1.global_variables().
    print_info: Optional, if true print variables and their shape.

  Returns:
    (total size of the variables, total bytes of the variables)
  "
  [variables  & {:keys [print_info]} ]
    (py/call-attr-kw model-analyzer "analyze_vars" [variables] {:print_info print_info }))

(defn tensor-description 
  "Returns a compact and informative string about a tensor.

  Args:
    var: A tensor variable.

  Returns:
    a string with type and size, e.g.: (float32 1x8x8x1024).
  "
  [ var ]
  (py/call-attr model-analyzer "tensor_description"  var ))
