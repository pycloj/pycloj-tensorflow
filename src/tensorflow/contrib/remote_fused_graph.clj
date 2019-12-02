(ns tensorflow.contrib.remote-fused-graph.pylib
  "Remote fused graph ops python library.

## This package provides classes for remote fused graph ops.

"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce remote-fused-graph (import-module "tensorflow.contrib.remote_fused_graph.pylib"))

(defn remote-fused-graph-execute 
  "A wrapper for remote_fused_graph_execute."
  [ inputs output_types graph_def graph_input_node_names graph_output_node_names executor_name serialized_executor_parameters default_graph_input_tensor_type_shapes default_graph_output_tensor_type_shapes ]
  (py/call-attr remote-fused-graph "remote_fused_graph_execute"  inputs output_types graph_def graph_input_node_names graph_output_node_names executor_name serialized_executor_parameters default_graph_input_tensor_type_shapes default_graph_output_tensor_type_shapes ))
