(ns tensorflow.contrib.receptive-field.receptive-field-api
  "Module that declares the functions in tf.contrib.receptive_field's API."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce receptive-field (import-module "tensorflow.contrib.receptive_field.receptive_field_api"))

(defn compute-receptive-field-from-graph-def 
  "Computes receptive field (RF) parameters from a Graph or GraphDef object.

  The algorithm stops the calculation of the receptive field whenever it
  encounters an operation in the list `stop_propagation`. Stopping the
  calculation early can be useful to calculate the receptive field of a
  subgraph such as a single branch of the
  [inception network](https://arxiv.org/abs/1512.00567).

  Args:
    graph_def: Graph or GraphDef object.
    input_node: Name of the input node or Tensor object from graph.
    output_node: Name of the output node or Tensor object from graph.
    stop_propagation: List of operations or scope names for which to stop the
      propagation of the receptive field.
    input_resolution: 2D list. If the input resolution to the model is fixed and
      known, this may be set. This is helpful for cases where the RF parameters
      vary depending on the input resolution (this happens since SAME padding in
      tensorflow depends on input resolution in general). If this is None, it is
      assumed that the input resolution is unknown, so some RF parameters may be
      unknown (depending on the model architecture).

  Returns:
    rf_size_x: Receptive field size of network in the horizontal direction, with
      respect to specified input and output.
    rf_size_y: Receptive field size of network in the vertical direction, with
      respect to specified input and output.
    effective_stride_x: Effective stride of network in the horizontal direction,
      with respect to specified input and output.
    effective_stride_y: Effective stride of network in the vertical direction,
      with respect to specified input and output.
    effective_padding_x: Effective padding of network in the horizontal
      direction, with respect to specified input and output.
    effective_padding_y: Effective padding of network in the vertical
      direction, with respect to specified input and output.

  Raises:
    ValueError: If network is not aligned or if either input or output nodes
      cannot be found. For network criterion alignment, see
      photos/vision/features/delf/g3doc/rf_computation.md
  "
  [ graph_def input_node output_node stop_propagation input_resolution ]
  (py/call-attr receptive-field "compute_receptive_field_from_graph_def"  graph_def input_node output_node stop_propagation input_resolution ))

(defn get-compute-order 
  "Computes order of computation for a given CNN graph.

  Optionally, the function may also compute the input and output feature map
  resolutions at each node. In this case, input_node_name and input_node_size
  must be set. Note that if a node's op type is unknown, the input and output
  resolutions are ignored and set to None.

  Args:
    graph_def: GraphDef object.
    input_node_name: Name of node with fixed input resolution (optional). This
      is usually the node name for the input image in a CNN.
    input_node_size: 2D list of integers, fixed input resolution to use
      (optional). This is usually the input resolution used for the input image
      in a CNN (common examples are: [224, 224], [299, 299], [321, 321]).
  Returns:
    node_info: Default dict keyed by node name, mapping to a named tuple with
      the following fields:
      - order: Integer denoting topological order;
      - node: NodeDef for the given node;
      - input_size: 2D list of integers, denoting the input spatial resolution
        to the node;
      - output_size: 2D list of integers, denoting the output spatial resolution
        of the node.
    name_to_node: Dict keyed by node name, each entry containing the node's
      NodeDef.
  "
  [graph_def & {:keys [input_node_name input_node_size]
                       :or {input_node_size None}} ]
    (py/call-attr-kw receptive-field "get_compute_order" [graph_def] {:input_node_name input_node_name :input_node_size input_node_size }))
