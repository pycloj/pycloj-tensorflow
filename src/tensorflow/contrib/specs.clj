(ns tensorflow.contrib.specs.python
  "Init file, giving convenient access to all specs ops."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce specs (import-module "tensorflow.contrib.specs.python"))

(defn Dwm 
  "Depth-wise convolution + softmax (used after LSTM)."
  [ n ]
  (py/call-attr specs "Dwm"  n ))

(defn Dws 
  "Depth-wise convolution + sigmoid (used after LSTM)."
  [ n ]
  (py/call-attr specs "Dws"  n ))

(defn External 
  "Import a function from an external module.

  Note that the `module_name` must be a module name
  that works with the usual import mechanisms. Shorthands
  like \"tf.nn\" will not work.

  Args:
      module_name: name of the module
      function_name: name of the function within the module

  Returns:
      Function-wrapped value of symbol.
  "
  [ module_name function_name ]
  (py/call-attr specs "External"  module_name function_name ))

(defn Import 
  "Import a function by exec.

  Args:
      statements: Python statements

  Returns:
      Function-wrapped value of `f`.

  Raises:
      ValueError: the statements didn't define a value for \"f\"
  "
  [ statements ]
  (py/call-attr specs "Import"  statements ))

(defn Lf 
  "Log-uniform distributed floatint point number."
  [ lo hi ]
  (py/call-attr specs "Lf"  lo hi ))

(defn Li 
  "Log-uniform distributed integer, inclusive limits."
  [ lo hi ]
  (py/call-attr specs "Li"  lo hi ))
(defn Nt 
  "Normally distributed floating point number with truncation."
  [mu sigma  & {:keys [limit]} ]
    (py/call-attr-kw specs "Nt" [mu sigma] {:limit limit }))

(defn Uf 
  "Uniformly distributed floating number."
  [ & {:keys [lo hi]} ]
   (py/call-attr-kw specs "Uf" [] {:lo lo :hi hi }))

(defn Ui 
  "Uniformly distributed integer, inclusive limits."
  [ lo hi ]
  (py/call-attr specs "Ui"  lo hi ))

(defn Var 
  "Implements an operator that generates a variable.

  This function is still experimental. Use it only
  for generating a single variable instance for
  each name.

  Args:
      name: Name of the variable.
      *args: Other arguments to get_variable.
      **kw: Other keywords for get_variable.

  Returns:
      A specs object for generating a variable.
  "
  [ name ]
  (py/call-attr specs "Var"  name ))

(defn check-keywords 
  "Check for common Python keywords in spec.

  This function discourages the use of complex constructs
  in TensorFlow specs; it doesn't completely prohibit them
  (if necessary, we could check the AST).

  Args:
      spec: spec string

  Raises:
      ValueError: raised if spec contains a prohibited keyword.
  "
  [ spec ]
  (py/call-attr specs "check_keywords"  spec ))

(defn create-net 
  "Evaluates a spec and creates a network instance given the inputs.

  Args:
      spec: specification as a string, ending with a `net = ...` statement
      inputs: input that `net` is applied to
      environment: a dictionary of input bindings

  Returns:
      A callable that instantiates the `net` binding.

  Raises:
      ValueError: spec failed to create a `net`
      Exception: other exceptions raised during execution of `spec`
  "
  [ spec inputs environment ]
  (py/call-attr specs "create_net"  spec inputs environment ))

(defn create-net-fun 
  "Evaluates a spec and returns the binding of `net`.

  Specs are written in a DSL based on function composition.  A spec
  like `net = Cr(64, [3, 3])` assigns an object that represents a
  single argument function capable of creating a network to
  the variable `net`.

  Args:
      spec: specification as a string, ending with a `net = ...` statement
      environment: a dictionary of input bindings

  Returns:
      A callable that instantiates the `net` binding.

  Raises:
      ValueError: spec failed to create a `net`
      Exception: other exceptions raised during execution of `spec`

  "
  [ spec environment ]
  (py/call-attr specs "create_net_fun"  spec environment ))

(defn debug 
  "Turn on/off debugging mode.

  Debugging mode prints more information about the construction
  of a network.

  Args:
      mode: True if turned on, False otherwise
  "
  [ & {:keys [mode]} ]
   (py/call-attr-kw specs "debug" [] {:mode mode }))

(defn eval-params 
  "Evaluates a parameter specification and returns the environment.

  Args:
      params: parameter assignments as a string
      environment: a dictionary of input bindings

  Returns:
      Environment with additional bindings created by
      executing `params`

  Raises:
      Exception: other exceptions raised during execution of `params`
  "
  [ params environment ]
  (py/call-attr specs "eval_params"  params environment ))

(defn eval-spec 
  "Evaluates a spec and returns the environment.

  This function allows you to use a spec to obtain multiple bindings
  in an environment. That is useful if you use the spec language to
  specify multiple components of a larger network, for example: \"left
  = Cr(64, [5,5]); right = Fc(64)\" Usually, you will want to use
  `create_net` or `create_net_fun` below.

  Args:
      spec: specification as a string
      environment: a dictionary of input bindings

  Returns:
      Environment with additional bindings created by spec.

  Raises:
      Exception: other exceptions raised during execution of `spec`

  "
  [ spec environment ]
  (py/call-attr specs "eval_spec"  spec environment ))
(defn get-positional 
  "Interpolates keyword arguments into argument lists.

  If `kw` contains keywords of the form \"_0\", \"_1\", etc., these
  are positionally interpolated into the argument list.

  Args:
      args: argument list
      kw: keyword dictionary
      kw_overrides: key/value pairs that override kw

  Returns:
      (new_args, new_kw), new argument lists and keyword dictionaries
      with values interpolated.
  "
  [args kw  & {:keys [kw_overrides]} ]
    (py/call-attr-kw specs "get_positional" [args kw] {:kw_overrides kw_overrides }))

(defn tf-left-split 
  "Split the parameters of op for left recursion.

  Args:
    op: tf.Operation

  Returns:
    A tuple of the leftmost input tensor and a list of the
    remaining arguments.
  "
  [ op ]
  (py/call-attr specs "tf_left_split"  op ))

(defn tf-num-params 
  "Number of parameters in a TensorFlow subgraph.

  Args:
      x: root of the subgraph (Tensor, Operation)

  Returns:
      Total number of elements found in all Variables
      in the subgraph.
  "
  [ x ]
  (py/call-attr specs "tf_num_params"  x ))

(defn tf-parameter-iter 
  "Iterate over the left branches of a graph and yield sizes.

  Args:
      x: root of the subgraph (Tensor, Operation)

  Yields:
      A triple of name, number of params, and shape.
  "
  [ x ]
  (py/call-attr specs "tf_parameter_iter"  x ))
(defn tf-parameter-summary 
  "Summarize parameters by depth.

  Args:
      x: root of the subgraph (Tensor, Operation)
      printer: print function for output
      combine: combine layers by top-level scope
  "
  [x  & {:keys [printer combine]} ]
    (py/call-attr-kw specs "tf_parameter_summary" [x] {:printer printer :combine combine }))

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
    (py/call-attr-kw specs "tf_print" [x] {:depth depth :finished finished :printer printer }))
(defn tf-spec-print 
  "Print a tree representing the spec.

  Args:
      spec: specification
      inputs: input to the spec construction (usually a Tensor)
      input_shape: optional shape of input
      input_type: type of the input tensor
  "
  [spec inputs input_shape  & {:keys [input_type]} ]
    (py/call-attr-kw specs "tf_spec_print" [spec inputs input_shape] {:input_type input_type }))
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
    (py/call-attr-kw specs "tf_spec_structure" [spec inputs input_shape] {:input_type input_type }))
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
    (py/call-attr-kw specs "tf_spec_summary" [spec inputs input_shape] {:input_type input_type }))

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
    (py/call-attr-kw specs "tf_structure" [x] {:include_shapes include_shapes :finished finished }))
