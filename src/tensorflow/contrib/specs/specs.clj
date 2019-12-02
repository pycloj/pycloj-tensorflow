(ns tensorflow.contrib.specs.python.specs
  "Builder for TensorFlow models specified using specs_ops.

"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce specs (import-module "tensorflow.contrib.specs.python.specs"))

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
