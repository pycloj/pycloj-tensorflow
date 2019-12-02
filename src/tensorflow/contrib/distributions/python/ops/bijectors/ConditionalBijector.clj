(ns tensorflow.contrib.distributions.python.ops.bijectors.ConditionalBijector
  "Conditional Bijector is a Bijector that allows intrinsic conditioning."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce bijectors (import-module "tensorflow.contrib.distributions.python.ops.bijectors"))

(defn ConditionalBijector 
  "Conditional Bijector is a Bijector that allows intrinsic conditioning."
  [graph_parents & {:keys [is_constant_jacobian validate_args dtype forward_min_event_ndims inverse_min_event_ndims name]
                       :or {dtype None forward_min_event_ndims None inverse_min_event_ndims None name None}} ]
    (py/call-attr-kw bijectors "ConditionalBijector" [graph_parents] {:is_constant_jacobian is_constant_jacobian :validate_args validate_args :dtype dtype :forward_min_event_ndims forward_min_event_ndims :inverse_min_event_ndims inverse_min_event_ndims :name name }))

(defn dtype 
  "dtype of `Tensor`s transformable by this distribution."
  [ self ]
    (py/call-attr self "dtype"))
(defn forward 
  "

##### `kwargs`:

*  `**condition_kwargs`: Named arguments forwarded to subclass implementation."
  [self x  & {:keys [name]} ]
    (py/call-attr-kw self "forward" [x] {:name name }))

(defn forward-event-shape 
  "Shape of a single sample from a single batch as a `TensorShape`.

    Same meaning as `forward_event_shape_tensor`. May be only partially defined.

    Args:
      input_shape: `TensorShape` indicating event-portion shape passed into
        `forward` function.

    Returns:
      forward_event_shape_tensor: `TensorShape` indicating event-portion shape
        after applying `forward`. Possibly unknown.
    "
  [ self input_shape ]
  (py/call-attr self "forward_event_shape"  self input_shape ))
(defn forward-event-shape-tensor 
  "Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

    Args:
      input_shape: `Tensor`, `int32` vector indicating event-portion shape
        passed into `forward` function.
      name: name to give to the op

    Returns:
      forward_event_shape_tensor: `Tensor`, `int32` vector indicating
        event-portion shape after applying `forward`.
    "
  [self input_shape  & {:keys [name]} ]
    (py/call-attr-kw self "forward_event_shape_tensor" [input_shape] {:name name }))
(defn forward-log-det-jacobian 
  "

##### `kwargs`:

*  `**condition_kwargs`: Named arguments forwarded to subclass implementation."
  [self x event_ndims  & {:keys [name]} ]
    (py/call-attr-kw self "forward_log_det_jacobian" [x event_ndims] {:name name }))

(defn forward-min-event-ndims 
  "Returns the minimal number of dimensions bijector.forward operates on."
  [ self ]
    (py/call-attr self "forward_min_event_ndims"))

(defn graph-parents 
  "Returns this `Bijector`'s graph_parents as a Python list."
  [ self ]
    (py/call-attr self "graph_parents"))
(defn inverse 
  "

##### `kwargs`:

*  `**condition_kwargs`: Named arguments forwarded to subclass implementation."
  [self y  & {:keys [name]} ]
    (py/call-attr-kw self "inverse" [y] {:name name }))

(defn inverse-event-shape 
  "Shape of a single sample from a single batch as a `TensorShape`.

    Same meaning as `inverse_event_shape_tensor`. May be only partially defined.

    Args:
      output_shape: `TensorShape` indicating event-portion shape passed into
        `inverse` function.

    Returns:
      inverse_event_shape_tensor: `TensorShape` indicating event-portion shape
        after applying `inverse`. Possibly unknown.
    "
  [ self output_shape ]
  (py/call-attr self "inverse_event_shape"  self output_shape ))
(defn inverse-event-shape-tensor 
  "Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

    Args:
      output_shape: `Tensor`, `int32` vector indicating event-portion shape
        passed into `inverse` function.
      name: name to give to the op

    Returns:
      inverse_event_shape_tensor: `Tensor`, `int32` vector indicating
        event-portion shape after applying `inverse`.
    "
  [self output_shape  & {:keys [name]} ]
    (py/call-attr-kw self "inverse_event_shape_tensor" [output_shape] {:name name }))
(defn inverse-log-det-jacobian 
  "

##### `kwargs`:

*  `**condition_kwargs`: Named arguments forwarded to subclass implementation."
  [self y event_ndims  & {:keys [name]} ]
    (py/call-attr-kw self "inverse_log_det_jacobian" [y event_ndims] {:name name }))

(defn inverse-min-event-ndims 
  "Returns the minimal number of dimensions bijector.inverse operates on."
  [ self ]
    (py/call-attr self "inverse_min_event_ndims"))

(defn is-constant-jacobian 
  "Returns true iff the Jacobian matrix is not a function of x.

    Note: Jacobian matrix is either constant for both forward and inverse or
    neither.

    Returns:
      is_constant_jacobian: Python `bool`.
    "
  [ self ]
    (py/call-attr self "is_constant_jacobian"))

(defn name 
  "Returns the string name of this `Bijector`."
  [ self ]
    (py/call-attr self "name"))

(defn validate-args 
  "Returns True if Tensor arguments will be validated."
  [ self ]
    (py/call-attr self "validate_args"))
