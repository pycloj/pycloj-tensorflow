(ns tensorflow.python.keras.api.-v1.keras.constraints.MaxNorm
  "MaxNorm weight constraint.

  Constrains the weights incident to each hidden unit
  to have a norm less than or equal to a desired value.

  Arguments:
      m: the maximum norm for the incoming weights.
      axis: integer, axis along which to calculate weight norms.
          For instance, in a `Dense` layer the weight matrix
          has shape `(input_dim, output_dim)`,
          set `axis` to `0` to constrain each weight vector
          of length `(input_dim,)`.
          In a `Conv2D` layer with `data_format=\"channels_last\"`,
          the weight tensor has shape
          `(rows, cols, input_depth, output_depth)`,
          set `axis` to `[0, 1, 2]`
          to constrain the weights of each filter tensor of size
          `(rows, cols, input_depth)`.

  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce constraints (import-module "tensorflow.python.keras.api._v1.keras.constraints"))

(defn MaxNorm 
  "MaxNorm weight constraint.

  Constrains the weights incident to each hidden unit
  to have a norm less than or equal to a desired value.

  Arguments:
      m: the maximum norm for the incoming weights.
      axis: integer, axis along which to calculate weight norms.
          For instance, in a `Dense` layer the weight matrix
          has shape `(input_dim, output_dim)`,
          set `axis` to `0` to constrain each weight vector
          of length `(input_dim,)`.
          In a `Conv2D` layer with `data_format=\"channels_last\"`,
          the weight tensor has shape
          `(rows, cols, input_depth, output_depth)`,
          set `axis` to `[0, 1, 2]`
          to constrain the weights of each filter tensor of size
          `(rows, cols, input_depth)`.

  "
  [ & {:keys [max_value axis]} ]
   (py/call-attr-kw constraints "MaxNorm" [] {:max_value max_value :axis axis }))

(defn get-config 
  ""
  [ self  ]
  (py/call-attr self "get_config"  self  ))
