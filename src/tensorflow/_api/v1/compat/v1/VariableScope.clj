(ns tensorflow.-api.v1.compat.v1.VariableScope
  "Variable scope object to carry defaults to provide to `get_variable`.

  Many of the arguments we need for `get_variable` in a variable store are most
  easily handled with a context. This object is used for the defaults.

  Attributes:
    name: name of the current scope, used as prefix in get_variable.
    initializer: default initializer passed to get_variable.
    regularizer: default regularizer passed to get_variable.
    reuse: Boolean, None, or tf.compat.v1.AUTO_REUSE, setting the reuse in
      get_variable. When eager execution is enabled this argument is always
      forced to be False.
    caching_device: string, callable, or None: the caching device passed to
      get_variable.
    partitioner: callable or `None`: the partitioner passed to `get_variable`.
    custom_getter: default custom getter passed to get_variable.
    name_scope: The name passed to `tf.name_scope`.
    dtype: default type passed to get_variable (defaults to DT_FLOAT).
    use_resource: if False, create a normal Variable; if True create an
      experimental ResourceVariable with well-defined semantics. Defaults to
      False (will later change to True). When eager execution is enabled this
      argument is always forced to be True.
    constraint: An optional projection function to be applied to the variable
      after being updated by an `Optimizer` (e.g. used to implement norm
      constraints or value constraints for layer weights). The function must
      take as input the unprojected Tensor representing the value of the
      variable and return the Tensor for the projected value (which must have
      the same shape). Constraints are not safe to use when doing asynchronous
      distributed training.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce v1 (import-module "tensorflow._api.v1.compat.v1"))

(defn VariableScope 
  "Variable scope object to carry defaults to provide to `get_variable`.

  Many of the arguments we need for `get_variable` in a variable store are most
  easily handled with a context. This object is used for the defaults.

  Attributes:
    name: name of the current scope, used as prefix in get_variable.
    initializer: default initializer passed to get_variable.
    regularizer: default regularizer passed to get_variable.
    reuse: Boolean, None, or tf.compat.v1.AUTO_REUSE, setting the reuse in
      get_variable. When eager execution is enabled this argument is always
      forced to be False.
    caching_device: string, callable, or None: the caching device passed to
      get_variable.
    partitioner: callable or `None`: the partitioner passed to `get_variable`.
    custom_getter: default custom getter passed to get_variable.
    name_scope: The name passed to `tf.name_scope`.
    dtype: default type passed to get_variable (defaults to DT_FLOAT).
    use_resource: if False, create a normal Variable; if True create an
      experimental ResourceVariable with well-defined semantics. Defaults to
      False (will later change to True). When eager execution is enabled this
      argument is always forced to be True.
    constraint: An optional projection function to be applied to the variable
      after being updated by an `Optimizer` (e.g. used to implement norm
      constraints or value constraints for layer weights). The function must
      take as input the unprojected Tensor representing the value of the
      variable and return the Tensor for the projected value (which must have
      the same shape). Constraints are not safe to use when doing asynchronous
      distributed training.
  "
  [reuse & {:keys [name initializer regularizer caching_device partitioner custom_getter name_scope dtype use_resource constraint]
                       :or {initializer None regularizer None caching_device None partitioner None custom_getter None use_resource None constraint None}} ]
    (py/call-attr-kw v1 "VariableScope" [reuse] {:name name :initializer initializer :regularizer regularizer :caching_device caching_device :partitioner partitioner :custom_getter custom_getter :name_scope name_scope :dtype dtype :use_resource use_resource :constraint constraint }))

(defn caching-device 
  ""
  [ self ]
    (py/call-attr self "caching_device"))

(defn constraint 
  ""
  [ self ]
    (py/call-attr self "constraint"))

(defn custom-getter 
  ""
  [ self ]
    (py/call-attr self "custom_getter"))

(defn dtype 
  ""
  [ self ]
    (py/call-attr self "dtype"))

(defn get-collection 
  "Get this scope's variables."
  [ self name ]
  (py/call-attr self "get_collection"  self name ))

(defn get-variable 
  "Gets an existing variable with this name or create a new one."
  [self var_store name shape dtype initializer regularizer reuse trainable collections caching_device partitioner & {:keys [validate_shape use_resource custom_getter constraint synchronization aggregation]
                       :or {use_resource None custom_getter None constraint None}} ]
    (py/call-attr-kw self "get_variable" [var_store name shape dtype initializer regularizer reuse trainable collections caching_device partitioner] {:validate_shape validate_shape :use_resource use_resource :custom_getter custom_getter :constraint constraint :synchronization synchronization :aggregation aggregation }))

(defn global-variables 
  "Get this scope's global variables."
  [ self  ]
  (py/call-attr self "global_variables"  self  ))

(defn initializer 
  ""
  [ self ]
    (py/call-attr self "initializer"))

(defn local-variables 
  "Get this scope's local variables."
  [ self  ]
  (py/call-attr self "local_variables"  self  ))

(defn name 
  ""
  [ self ]
    (py/call-attr self "name"))

(defn original-name-scope 
  ""
  [ self ]
    (py/call-attr self "original_name_scope"))

(defn partitioner 
  ""
  [ self ]
    (py/call-attr self "partitioner"))

(defn regularizer 
  ""
  [ self ]
    (py/call-attr self "regularizer"))

(defn reuse 
  ""
  [ self ]
    (py/call-attr self "reuse"))

(defn reuse-variables 
  "Reuse variables in this scope."
  [ self  ]
  (py/call-attr self "reuse_variables"  self  ))

(defn set-caching-device 
  "Set caching_device for this scope."
  [ self caching_device ]
  (py/call-attr self "set_caching_device"  self caching_device ))

(defn set-custom-getter 
  "Set custom getter for this scope."
  [ self custom_getter ]
  (py/call-attr self "set_custom_getter"  self custom_getter ))

(defn set-dtype 
  "Set data type for this scope."
  [ self dtype ]
  (py/call-attr self "set_dtype"  self dtype ))

(defn set-initializer 
  "Set initializer for this scope."
  [ self initializer ]
  (py/call-attr self "set_initializer"  self initializer ))

(defn set-partitioner 
  "Set partitioner for this scope."
  [ self partitioner ]
  (py/call-attr self "set_partitioner"  self partitioner ))

(defn set-regularizer 
  "Set regularizer for this scope."
  [ self regularizer ]
  (py/call-attr self "set_regularizer"  self regularizer ))

(defn set-use-resource 
  "Sets whether to use ResourceVariables for this scope."
  [ self use_resource ]
  (py/call-attr self "set_use_resource"  self use_resource ))

(defn trainable-variables 
  "Get this scope's trainable variables."
  [ self  ]
  (py/call-attr self "trainable_variables"  self  ))

(defn use-resource 
  ""
  [ self ]
    (py/call-attr self "use_resource"))
