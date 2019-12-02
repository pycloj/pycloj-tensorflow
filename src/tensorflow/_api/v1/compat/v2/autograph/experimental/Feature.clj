(ns tensorflow.-api.v1.compat.v2.autograph.experimental.Feature
  "This enumeration represents optional conversion options.

  These conversion options are experimental. They are subject to change without
  notice and offer no guarantees.

  _Example Usage_

  ```python
  optionals= tf.autograph.experimental.Feature.EQUALITY_OPERATORS
  @tf.function(experimental_autograph_options=optionals)
  def f(i):
    if i == 0:  # EQUALITY_OPERATORS allows the use of == here.
      tf.print('i is zero')
  ```

  Attributes:
    ALL: Enable all features.
    AUTO_CONTROL_DEPS: Insert of control dependencies in the generated code.
    ASSERT_STATEMENTS: Convert Tensor-dependent assert statements to tf.Assert.
    BUILTIN_FUNCTIONS: Convert builtin functions applied to Tensors to
      their TF counterparts.
    EQUALITY_OPERATORS: Whether to convert the comparison operators, like
      equality. This is soon to be deprecated as support is being added to the
      Tensor class.
    LISTS: Convert list idioms, like initializers, slices, append, etc.
    NAME_SCOPES: Insert name scopes that name ops according to context, like the
      function they were defined in.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce experimental (import-module "tensorflow._api.v1.compat.v2.autograph.experimental"))
(defn Feature 
  "This enumeration represents optional conversion options.

  These conversion options are experimental. They are subject to change without
  notice and offer no guarantees.

  _Example Usage_

  ```python
  optionals= tf.autograph.experimental.Feature.EQUALITY_OPERATORS
  @tf.function(experimental_autograph_options=optionals)
  def f(i):
    if i == 0:  # EQUALITY_OPERATORS allows the use of == here.
      tf.print('i is zero')
  ```

  Attributes:
    ALL: Enable all features.
    AUTO_CONTROL_DEPS: Insert of control dependencies in the generated code.
    ASSERT_STATEMENTS: Convert Tensor-dependent assert statements to tf.Assert.
    BUILTIN_FUNCTIONS: Convert builtin functions applied to Tensors to
      their TF counterparts.
    EQUALITY_OPERATORS: Whether to convert the comparison operators, like
      equality. This is soon to be deprecated as support is being added to the
      Tensor class.
    LISTS: Convert list idioms, like initializers, slices, append, etc.
    NAME_SCOPES: Insert name scopes that name ops according to context, like the
      function they were defined in.
  "
  [value names module qualname type  & {:keys [start]} ]
    (py/call-attr-kw experimental "Feature" [value names module qualname type] {:start start }))
