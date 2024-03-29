(ns tensorflow.Module
  "Base neural network module class.

  A module is a named container for `tf.Variable`s, other `tf.Module`s and
  functions which apply to user input. For example a dense layer in a neural
  network might be implemented as a `tf.Module`:

  ```python
   class Dense(tf.Module):
     def __init__(self, in_features, output_features, name=None):
       super(Dense, self).__init__(name=name)
       self.w = tf.Variable(
           tf.random.normal([input_features, output_features]), name='w')
       self.b = tf.Variable(tf.zeros([output_features]), name='b')

     def __call__(self, x):
       y = tf.matmul(x, self.w) + self.b
       return tf.nn.relu(y)
  ```

  You can use the Dense layer as you would expect:

  ```python
  d = Dense(input_features=64, output_features=10)
  d(tf.ones([100, 64]))
  #==> <tf.Tensor: ...>
  ```

  By subclassing `tf.Module` instead of `object` any `tf.Variable` or
  `tf.Module` instances assigned to object properties can be collected using
  the `variables`, `trainable_variables` or `submodules` property:

  ```python
  d.variables
  #==> (<tf.Variable 'b:0' ...>, <tf.Variable 'w:0' ...>)
  ```

  Subclasses of `tf.Module` can also take advantage of the `_flatten` method
  which can be used to implement tracking of any other types.

  All `tf.Module` classes have an associated `tf.name_scope` which can be used
  to group operations in TensorBoard and create hierarchies for variable names
  which can help with debugging. We suggest using the name scope when creating
  nested submodules/parameters or for forward methods whose graph you might want
  to inspect in TensorBoard. You can enter the name scope explicitly using
  `with self.name_scope:` or you can annotate methods (apart from `__init__`)
  with `@tf.Module.with_name_scope`.

  ```python
  class MLP(tf.Module):
    def __init__(self, input_size, sizes, name=None):
      super(MLP, self).__init__(name=name)
      self.layers = []
      with self.name_scope:
        for size in sizes:
          self.layers.append(Dense(input_size=input_size, output_size=size))
          input_size = size

    @tf.Module.with_name_scope
    def __call__(self, x):
      for layer in self.layers:
        x = layer(x)
      return x
  ```
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce tensorflow (import-module "tensorflow"))

(defn Module 
  "Base neural network module class.

  A module is a named container for `tf.Variable`s, other `tf.Module`s and
  functions which apply to user input. For example a dense layer in a neural
  network might be implemented as a `tf.Module`:

  ```python
   class Dense(tf.Module):
     def __init__(self, in_features, output_features, name=None):
       super(Dense, self).__init__(name=name)
       self.w = tf.Variable(
           tf.random.normal([input_features, output_features]), name='w')
       self.b = tf.Variable(tf.zeros([output_features]), name='b')

     def __call__(self, x):
       y = tf.matmul(x, self.w) + self.b
       return tf.nn.relu(y)
  ```

  You can use the Dense layer as you would expect:

  ```python
  d = Dense(input_features=64, output_features=10)
  d(tf.ones([100, 64]))
  #==> <tf.Tensor: ...>
  ```

  By subclassing `tf.Module` instead of `object` any `tf.Variable` or
  `tf.Module` instances assigned to object properties can be collected using
  the `variables`, `trainable_variables` or `submodules` property:

  ```python
  d.variables
  #==> (<tf.Variable 'b:0' ...>, <tf.Variable 'w:0' ...>)
  ```

  Subclasses of `tf.Module` can also take advantage of the `_flatten` method
  which can be used to implement tracking of any other types.

  All `tf.Module` classes have an associated `tf.name_scope` which can be used
  to group operations in TensorBoard and create hierarchies for variable names
  which can help with debugging. We suggest using the name scope when creating
  nested submodules/parameters or for forward methods whose graph you might want
  to inspect in TensorBoard. You can enter the name scope explicitly using
  `with self.name_scope:` or you can annotate methods (apart from `__init__`)
  with `@tf.Module.with_name_scope`.

  ```python
  class MLP(tf.Module):
    def __init__(self, input_size, sizes, name=None):
      super(MLP, self).__init__(name=name)
      self.layers = []
      with self.name_scope:
        for size in sizes:
          self.layers.append(Dense(input_size=input_size, output_size=size))
          input_size = size

    @tf.Module.with_name_scope
    def __call__(self, x):
      for layer in self.layers:
        x = layer(x)
      return x
  ```
  "
  [ name ]
  (py/call-attr tensorflow "Module"  name ))

(defn name 
  "Returns the name of this module as passed or determined in the ctor.

    NOTE: This is not the same as the `self.name_scope.name` which includes
    parent module names.
    "
  [ self ]
    (py/call-attr self "name"))

(defn name-scope 
  "Returns a `tf.name_scope` instance for this class."
  [ self ]
    (py/call-attr self "name_scope"))

(defn submodules 
  "Sequence of all sub-modules.

    Submodules are modules which are properties of this module, or found as
    properties of modules which are properties of this module (and so on).

    ```
    a = tf.Module()
    b = tf.Module()
    c = tf.Module()
    a.b = b
    b.c = c
    assert list(a.submodules) == [b, c]
    assert list(b.submodules) == [c]
    assert list(c.submodules) == []
    ```

    Returns:
      A sequence of all submodules.
    "
  [ self ]
    (py/call-attr self "submodules"))

(defn trainable-variables 
  "Sequence of variables owned by this module and it's submodules.

    Note: this method uses reflection to find variables on the current instance
    and submodules. For performance reasons you may wish to cache the result
    of calling this method if you don't expect the return value to change.

    Returns:
      A sequence of variables for the current module (sorted by attribute
      name) followed by variables from all submodules recursively (breadth
      first).
    "
  [ self ]
    (py/call-attr self "trainable_variables"))

(defn variables 
  "Sequence of variables owned by this module and it's submodules.

    Note: this method uses reflection to find variables on the current instance
    and submodules. For performance reasons you may wish to cache the result
    of calling this method if you don't expect the return value to change.

    Returns:
      A sequence of variables for the current module (sorted by attribute
      name) followed by variables from all submodules recursively (breadth
      first).
    "
  [ self ]
    (py/call-attr self "variables"))
