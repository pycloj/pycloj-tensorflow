(ns tensorflow.variable-scope
  "A context manager for defining ops that creates variables (layers).

  This context manager validates that the (optional) `values` are from the same
  graph, ensures that graph is the default graph, and pushes a name scope and a
  variable scope.

  If `name_or_scope` is not None, it is used as is. If `name_or_scope` is None,
  then `default_name` is used.  In that case, if the same name has been
  previously used in the same scope, it will be made unique by appending `_N`
  to it.

  Variable scope allows you to create new variables and to share already created
  ones while providing checks to not create or share by accident. For details,
  see the [Variable Scope How To](https://tensorflow.org/guide/variables), here
  we present only a few basic examples.

  Simple example of how to create a new variable:

  ```python
  with tf.compat.v1.variable_scope(\"foo\"):
      with tf.compat.v1.variable_scope(\"bar\"):
          v = tf.compat.v1.get_variable(\"v\", [1])
          assert v.name == \"foo/bar/v:0\"
  ```

  Simple example of how to reenter a premade variable scope safely:

  ```python
  with tf.compat.v1.variable_scope(\"foo\") as vs:
    pass

  # Re-enter the variable scope.
  with tf.compat.v1.variable_scope(vs,
                         auxiliary_name_scope=False) as vs1:
    # Restore the original name_scope.
    with tf.name_scope(vs1.original_name_scope):
        v = tf.compat.v1.get_variable(\"v\", [1])
        assert v.name == \"foo/v:0\"
        c = tf.constant([1], name=\"c\")
        assert c.name == \"foo/c:0\"
  ```

  Basic example of sharing a variable AUTO_REUSE:

  ```python
  def foo():
    with tf.compat.v1.variable_scope(\"foo\", reuse=tf.compat.v1.AUTO_REUSE):
      v = tf.compat.v1.get_variable(\"v\", [1])
    return v

  v1 = foo()  # Creates v.
  v2 = foo()  # Gets the same, existing v.
  assert v1 == v2
  ```

  Basic example of sharing a variable with reuse=True:

  ```python
  with tf.compat.v1.variable_scope(\"foo\"):
      v = tf.compat.v1.get_variable(\"v\", [1])
  with tf.compat.v1.variable_scope(\"foo\", reuse=True):
      v1 = tf.compat.v1.get_variable(\"v\", [1])
  assert v1 == v
  ```

  Sharing a variable by capturing a scope and setting reuse:

  ```python
  with tf.compat.v1.variable_scope(\"foo\") as scope:
      v = tf.compat.v1.get_variable(\"v\", [1])
      scope.reuse_variables()
      v1 = tf.compat.v1.get_variable(\"v\", [1])
  assert v1 == v
  ```

  To prevent accidental sharing of variables, we raise an exception when getting
  an existing variable in a non-reusing scope.

  ```python
  with tf.compat.v1.variable_scope(\"foo\"):
      v = tf.compat.v1.get_variable(\"v\", [1])
      v1 = tf.compat.v1.get_variable(\"v\", [1])
      #  Raises ValueError(\"... v already exists ...\").
  ```

  Similarly, we raise an exception when trying to get a variable that does not
  exist in reuse mode.

  ```python
  with tf.compat.v1.variable_scope(\"foo\", reuse=True):
      v = tf.compat.v1.get_variable(\"v\", [1])
      #  Raises ValueError(\"... v does not exists ...\").
  ```

  Note that the `reuse` flag is inherited: if we open a reusing scope, then all
  its sub-scopes become reusing as well.

  A note about name scoping: Setting `reuse` does not impact the naming of other
  ops such as mult. See related discussion on
  [github#6189](https://github.com/tensorflow/tensorflow/issues/6189)

  Note that up to and including version 1.0, it was allowed (though explicitly
  discouraged) to pass False to the reuse argument, yielding undocumented
  behaviour slightly different from None. Starting at 1.1.0 passing None and
  False as reuse has exactly the same effect.

  A note about using variable scopes in multi-threaded environment: Variable
  scopes are thread local, so one thread will not see another thread's current
  scope. Also, when using `default_name`, unique scopes names are also generated
  only on a per thread basis. If the same name was used within a different
  thread, that doesn't prevent a new thread from creating the same scope.
  However, the underlying variable store is shared across threads (within the
  same graph). As such, if another thread tries to create a new variable with
  the same name as a variable created by a previous thread, it will fail unless
  reuse is True.

  Further, each thread starts with an empty variable scope. So if you wish to
  preserve name prefixes from a scope from the main thread, you should capture
  the main thread's scope and re-enter it in each thread. For e.g.

  ```
  main_thread_scope = variable_scope.get_variable_scope()

  # Thread's target function:
  def thread_target_fn(captured_scope):
    with variable_scope.variable_scope(captured_scope):
      # .... regular code for this thread


  thread = threading.Thread(target=thread_target_fn, args=(main_thread_scope,))
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
(defn variable-scope 
  "A context manager for defining ops that creates variables (layers).

  This context manager validates that the (optional) `values` are from the same
  graph, ensures that graph is the default graph, and pushes a name scope and a
  variable scope.

  If `name_or_scope` is not None, it is used as is. If `name_or_scope` is None,
  then `default_name` is used.  In that case, if the same name has been
  previously used in the same scope, it will be made unique by appending `_N`
  to it.

  Variable scope allows you to create new variables and to share already created
  ones while providing checks to not create or share by accident. For details,
  see the [Variable Scope How To](https://tensorflow.org/guide/variables), here
  we present only a few basic examples.

  Simple example of how to create a new variable:

  ```python
  with tf.compat.v1.variable_scope(\"foo\"):
      with tf.compat.v1.variable_scope(\"bar\"):
          v = tf.compat.v1.get_variable(\"v\", [1])
          assert v.name == \"foo/bar/v:0\"
  ```

  Simple example of how to reenter a premade variable scope safely:

  ```python
  with tf.compat.v1.variable_scope(\"foo\") as vs:
    pass

  # Re-enter the variable scope.
  with tf.compat.v1.variable_scope(vs,
                         auxiliary_name_scope=False) as vs1:
    # Restore the original name_scope.
    with tf.name_scope(vs1.original_name_scope):
        v = tf.compat.v1.get_variable(\"v\", [1])
        assert v.name == \"foo/v:0\"
        c = tf.constant([1], name=\"c\")
        assert c.name == \"foo/c:0\"
  ```

  Basic example of sharing a variable AUTO_REUSE:

  ```python
  def foo():
    with tf.compat.v1.variable_scope(\"foo\", reuse=tf.compat.v1.AUTO_REUSE):
      v = tf.compat.v1.get_variable(\"v\", [1])
    return v

  v1 = foo()  # Creates v.
  v2 = foo()  # Gets the same, existing v.
  assert v1 == v2
  ```

  Basic example of sharing a variable with reuse=True:

  ```python
  with tf.compat.v1.variable_scope(\"foo\"):
      v = tf.compat.v1.get_variable(\"v\", [1])
  with tf.compat.v1.variable_scope(\"foo\", reuse=True):
      v1 = tf.compat.v1.get_variable(\"v\", [1])
  assert v1 == v
  ```

  Sharing a variable by capturing a scope and setting reuse:

  ```python
  with tf.compat.v1.variable_scope(\"foo\") as scope:
      v = tf.compat.v1.get_variable(\"v\", [1])
      scope.reuse_variables()
      v1 = tf.compat.v1.get_variable(\"v\", [1])
  assert v1 == v
  ```

  To prevent accidental sharing of variables, we raise an exception when getting
  an existing variable in a non-reusing scope.

  ```python
  with tf.compat.v1.variable_scope(\"foo\"):
      v = tf.compat.v1.get_variable(\"v\", [1])
      v1 = tf.compat.v1.get_variable(\"v\", [1])
      #  Raises ValueError(\"... v already exists ...\").
  ```

  Similarly, we raise an exception when trying to get a variable that does not
  exist in reuse mode.

  ```python
  with tf.compat.v1.variable_scope(\"foo\", reuse=True):
      v = tf.compat.v1.get_variable(\"v\", [1])
      #  Raises ValueError(\"... v does not exists ...\").
  ```

  Note that the `reuse` flag is inherited: if we open a reusing scope, then all
  its sub-scopes become reusing as well.

  A note about name scoping: Setting `reuse` does not impact the naming of other
  ops such as mult. See related discussion on
  [github#6189](https://github.com/tensorflow/tensorflow/issues/6189)

  Note that up to and including version 1.0, it was allowed (though explicitly
  discouraged) to pass False to the reuse argument, yielding undocumented
  behaviour slightly different from None. Starting at 1.1.0 passing None and
  False as reuse has exactly the same effect.

  A note about using variable scopes in multi-threaded environment: Variable
  scopes are thread local, so one thread will not see another thread's current
  scope. Also, when using `default_name`, unique scopes names are also generated
  only on a per thread basis. If the same name was used within a different
  thread, that doesn't prevent a new thread from creating the same scope.
  However, the underlying variable store is shared across threads (within the
  same graph). As such, if another thread tries to create a new variable with
  the same name as a variable created by a previous thread, it will fail unless
  reuse is True.

  Further, each thread starts with an empty variable scope. So if you wish to
  preserve name prefixes from a scope from the main thread, you should capture
  the main thread's scope and re-enter it in each thread. For e.g.

  ```
  main_thread_scope = variable_scope.get_variable_scope()

  # Thread's target function:
  def thread_target_fn(captured_scope):
    with variable_scope.variable_scope(captured_scope):
      # .... regular code for this thread


  thread = threading.Thread(target=thread_target_fn, args=(main_thread_scope,))
  ```
  "
  [name_or_scope default_name values initializer regularizer caching_device partitioner custom_getter reuse dtype use_resource constraint  & {:keys [auxiliary_name_scope]} ]
    (py/call-attr-kw tensorflow "variable_scope" [name_or_scope default_name values initializer regularizer caching_device partitioner custom_getter reuse dtype use_resource constraint] {:auxiliary_name_scope auxiliary_name_scope }))
