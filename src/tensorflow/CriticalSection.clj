(ns tensorflow.CriticalSection
  "Critical section.

  A `CriticalSection` object is a resource in the graph which executes subgraphs
  in **serial** order.  A common example of a subgraph one may wish to run
  exclusively is the one given by the following function:

  ```python
  v = resource_variable_ops.ResourceVariable(0.0, name=\"v\")

  def count():
    value = v.read_value()
    with tf.control_dependencies([value]):
      with tf.control_dependencies([v.assign_add(1)]):
        return tf.identity(value)
  ```

  Here, a snapshot of `v` is captured in `value`; and then `v` is updated.
  The snapshot value is returned.

  If multiple workers or threads all execute `count` in parallel, there is no
  guarantee that access to the variable `v` is atomic at any point within
  any thread's calculation of `count`.  In fact, even implementing an atomic
  counter that guarantees that the user will see each value `0, 1, ...,` is
  currently impossible.

  The solution is to ensure any access to the underlying resource `v` is
  only processed through a critical section:

  ```python
  cs = CriticalSection()
  f1 = cs.execute(count)
  f2 = cs.execute(count)
  output = f1 + f2
  session.run(output)
  ```
  The functions `f1` and `f2` will be executed serially, and updates to `v`
  will be atomic.

  **NOTES**

  All resource objects, including the critical section and any captured
  variables of functions executed on that critical section, will be
  colocated to the same device (host and cpu/gpu).

  When using multiple critical sections on the same resources, there is no
  guarantee of exclusive access to those resources.  This behavior is disallowed
  by default (but see the kwarg `exclusive_resource_access`).

  For example, running the same function in two separate critical sections
  will not ensure serial execution:

  ```python
  v = tf.compat.v1.get_variable(\"v\", initializer=0.0, use_resource=True)
  def accumulate(up):
    x = v.read_value()
    with tf.control_dependencies([x]):
      with tf.control_dependencies([v.assign_add(up)]):
        return tf.identity(x)
  ex1 = CriticalSection().execute(
    accumulate, 1.0, exclusive_resource_access=False)
  ex2 = CriticalSection().execute(
    accumulate, 1.0, exclusive_resource_access=False)
  bad_sum = ex1 + ex2
  sess.run(v.initializer)
  sess.run(bad_sum)  # May return 0.0
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

(defn CriticalSection 
  "Critical section.

  A `CriticalSection` object is a resource in the graph which executes subgraphs
  in **serial** order.  A common example of a subgraph one may wish to run
  exclusively is the one given by the following function:

  ```python
  v = resource_variable_ops.ResourceVariable(0.0, name=\"v\")

  def count():
    value = v.read_value()
    with tf.control_dependencies([value]):
      with tf.control_dependencies([v.assign_add(1)]):
        return tf.identity(value)
  ```

  Here, a snapshot of `v` is captured in `value`; and then `v` is updated.
  The snapshot value is returned.

  If multiple workers or threads all execute `count` in parallel, there is no
  guarantee that access to the variable `v` is atomic at any point within
  any thread's calculation of `count`.  In fact, even implementing an atomic
  counter that guarantees that the user will see each value `0, 1, ...,` is
  currently impossible.

  The solution is to ensure any access to the underlying resource `v` is
  only processed through a critical section:

  ```python
  cs = CriticalSection()
  f1 = cs.execute(count)
  f2 = cs.execute(count)
  output = f1 + f2
  session.run(output)
  ```
  The functions `f1` and `f2` will be executed serially, and updates to `v`
  will be atomic.

  **NOTES**

  All resource objects, including the critical section and any captured
  variables of functions executed on that critical section, will be
  colocated to the same device (host and cpu/gpu).

  When using multiple critical sections on the same resources, there is no
  guarantee of exclusive access to those resources.  This behavior is disallowed
  by default (but see the kwarg `exclusive_resource_access`).

  For example, running the same function in two separate critical sections
  will not ensure serial execution:

  ```python
  v = tf.compat.v1.get_variable(\"v\", initializer=0.0, use_resource=True)
  def accumulate(up):
    x = v.read_value()
    with tf.control_dependencies([x]):
      with tf.control_dependencies([v.assign_add(up)]):
        return tf.identity(x)
  ex1 = CriticalSection().execute(
    accumulate, 1.0, exclusive_resource_access=False)
  ex2 = CriticalSection().execute(
    accumulate, 1.0, exclusive_resource_access=False)
  bad_sum = ex1 + ex2
  sess.run(v.initializer)
  sess.run(bad_sum)  # May return 0.0
  ```
  "
  [ name shared_name critical_section_def import_scope ]
  (py/call-attr tensorflow "CriticalSection"  name shared_name critical_section_def import_scope ))

(defn execute 
  "Execute function `fn()` inside the critical section.

    `fn` should not accept any arguments.  To add extra arguments to when
    calling `fn` in the critical section, create a lambda:

    ```python
    critical_section.execute(lambda: fn(*my_args, **my_kwargs))
    ```

    Args:
      fn: The function to execute.  Must return at least one tensor.
      exclusive_resource_access: Whether the resources required by
        `fn` should be exclusive to this `CriticalSection`.  Default: `True`.
        You may want to set this to `False` if you will be accessing a
        resource in read-only mode in two different CriticalSections.
      name: The name to use when creating the execute operation.

    Returns:
      The tensors returned from `fn()`.

    Raises:
      ValueError: If `fn` attempts to lock this `CriticalSection` in any nested
        or lazy way that may cause a deadlock.
      ValueError: If `exclusive_resource_access == True` and
        another `CriticalSection` has an execution requesting the same
        resources as `fn``.  Note, even if `exclusive_resource_access` is
        `True`, if another execution in another `CriticalSection` was created
        without `exclusive_resource_access=True`, a `ValueError` will be raised.
    "
  [self fn & {:keys [exclusive_resource_access name]
                       :or {name None}} ]
    (py/call-attr-kw self "execute" [fn] {:exclusive_resource_access exclusive_resource_access :name name }))

(defn name 
  ""
  [ self ]
    (py/call-attr self "name"))
