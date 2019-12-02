(ns tensorflow.contrib.copy-graph
  "Functions to copy elements between graphs.
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce copy-graph (import-module "tensorflow.contrib.copy_graph"))
(defn copy-op-to-graph 
  "Returns a copy of an operation from another Graph under a specified scope.

  Given an `Operation` `org_instance` from one `Graph`,
  initializes and returns a copy of it from another `Graph`,
  under the specified scope (default `\"\"`).

  The copying is done recursively, so any `Operation` whose output
  is required to evaluate the `org_instance`, is also copied (unless
  already done).

  Since `Variable` instances are copied separately, those required
  to evaluate `org_instance` must be provided as input.

  Args:
    org_instance: An `Operation` from some `Graph`. Could be a
      `Placeholder` as well.
    to_graph: The `Graph` to copy `org_instance` to.
    variables: An iterable of `Variable` instances to copy `org_instance` to.
    scope: A scope for the new `Variable` (default `\"\"`).

  Returns:
    The copied `Operation` from `to_graph`.

  Raises:
    TypeError: If `org_instance` is not an `Operation` or `Tensor`.
  "
  [org_instance to_graph variables  & {:keys [scope]} ]
    (py/call-attr-kw copy-graph "copy_op_to_graph" [org_instance to_graph variables] {:scope scope }))
(defn copy-variable-to-graph 
  "Given a `Variable` instance from one `Graph`, initializes and returns
  a copy of it from another `Graph`, under the specified scope
  (default `\"\"`).

  Args:
    org_instance: A `Variable` from some `Graph`.
    to_graph: The `Graph` to copy the `Variable` to.
    scope: A scope for the new `Variable` (default `\"\"`).

  Returns:
    The copied `Variable` from `to_graph`.

  Raises:
    TypeError: If `org_instance` is not a `Variable`.
  "
  [org_instance to_graph  & {:keys [scope]} ]
    (py/call-attr-kw copy-graph "copy_variable_to_graph" [org_instance to_graph] {:scope scope }))
(defn get-copied-op 
  "Given an `Operation` instance from some `Graph`, returns
  its namesake from `graph`, under the specified scope
  (default `\"\"`).

  If a copy of `org_instance` is present in `graph` under the given
  `scope`, it will be returned.

  Args:
    org_instance: An `Operation` from some `Graph`.
    graph: The `Graph` to be searched for a copr of `org_instance`.
    scope: The scope `org_instance` is present in.

  Returns:
    The `Operation` copy from `graph`.
  "
  [org_instance graph  & {:keys [scope]} ]
    (py/call-attr-kw copy-graph "get_copied_op" [org_instance graph] {:scope scope }))
