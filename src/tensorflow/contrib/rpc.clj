(ns tensorflow.contrib.rpc
  "Ops and modules related to RPC.

@@rpc
@@try_rpc
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce rpc (import-module "tensorflow.contrib.rpc"))

(defn rpc 
  "TODO: add doc.

  Args:
    address: A `Tensor` of type `string`.
    method: A `Tensor` of type `string`.
    request: A `Tensor` of type `string`.
    protocol: An optional `string`. Defaults to `\"\"`.
    fail_fast: An optional `bool`. Defaults to `True`.
    timeout_in_ms: An optional `int`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  "
  [address method request & {:keys [protocol fail_fast timeout_in_ms name]
                       :or {name None}} ]
    (py/call-attr-kw rpc "rpc" [address method request] {:protocol protocol :fail_fast fail_fast :timeout_in_ms timeout_in_ms :name name }))

(defn try-rpc 
  "TODO: add doc.

  Args:
    address: A `Tensor` of type `string`.
    method: A `Tensor` of type `string`.
    request: A `Tensor` of type `string`.
    protocol: An optional `string`. Defaults to `\"\"`.
    fail_fast: An optional `bool`. Defaults to `True`.
    timeout_in_ms: An optional `int`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (response, status_code, status_message).

    response: A `Tensor` of type `string`.
    status_code: A `Tensor` of type `int32`.
    status_message: A `Tensor` of type `string`.
  "
  [address method request & {:keys [protocol fail_fast timeout_in_ms name]
                       :or {name None}} ]
    (py/call-attr-kw rpc "try_rpc" [address method request] {:protocol protocol :fail_fast fail_fast :timeout_in_ms timeout_in_ms :name name }))
