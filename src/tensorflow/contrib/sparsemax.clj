(ns tensorflow.contrib.sparsemax
  "Module that implements sparsemax and sparsemax loss, see [1].

[1]: https://arxiv.org/abs/1602.02068

## Sparsemax

@@sparsemax
@@sparsemax_loss
"
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce sparsemax (import-module "tensorflow.contrib.sparsemax"))

(defn sparsemax 
  "Computes sparsemax activations [1].

  For each batch `i` and class `j` we have
    $$sparsemax[i, j] = max(logits[i, j] - tau(logits[i, :]), 0)$$

  [1]: https://arxiv.org/abs/1602.02068

  Args:
    logits: A `Tensor`. Must be one of the following types: `half`, `float32`,
      `float64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `logits`.
  "
  [ logits name ]
  (py/call-attr sparsemax "sparsemax"  logits name ))

(defn sparsemax-loss 
  "Computes sparsemax loss function [1].

  [1]: https://arxiv.org/abs/1602.02068

  Args:
    logits: A `Tensor`. Must be one of the following types: `half`, `float32`,
      `float64`.
    sparsemax: A `Tensor`. Must have the same type as `logits`.
    labels: A `Tensor`. Must have the same type as `logits`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `logits`.
  "
  [ logits sparsemax labels name ]
  (py/call-attr sparsemax "sparsemax_loss"  logits sparsemax labels name ))
