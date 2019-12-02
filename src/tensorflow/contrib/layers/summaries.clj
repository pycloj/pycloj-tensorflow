(ns tensorflow.contrib.layers.python.layers.summaries
  "Utility functions for summary creation."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce summaries (import-module "tensorflow.contrib.layers.python.layers.summaries"))

(defn summarize-activation 
  "Summarize an activation.

  This applies the given activation and adds useful summaries specific to the
  activation.

  Args:
    op: The tensor to summarize (assumed to be a layer activation).
  Returns:
    The summary op created to summarize `op`.
  "
  [ op ]
  (py/call-attr summaries "summarize_activation"  op ))
(defn summarize-activations 
  "Summarize activations, using `summarize_activation` to summarize."
  [name_filter  & {:keys [summarizer]} ]
    (py/call-attr-kw summaries "summarize_activations" [name_filter] {:summarizer summarizer }))
(defn summarize-collection 
  "Summarize a graph collection of tensors, possibly filtered by name."
  [collection name_filter  & {:keys [summarizer]} ]
    (py/call-attr-kw summaries "summarize_collection" [collection name_filter] {:summarizer summarizer }))

(defn summarize-tensor 
  "Summarize a tensor using a suitable summary type.

  This function adds a summary op for `tensor`. The type of summary depends on
  the shape of `tensor`. For scalars, a `scalar_summary` is created, for all
  other tensors, `histogram_summary` is used.

  Args:
    tensor: The tensor to summarize
    tag: The tag to use, if None then use tensor's op's name.

  Returns:
    The summary op created or None for string tensors.
  "
  [ tensor tag ]
  (py/call-attr summaries "summarize_tensor"  tensor tag ))
(defn summarize-tensors 
  "Summarize a set of tensors."
  [tensors  & {:keys [summarizer]} ]
    (py/call-attr-kw summaries "summarize_tensors" [tensors] {:summarizer summarizer }))
