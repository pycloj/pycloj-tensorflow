(ns tensorflow.contrib.model-pruning.Pruning
  ""
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce model-pruning (import-module "tensorflow.contrib.model_pruning"))

(defn Pruning 
  ""
  [ spec global_step sparsity ]
  (py/call-attr model-pruning "Pruning"  spec global_step sparsity ))

(defn add-pruning-summaries 
  "Adds summaries of weight sparsities and thresholds."
  [ self  ]
  (py/call-attr self "add_pruning_summaries"  self  ))

(defn conditional-mask-update-op 
  ""
  [ self  ]
  (py/call-attr self "conditional_mask_update_op"  self  ))

(defn mask-update-op 
  ""
  [ self  ]
  (py/call-attr self "mask_update_op"  self  ))

(defn print-hparams 
  ""
  [ self  ]
  (py/call-attr self "print_hparams"  self  ))
