(ns tensorflow-core.contrib.autograph.ConversionOptions
  "Immutable container for global conversion flags.

  Attributes:
    recursive: bool, whether to recursively convert any user functions or
      classes that the converted function may use.
    user_requested: bool, whether the conversion was explicitly requested by
      the user, as opposed to being performed as a result of other logic. This
      value always auto-resets resets to False in child conversions.
    optional_features: Union[Feature, Set[Feature]], controls the use of
      optional features in the conversion process. See Feature for available
      options.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce autograph (import-module "tensorflow_core.contrib.autograph"))

(defn ConversionOptions 
  "Immutable container for global conversion flags.

  Attributes:
    recursive: bool, whether to recursively convert any user functions or
      classes that the converted function may use.
    user_requested: bool, whether the conversion was explicitly requested by
      the user, as opposed to being performed as a result of other logic. This
      value always auto-resets resets to False in child conversions.
    optional_features: Union[Feature, Set[Feature]], controls the use of
      optional features in the conversion process. See Feature for available
      options.
  "
  [ & {:keys [recursive user_requested internal_convert_user_code optional_features]} ]
   (py/call-attr-kw autograph "ConversionOptions" [] {:recursive recursive :user_requested user_requested :internal_convert_user_code internal_convert_user_code :optional_features optional_features }))

(defn as-tuple 
  ""
  [ self  ]
  (py/call-attr self "as_tuple"  self  ))

(defn call-options 
  "Returns the corresponding options to be used for recursive conversion."
  [ self  ]
  (py/call-attr self "call_options"  self  ))

(defn to-ast 
  "Returns a representation of this object as an AST node.

    The AST node encodes a constructor that would create an object with the
    same contents.

    Returns:
      ast.Node
    "
  [ self  ]
  (py/call-attr self "to_ast"  self  ))

(defn uses 
  ""
  [ self feature ]
  (py/call-attr self "uses"  self feature ))
