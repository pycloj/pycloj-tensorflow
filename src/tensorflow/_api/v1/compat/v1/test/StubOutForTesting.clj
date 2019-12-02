(ns tensorflow.-api.v1.compat.v1.test.StubOutForTesting
  "Support class for stubbing methods out for unit testing.

  Sample Usage:

  You want os.path.exists() to always return true during testing.

     stubs = StubOutForTesting()
     stubs.Set(os.path, 'exists', lambda x: 1)
       ...
     stubs.CleanUp()

  The above changes os.path.exists into a lambda that returns 1.  Once
  the ... part of the code finishes, the CleanUp() looks up the old
  value of os.path.exists and restores it.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce test (import-module "tensorflow._api.v1.compat.v1.test"))

(defn StubOutForTesting 
  "Support class for stubbing methods out for unit testing.

  Sample Usage:

  You want os.path.exists() to always return true during testing.

     stubs = StubOutForTesting()
     stubs.Set(os.path, 'exists', lambda x: 1)
       ...
     stubs.CleanUp()

  The above changes os.path.exists into a lambda that returns 1.  Once
  the ... part of the code finishes, the CleanUp() looks up the old
  value of os.path.exists and restores it.
  "
  [  ]
  (py/call-attr test "StubOutForTesting"  ))

(defn CleanUp 
  "Undoes all SmartSet() & Set() calls, restoring original definitions."
  [ self  ]
  (py/call-attr self "CleanUp"  self  ))

(defn Set 
  "In parent, replace child_name's old definition with new_child.

    The parent could be a module when the child is a function at
    module scope.  Or the parent could be a class when a class' method
    is being replaced.  The named child is set to new_child, while the
    prior definition is saved away for later, when UnsetAll() is
    called.

    This method supports the case where child_name is a staticmethod or a
    classmethod of parent.

    Args:
      parent: The context in which the attribute child_name is to be changed.
      child_name: The name of the attribute to change.
      new_child: The new value of the attribute.
    "
  [ self parent child_name new_child ]
  (py/call-attr self "Set"  self parent child_name new_child ))

(defn SmartSet 
  "Replace obj.attr_name with new_attr.

    This method is smart and works at the module, class, and instance level
    while preserving proper inheritance. It will not stub out C types however
    unless that has been explicitly allowed by the type.

    This method supports the case where attr_name is a staticmethod or a
    classmethod of obj.

    Notes:
      - If obj is an instance, then it is its class that will actually be
        stubbed. Note that the method Set() does not do that: if obj is
        an instance, it (and not its class) will be stubbed.
      - The stubbing is using the builtin getattr and setattr. So, the __get__
        and __set__ will be called when stubbing (TODO: A better idea would
        probably be to manipulate obj.__dict__ instead of getattr() and
        setattr()).

    Args:
      obj: The object whose attributes we want to modify.
      attr_name: The name of the attribute to modify.
      new_attr: The new value for the attribute.

    Raises:
      AttributeError: If the attribute cannot be found.
    "
  [ self obj attr_name new_attr ]
  (py/call-attr self "SmartSet"  self obj attr_name new_attr ))

(defn SmartUnsetAll 
  "Reverses SmartSet() calls, restoring things to original definitions.

    This method is automatically called when the StubOutForTesting()
    object is deleted; there is no need to call it explicitly.

    It is okay to call SmartUnsetAll() repeatedly, as later calls have
    no effect if no SmartSet() calls have been made.
    "
  [ self  ]
  (py/call-attr self "SmartUnsetAll"  self  ))

(defn UnsetAll 
  "Reverses Set() calls, restoring things to their original definitions.

    This method is automatically called when the StubOutForTesting()
    object is deleted; there is no need to call it explicitly.

    It is okay to call UnsetAll() repeatedly, as later calls have no
    effect if no Set() calls have been made.
    "
  [ self  ]
  (py/call-attr self "UnsetAll"  self  ))
