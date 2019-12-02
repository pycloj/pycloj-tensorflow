(ns tensorflow.python.platform.flags
  "Import router for absl.flags. See https://github.com/abseil/abseil-py."
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce flags (import-module "tensorflow.python.platform.flags"))

(defn DEFINE 
  "Registers a generic Flag object.

  NOTE: in the docstrings of all DEFINE* functions, \"registers\" is short
  for \"creates a new flag and registers it\".

  Auxiliary function: clients should use the specialized DEFINE_<type>
  function instead.

  Args:
    parser: ArgumentParser, used to parse the flag arguments.
    name: str, the flag name.
    default: The default value of the flag.
    help: str, the help message.
    flag_values: FlagValues, the FlagValues instance with which the flag will
        be registered. This should almost never need to be overridden.
    serializer: ArgumentSerializer, the flag serializer instance.
    module_name: str, the name of the Python module declaring this flag.
        If not provided, it will be computed using the stack trace of this call.
    **args: dict, the extra keyword args that are passed to Flag __init__.
  "
  [parser name default help & {:keys [flag_values serializer module_name]
                       :or {serializer None module_name None}} ]
    (py/call-attr-kw flags "DEFINE" [parser name default help] {:flag_values flag_values :serializer serializer :module_name module_name }))

(defn DEFINE-alias 
  "Defines an alias flag for an existing one.

  Args:
    name: str, the flag name.
    original_name: str, the original flag name.
    flag_values: FlagValues, the FlagValues instance with which the flag will
        be registered. This should almost never need to be overridden.
    module_name: A string, the name of the module that defines this flag.

  Raises:
    flags.FlagError:
      UnrecognizedFlagError: if the referenced flag doesn't exist.
      DuplicateFlagError: if the alias name has been used by some existing flag.
  "
  [name original_name & {:keys [flag_values module_name]
                       :or {module_name None}} ]
    (py/call-attr-kw flags "DEFINE_alias" [name original_name] {:flag_values flag_values :module_name module_name }))

(defn DEFINE-bool 
  "Registers a boolean flag.

  Such a boolean flag does not take an argument.  If a user wants to
  specify a false value explicitly, the long option beginning with 'no'
  must be used: i.e. --noflag

  This flag will have a value of None, True or False.  None is possible
  if default=None and the user does not specify the flag on the command
  line.

  Args:
    name: str, the flag name.
    default: bool|str|None, the default value of the flag.
    help: str, the help message.
    flag_values: FlagValues, the FlagValues instance with which the flag will
        be registered. This should almost never need to be overridden.
    module_name: str, the name of the Python module declaring this flag.
        If not provided, it will be computed using the stack trace of this call.
    **args: dict, the extra keyword args that are passed to Flag __init__.
  "
  [name default help & {:keys [flag_values module_name]
                       :or {module_name None}} ]
    (py/call-attr-kw flags "DEFINE_bool" [name default help] {:flag_values flag_values :module_name module_name }))

(defn DEFINE-boolean 
  "Registers a boolean flag.

  Such a boolean flag does not take an argument.  If a user wants to
  specify a false value explicitly, the long option beginning with 'no'
  must be used: i.e. --noflag

  This flag will have a value of None, True or False.  None is possible
  if default=None and the user does not specify the flag on the command
  line.

  Args:
    name: str, the flag name.
    default: bool|str|None, the default value of the flag.
    help: str, the help message.
    flag_values: FlagValues, the FlagValues instance with which the flag will
        be registered. This should almost never need to be overridden.
    module_name: str, the name of the Python module declaring this flag.
        If not provided, it will be computed using the stack trace of this call.
    **args: dict, the extra keyword args that are passed to Flag __init__.
  "
  [name default help & {:keys [flag_values module_name]
                       :or {module_name None}} ]
    (py/call-attr-kw flags "DEFINE_boolean" [name default help] {:flag_values flag_values :module_name module_name }))

(defn DEFINE-enum 
  "Registers a flag whose value can be any string from enum_values.

  Instead of a string enum, prefer `DEFINE_enum_class`, which allows
  defining enums from an `enum.Enum` class.

  Args:
    name: str, the flag name.
    default: str|None, the default value of the flag.
    enum_values: [str], a non-empty list of strings with the possible values for
        the flag.
    help: str, the help message.
    flag_values: FlagValues, the FlagValues instance with which the flag will
        be registered. This should almost never need to be overridden.
    module_name: str, the name of the Python module declaring this flag.
        If not provided, it will be computed using the stack trace of this call.
    **args: dict, the extra keyword args that are passed to Flag __init__.
  "
  [name default enum_values help & {:keys [flag_values module_name]
                       :or {module_name None}} ]
    (py/call-attr-kw flags "DEFINE_enum" [name default enum_values help] {:flag_values flag_values :module_name module_name }))

(defn DEFINE-enum-class 
  "Registers a flag whose value can be the name of enum members.

  Args:
    name: str, the flag name.
    default: Enum|str|None, the default value of the flag.
    enum_class: class, the Enum class with all the possible values for the flag.
    help: str, the help message.
    flag_values: FlagValues, the FlagValues instance with which the flag will
        be registered. This should almost never need to be overridden.
    module_name: str, the name of the Python module declaring this flag.
        If not provided, it will be computed using the stack trace of this call.
    **args: dict, the extra keyword args that are passed to Flag __init__.
  "
  [name default enum_class help & {:keys [flag_values module_name]
                       :or {module_name None}} ]
    (py/call-attr-kw flags "DEFINE_enum_class" [name default enum_class help] {:flag_values flag_values :module_name module_name }))

(defn DEFINE-flag 
  "Registers a 'Flag' object with a 'FlagValues' object.

  By default, the global FLAGS 'FlagValue' object is used.

  Typical users will use one of the more specialized DEFINE_xxx
  functions, such as DEFINE_string or DEFINE_integer.  But developers
  who need to create Flag objects themselves should use this function
  to register their flags.

  Args:
    flag: Flag, a flag that is key to the module.
    flag_values: FlagValues, the FlagValues instance with which the flag will
        be registered. This should almost never need to be overridden.
    module_name: str, the name of the Python module declaring this flag.
        If not provided, it will be computed using the stack trace of this call.
  "
  [flag & {:keys [flag_values module_name]
                       :or {module_name None}} ]
    (py/call-attr-kw flags "DEFINE_flag" [flag] {:flag_values flag_values :module_name module_name }))
(defn DEFINE-float 
  "Registers a flag whose value must be a float.

  If lower_bound or upper_bound are set, then this flag must be
  within the given range.

  Args:
    name: str, the flag name.
    default: float|str|None, the default value of the flag.
    help: str, the help message.
    lower_bound: float, min value of the flag.
    upper_bound: float, max value of the flag.
    flag_values: FlagValues, the FlagValues instance with which the flag will
        be registered. This should almost never need to be overridden.
    **args: dict, the extra keyword args that are passed to DEFINE.
  "
  [name default help lower_bound upper_bound  & {:keys [flag_values]} ]
    (py/call-attr-kw flags "DEFINE_float" [name default help lower_bound upper_bound] {:flag_values flag_values }))
(defn DEFINE-integer 
  "Registers a flag whose value must be an integer.

  If lower_bound, or upper_bound are set, then this flag must be
  within the given range.

  Args:
    name: str, the flag name.
    default: int|str|None, the default value of the flag.
    help: str, the help message.
    lower_bound: int, min value of the flag.
    upper_bound: int, max value of the flag.
    flag_values: FlagValues, the FlagValues instance with which the flag will
        be registered. This should almost never need to be overridden.
    **args: dict, the extra keyword args that are passed to DEFINE.
  "
  [name default help lower_bound upper_bound  & {:keys [flag_values]} ]
    (py/call-attr-kw flags "DEFINE_integer" [name default help lower_bound upper_bound] {:flag_values flag_values }))
(defn DEFINE-list 
  "Registers a flag whose value is a comma-separated list of strings.

  The flag value is parsed with a CSV parser.

  Args:
    name: str, the flag name.
    default: list|str|None, the default value of the flag.
    help: str, the help message.
    flag_values: FlagValues, the FlagValues instance with which the flag will
        be registered. This should almost never need to be overridden.
    **args: Dictionary with extra keyword args that are passed to the
        Flag __init__.
  "
  [name default help  & {:keys [flag_values]} ]
    (py/call-attr-kw flags "DEFINE_list" [name default help] {:flag_values flag_values }))

(defn DEFINE-multi 
  "Registers a generic MultiFlag that parses its args with a given parser.

  Auxiliary function.  Normal users should NOT use it directly.

  Developers who need to create their own 'Parser' classes for options
  which can appear multiple times can call this module function to
  register their flags.

  Args:
    parser: ArgumentParser, used to parse the flag arguments.
    serializer: ArgumentSerializer, the flag serializer instance.
    name: str, the flag name.
    default: Union[Iterable[T], Text, None], the default value of the flag.
        If the value is text, it will be parsed as if it was provided from
        the command line. If the value is a non-string iterable, it will be
        iterated over to create a shallow copy of the values. If it is None,
        it is left as-is.
    help: str, the help message.
    flag_values: FlagValues, the FlagValues instance with which the flag will
        be registered. This should almost never need to be overridden.
    module_name: A string, the name of the Python module declaring this flag.
        If not provided, it will be computed using the stack trace of this call.
    **args: Dictionary with extra keyword args that are passed to the
        Flag __init__.
  "
  [parser serializer name default help & {:keys [flag_values module_name]
                       :or {module_name None}} ]
    (py/call-attr-kw flags "DEFINE_multi" [parser serializer name default help] {:flag_values flag_values :module_name module_name }))
(defn DEFINE-multi-enum 
  "Registers a flag whose value can be a list strings from enum_values.

  Use the flag on the command line multiple times to place multiple
  enum values into the list.  The 'default' may be a single string
  (which will be converted into a single-element list) or a list of
  strings.

  Args:
    name: str, the flag name.
    default: Union[Iterable[Text], Text, None], the default value of the flag;
        see `DEFINE_multi`.
    enum_values: [str], a non-empty list of strings with the possible values for
        the flag.
    help: str, the help message.
    flag_values: FlagValues, the FlagValues instance with which the flag will
        be registered. This should almost never need to be overridden.
    case_sensitive: Whether or not the enum is to be case-sensitive.
    **args: Dictionary with extra keyword args that are passed to the
        Flag __init__.
  "
  [name default enum_values help  & {:keys [flag_values case_sensitive]} ]
    (py/call-attr-kw flags "DEFINE_multi_enum" [name default enum_values help] {:flag_values flag_values :case_sensitive case_sensitive }))

(defn DEFINE-multi-enum-class 
  "Registers a flag whose value can be a list of enum members.

  Use the flag on the command line multiple times to place multiple
  enum values into the list.

  Args:
    name: str, the flag name.
    default: Union[Iterable[Enum], Iterable[Text], Enum, Text, None], the
        default value of the flag; see
        `DEFINE_multi`; only differences are documented here. If the value is
        a single Enum, it is treated as a single-item list of that Enum value.
        If it is an iterable, text values within the iterable will be converted
        to the equivalent Enum objects.
    enum_class: class, the Enum class with all the possible values for the flag.
        help: str, the help message.
    flag_values: FlagValues, the FlagValues instance with which the flag will be
      registered. This should almost never need to be overridden.
    module_name: A string, the name of the Python module declaring this flag. If
      not provided, it will be computed using the stack trace of this call.
    **args: Dictionary with extra keyword args that are passed to the Flag
      __init__.
  "
  [name default enum_class help & {:keys [flag_values module_name]
                       :or {module_name None}} ]
    (py/call-attr-kw flags "DEFINE_multi_enum_class" [name default enum_class help] {:flag_values flag_values :module_name module_name }))
(defn DEFINE-multi-float 
  "Registers a flag whose value can be a list of arbitrary floats.

  Use the flag on the command line multiple times to place multiple
  float values into the list.  The 'default' may be a single float
  (which will be converted into a single-element list) or a list of
  floats.

  Args:
    name: str, the flag name.
    default: Union[Iterable[float], Text, None], the default value of the flag;
        see `DEFINE_multi`.
    help: str, the help message.
    lower_bound: float, min values of the flag.
    upper_bound: float, max values of the flag.
    flag_values: FlagValues, the FlagValues instance with which the flag will
        be registered. This should almost never need to be overridden.
    **args: Dictionary with extra keyword args that are passed to the
        Flag __init__.
  "
  [name default help lower_bound upper_bound  & {:keys [flag_values]} ]
    (py/call-attr-kw flags "DEFINE_multi_float" [name default help lower_bound upper_bound] {:flag_values flag_values }))
(defn DEFINE-multi-integer 
  "Registers a flag whose value can be a list of arbitrary integers.

  Use the flag on the command line multiple times to place multiple
  integer values into the list.  The 'default' may be a single integer
  (which will be converted into a single-element list) or a list of
  integers.

  Args:
    name: str, the flag name.
    default: Union[Iterable[int], Text, None], the default value of the flag;
        see `DEFINE_multi`.
    help: str, the help message.
    lower_bound: int, min values of the flag.
    upper_bound: int, max values of the flag.
    flag_values: FlagValues, the FlagValues instance with which the flag will
        be registered. This should almost never need to be overridden.
    **args: Dictionary with extra keyword args that are passed to the
        Flag __init__.
  "
  [name default help lower_bound upper_bound  & {:keys [flag_values]} ]
    (py/call-attr-kw flags "DEFINE_multi_integer" [name default help lower_bound upper_bound] {:flag_values flag_values }))
(defn DEFINE-multi-string 
  "Registers a flag whose value can be a list of any strings.

  Use the flag on the command line multiple times to place multiple
  string values into the list.  The 'default' may be a single string
  (which will be converted into a single-element list) or a list of
  strings.


  Args:
    name: str, the flag name.
    default: Union[Iterable[Text], Text, None], the default value of the flag;
        see `DEFINE_multi`.
    help: str, the help message.
    flag_values: FlagValues, the FlagValues instance with which the flag will
        be registered. This should almost never need to be overridden.
    **args: Dictionary with extra keyword args that are passed to the
        Flag __init__.
  "
  [name default help  & {:keys [flag_values]} ]
    (py/call-attr-kw flags "DEFINE_multi_string" [name default help] {:flag_values flag_values }))
(defn DEFINE-spaceseplist 
  "Registers a flag whose value is a whitespace-separated list of strings.

  Any whitespace can be used as a separator.

  Args:
    name: str, the flag name.
    default: list|str|None, the default value of the flag.
    help: str, the help message.
    comma_compat: bool - Whether to support comma as an additional separator.
        If false then only whitespace is supported.  This is intended only for
        backwards compatibility with flags that used to be comma-separated.
    flag_values: FlagValues, the FlagValues instance with which the flag will
        be registered. This should almost never need to be overridden.
    **args: Dictionary with extra keyword args that are passed to the
        Flag __init__.
  "
  [name default help  & {:keys [comma_compat flag_values]} ]
    (py/call-attr-kw flags "DEFINE_spaceseplist" [name default help] {:comma_compat comma_compat :flag_values flag_values }))
(defn DEFINE-string 
  "Registers a flag whose value can be any string."
  [name default help  & {:keys [flag_values]} ]
    (py/call-attr-kw flags "DEFINE_string" [name default help] {:flag_values flag_values }))
(defn adopt-module-key-flags 
  "Declares that all flags key to a module are key to the current module.

  Args:
    module: module, the module object from which all key flags will be declared
        as key flags to the current module.
    flag_values: FlagValues, the FlagValues instance in which the flags will
        be declared as key flags. This should almost never need to be
        overridden.

  Raises:
    Error: Raised when given an argument that is a module name (a string),
        instead of a module object.
  "
  [module  & {:keys [flag_values]} ]
    (py/call-attr-kw flags "adopt_module_key_flags" [module] {:flag_values flag_values }))
(defn declare-key-flag 
  "Declares one flag as key to the current module.

  Key flags are flags that are deemed really important for a module.
  They are important when listing help messages; e.g., if the
  --helpshort command-line flag is used, then only the key flags of the
  main module are listed (instead of all flags, as in the case of
  --helpfull).

  Sample usage:

    flags.declare_key_flag('flag_1')

  Args:
    flag_name: str, the name of an already declared flag.
        (Redeclaring flags as key, including flags implicitly key
        because they were declared in this module, is a no-op.)
    flag_values: FlagValues, the FlagValues instance in which the flag will
        be declared as a key flag. This should almost never need to be
        overridden.

  Raises:
    ValueError: Raised if flag_name not defined as a Python flag.
  "
  [flag_name  & {:keys [flag_values]} ]
    (py/call-attr-kw flags "declare_key_flag" [flag_name] {:flag_values flag_values }))

(defn disclaim-key-flags 
  "Declares that the current module will not define any more key flags.

  Normally, the module that calls the DEFINE_xxx functions claims the
  flag to be its key flag.  This is undesirable for modules that
  define additional DEFINE_yyy functions with its own flag parsers and
  serializers, since that module will accidentally claim flags defined
  by DEFINE_yyy as its key flags.  After calling this function, the
  module disclaims flag definitions thereafter, so the key flags will
  be correctly attributed to the caller of DEFINE_yyy.

  After calling this function, the module will not be able to define
  any more flags.  This function will affect all FlagValues objects.
  "
  [  ]
  (py/call-attr flags "disclaim_key_flags"  ))

(defn doc-to-help 
  "Takes a __doc__ string and reformats it as help."
  [ doc ]
  (py/call-attr flags "doc_to_help"  doc ))

(defn flag-dict-to-args 
  "Convert a dict of values into process call parameters.

  This method is used to convert a dictionary into a sequence of parameters
  for a binary that parses arguments using this module.

  Args:
    flag_map: dict, a mapping where the keys are flag names (strings).
        values are treated according to their type:
        * If value is None, then only the name is emitted.
        * If value is True, then only the name is emitted.
        * If value is False, then only the name prepended with 'no' is emitted.
        * If value is a string then --name=value is emitted.
        * If value is a collection, this will emit --name=value1,value2,value3.
        * Everything else is converted to string an passed as such.
  Yields:
    sequence of string suitable for a subprocess execution.
  "
  [ flag_map ]
  (py/call-attr flags "flag_dict_to_args"  flag_map ))

(defn get-help-width 
  "Returns the integer width of help lines that is used in TextWrap."
  [  ]
  (py/call-attr flags "get_help_width"  ))
(defn mark-bool-flags-as-mutual-exclusive 
  "Ensures that only one flag among flag_names is True.

  Args:
    flag_names: [str], names of the flags.
    required: bool. If true, exactly one flag must be True. Otherwise, at most
        one flag can be True, and it is valid for all flags to be False.
    flag_values: flags.FlagValues, optional FlagValues instance where the flags
        are defined.
  "
  [flag_names  & {:keys [required flag_values]} ]
    (py/call-attr-kw flags "mark_bool_flags_as_mutual_exclusive" [flag_names] {:required required :flag_values flag_values }))
(defn mark-flag-as-required 
  "Ensures that flag is not None during program execution.

  Registers a flag validator, which will follow usual validator rules.
  Important note: validator will pass for any non-None value, such as False,
  0 (zero), '' (empty string) and so on.

  It is recommended to call this method like this:

    if __name__ == '__main__':
      flags.mark_flag_as_required('your_flag_name')
      app.run()

  Because validation happens at app.run() we want to ensure required-ness
  is enforced at that time. You generally do not want to force users who import
  your code to have additional required flags for their own binaries or tests.

  Args:
    flag_name: str, name of the flag
    flag_values: flags.FlagValues, optional FlagValues instance where the flag
        is defined.
  Raises:
    AttributeError: Raised when flag_name is not registered as a valid flag
        name.
  "
  [flag_name  & {:keys [flag_values]} ]
    (py/call-attr-kw flags "mark_flag_as_required" [flag_name] {:flag_values flag_values }))
(defn mark-flags-as-mutual-exclusive 
  "Ensures that only one flag among flag_names is not None.

  Important note: This validator checks if flag values are None, and it does not
  distinguish between default and explicit values. Therefore, this validator
  does not make sense when applied to flags with default values other than None,
  including other false values (e.g. False, 0, '', []). That includes multi
  flags with a default value of [] instead of None.

  Args:
    flag_names: [str], names of the flags.
    required: bool. If true, exactly one of the flags must have a value other
        than None. Otherwise, at most one of the flags can have a value other
        than None, and it is valid for all of the flags to be None.
    flag_values: flags.FlagValues, optional FlagValues instance where the flags
        are defined.
  "
  [flag_names  & {:keys [required flag_values]} ]
    (py/call-attr-kw flags "mark_flags_as_mutual_exclusive" [flag_names] {:required required :flag_values flag_values }))
(defn mark-flags-as-required 
  "Ensures that flags are not None during program execution.

  Recommended usage:

    if __name__ == '__main__':
      flags.mark_flags_as_required(['flag1', 'flag2', 'flag3'])
      app.run()

  Args:
    flag_names: Sequence[str], names of the flags.
    flag_values: flags.FlagValues, optional FlagValues instance where the flags
        are defined.
  Raises:
    AttributeError: If any of flag name has not already been defined as a flag.
  "
  [flag_names  & {:keys [flag_values]} ]
    (py/call-attr-kw flags "mark_flags_as_required" [flag_names] {:flag_values flag_values }))
(defn multi-flags-validator 
  "A function decorator for defining a multi-flag validator.

  Registers the decorated function as a validator for flag_names, e.g.

  @flags.multi_flags_validator(['foo', 'bar'])
  def _CheckFooBar(flags_dict):
    ...

  See register_multi_flags_validator() for the specification of checker
  function.

  Args:
    flag_names: [str], a list of the flag names to be checked.
    message: str, error text to be shown to the user if checker returns False.
        If checker raises flags.ValidationError, message from the raised
        error will be shown.
    flag_values: flags.FlagValues, optional FlagValues instance to validate
        against.

  Returns:
    A function decorator that registers its function argument as a validator.

  Raises:
    AttributeError: Raised when a flag is not registered as a valid flag name.
  "
  [flag_names  & {:keys [message flag_values]} ]
    (py/call-attr-kw flags "multi_flags_validator" [flag_names] {:message message :flag_values flag_values }))
(defn register-multi-flags-validator 
  "Adds a constraint to multiple flags.

  The constraint is validated when flags are initially parsed, and after each
  change of the corresponding flag's value.

  Args:
    flag_names: [str], a list of the flag names to be checked.
    multi_flags_checker: callable, a function to validate the flag.
        input - dict, with keys() being flag_names, and value for each key
            being the value of the corresponding flag (string, boolean, etc).
        output - bool, True if validator constraint is satisfied.
            If constraint is not satisfied, it should either return False or
            raise flags.ValidationError.
    message: str, error text to be shown to the user if checker returns False.
        If checker raises flags.ValidationError, message from the raised
        error will be shown.
    flag_values: flags.FlagValues, optional FlagValues instance to validate
        against.

  Raises:
    AttributeError: Raised when a flag is not registered as a valid flag name.
  "
  [flag_names multi_flags_checker  & {:keys [message flag_values]} ]
    (py/call-attr-kw flags "register_multi_flags_validator" [flag_names multi_flags_checker] {:message message :flag_values flag_values }))
(defn register-validator 
  "Adds a constraint, which will be enforced during program execution.

  The constraint is validated when flags are initially parsed, and after each
  change of the corresponding flag's value.
  Args:
    flag_name: str, name of the flag to be checked.
    checker: callable, a function to validate the flag.
        input - A single positional argument: The value of the corresponding
            flag (string, boolean, etc.  This value will be passed to checker
            by the library).
        output - bool, True if validator constraint is satisfied.
            If constraint is not satisfied, it should either return False or
            raise flags.ValidationError(desired_error_message).
    message: str, error text to be shown to the user if checker returns False.
        If checker raises flags.ValidationError, message from the raised
        error will be shown.
    flag_values: flags.FlagValues, optional FlagValues instance to validate
        against.
  Raises:
    AttributeError: Raised when flag_name is not registered as a valid flag
        name.
  "
  [flag_name checker  & {:keys [message flag_values]} ]
    (py/call-attr-kw flags "register_validator" [flag_name checker] {:message message :flag_values flag_values }))

(defn text-wrap 
  "Wraps a given text to a maximum line length and returns it.

  It turns lines that only contain whitespace into empty lines, keeps new lines,
  and expands tabs using 4 spaces.

  Args:
    text: str, text to wrap.
    length: int, maximum length of a line, includes indentation.
        If this is None then use get_help_width()
    indent: str, indent for all but first line.
    firstline_indent: str, indent for first line; if None, fall back to indent.

  Returns:
    str, the wrapped text.

  Raises:
    ValueError: Raised if indent or firstline_indent not shorter than length.
  "
  [text length & {:keys [indent firstline_indent]
                       :or {firstline_indent None}} ]
    (py/call-attr-kw flags "text_wrap" [text length] {:indent indent :firstline_indent firstline_indent }))
(defn validator 
  "A function decorator for defining a flag validator.

  Registers the decorated function as a validator for flag_name, e.g.

  @flags.validator('foo')
  def _CheckFoo(foo):
    ...

  See register_validator() for the specification of checker function.

  Args:
    flag_name: str, name of the flag to be checked.
    message: str, error text to be shown to the user if checker returns False.
        If checker raises flags.ValidationError, message from the raised
        error will be shown.
    flag_values: flags.FlagValues, optional FlagValues instance to validate
        against.
  Returns:
    A function decorator that registers its function argument as a validator.
  Raises:
    AttributeError: Raised when flag_name is not registered as a valid flag
        name.
  "
  [flag_name  & {:keys [message flag_values]} ]
    (py/call-attr-kw flags "validator" [flag_name] {:message message :flag_values flag_values }))
