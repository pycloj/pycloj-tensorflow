(ns tensorflow.python.platform.flags.FlagValues
  "Registry of 'Flag' objects.

  A 'FlagValues' can then scan command line arguments, passing flag
  arguments through to the 'Flag' objects that it owns.  It also
  provides easy access to the flag values.  Typically only one
  'FlagValues' object is needed by an application: flags.FLAGS

  This class is heavily overloaded:

  'Flag' objects are registered via __setitem__:
       FLAGS['longname'] = x   # register a new flag

  The .value attribute of the registered 'Flag' objects can be accessed
  as attributes of this 'FlagValues' object, through __getattr__.  Both
  the long and short name of the original 'Flag' objects can be used to
  access its value:
       FLAGS.longname          # parsed flag value
       FLAGS.x                 # parsed flag value (short name)

  Command line arguments are scanned and passed to the registered 'Flag'
  objects through the __call__ method.  Unparsed arguments, including
  argv[0] (e.g. the program name) are returned.
       argv = FLAGS(sys.argv)  # scan command line arguments

  The original registered Flag objects can be retrieved through the use
  of the dictionary-like operator, __getitem__:
       x = FLAGS['longname']   # access the registered Flag object

  The str() operator of a 'FlagValues' object provides help for all of
  the registered 'Flag' objects.
  "
  (:require [libpython-clj.python
             :refer [import-module
                     get-item
                     get-attr
                     python-type
                     call-attr
                     call-attr-kw]:as py]))

(py/initialize!)
(defonce flags (import-module "tensorflow.python.platform.flags"))

(defn FlagValues 
  "Registry of 'Flag' objects.

  A 'FlagValues' can then scan command line arguments, passing flag
  arguments through to the 'Flag' objects that it owns.  It also
  provides easy access to the flag values.  Typically only one
  'FlagValues' object is needed by an application: flags.FLAGS

  This class is heavily overloaded:

  'Flag' objects are registered via __setitem__:
       FLAGS['longname'] = x   # register a new flag

  The .value attribute of the registered 'Flag' objects can be accessed
  as attributes of this 'FlagValues' object, through __getattr__.  Both
  the long and short name of the original 'Flag' objects can be used to
  access its value:
       FLAGS.longname          # parsed flag value
       FLAGS.x                 # parsed flag value (short name)

  Command line arguments are scanned and passed to the registered 'Flag'
  objects through the __call__ method.  Unparsed arguments, including
  argv[0] (e.g. the program name) are returned.
       argv = FLAGS(sys.argv)  # scan command line arguments

  The original registered Flag objects can be retrieved through the use
  of the dictionary-like operator, __getitem__:
       x = FLAGS['longname']   # access the registered Flag object

  The str() operator of a 'FlagValues' object provides help for all of
  the registered 'Flag' objects.
  "
  [  ]
  (py/call-attr flags "FlagValues"  ))

(defn append-flag-values 
  "Appends flags registered in another FlagValues instance.

    Args:
      flag_values: FlagValues, the FlagValues instance from which to copy flags.
    "
  [ self flag_values ]
  (py/call-attr self "append_flag_values"  self flag_values ))

(defn append-flags-into-file 
  "Appends all flags assignments from this FlagInfo object to a file.

    Output will be in the format of a flagfile.

    NOTE: MUST mirror the behavior of the C++ AppendFlagsIntoFile
    from https://github.com/gflags/gflags.

    Args:
      filename: str, name of the file.
    "
  [ self filename ]
  (py/call-attr self "append_flags_into_file"  self filename ))

(defn find-module-defining-flag 
  "Return the name of the module defining this flag, or default.

    Args:
      flagname: str, name of the flag to lookup.
      default: Value to return if flagname is not defined. Defaults
          to None.

    Returns:
      The name of the module which registered the flag with this name.
      If no such module exists (i.e. no flag with this name exists),
      we return default.
    "
  [ self flagname default ]
  (py/call-attr self "find_module_defining_flag"  self flagname default ))

(defn find-module-id-defining-flag 
  "Return the ID of the module defining this flag, or default.

    Args:
      flagname: str, name of the flag to lookup.
      default: Value to return if flagname is not defined. Defaults
          to None.

    Returns:
      The ID of the module which registered the flag with this name.
      If no such module exists (i.e. no flag with this name exists),
      we return default.
    "
  [ self flagname default ]
  (py/call-attr self "find_module_id_defining_flag"  self flagname default ))

(defn flag-values-dict 
  "Returns a dictionary that maps flag names to flag values."
  [ self  ]
  (py/call-attr self "flag_values_dict"  self  ))

(defn flags-by-module-dict 
  "Returns the dictionary of module_name -> list of defined flags.

    Returns:
      A dictionary.  Its keys are module names (strings).  Its values
      are lists of Flag objects.
    "
  [ self  ]
  (py/call-attr self "flags_by_module_dict"  self  ))

(defn flags-by-module-id-dict 
  "Returns the dictionary of module_id -> list of defined flags.

    Returns:
      A dictionary.  Its keys are module IDs (ints).  Its values
      are lists of Flag objects.
    "
  [ self  ]
  (py/call-attr self "flags_by_module_id_dict"  self  ))

(defn flags-into-string 
  "Returns a string with the flags assignments from this FlagValues object.

    This function ignores flags whose value is None.  Each flag
    assignment is separated by a newline.

    NOTE: MUST mirror the behavior of the C++ CommandlineFlagsIntoString
    from https://github.com/gflags/gflags.

    Returns:
      str, the string with the flags assignments from this FlagValues object.
      The flags are ordered by (module_name, flag_name).
    "
  [ self  ]
  (py/call-attr self "flags_into_string"  self  ))

(defn get-flag-value 
  "Returns the value of a flag (if not None) or a default value.

    Args:
      name: str, the name of a flag.
      default: Default value to use if the flag value is None.

    Returns:
      Requested flag value or default.
    "
  [ self name default ]
  (py/call-attr self "get_flag_value"  self name default ))
(defn get-help 
  "Returns a help string for all known flags.

    Args:
      prefix: str, per-line output prefix.
      include_special_flags: bool, whether to include description of
        SPECIAL_FLAGS, i.e. --flagfile and --undefok.

    Returns:
      str, formatted help message.
    "
  [self   & {:keys [prefix include_special_flags]} ]
    (py/call-attr-kw self "get_help" [] {:prefix prefix :include_special_flags include_special_flags }))

(defn get-key-flags-for-module 
  "Returns the list of key flags for a module.

    Args:
      module: module|str, the module to get key flags from.

    Returns:
      [Flag], a new list of Flag instances.  Caller may update this list as
      desired: none of those changes will affect the internals of this
      FlagValue instance.
    "
  [ self module ]
  (py/call-attr self "get_key_flags_for_module"  self module ))

(defn is-gnu-getopt 
  ""
  [ self  ]
  (py/call-attr self "is_gnu_getopt"  self  ))

(defn is-parsed 
  "Returns whether flags were parsed."
  [ self  ]
  (py/call-attr self "is_parsed"  self  ))

(defn key-flags-by-module-dict 
  "Returns the dictionary of module_name -> list of key flags.

    Returns:
      A dictionary.  Its keys are module names (strings).  Its values
      are lists of Flag objects.
    "
  [ self  ]
  (py/call-attr self "key_flags_by_module_dict"  self  ))

(defn main-module-help 
  "Describes the key flags of the main module.

    Returns:
      str, describing the key flags of the main module.
    "
  [ self  ]
  (py/call-attr self "main_module_help"  self  ))

(defn mark-as-parsed 
  "Explicitly marks flags as parsed.

    Use this when the caller knows that this FlagValues has been parsed as if
    a __call__() invocation has happened.  This is only a public method for
    use by things like appcommands which do additional command like parsing.
    "
  [ self  ]
  (py/call-attr self "mark_as_parsed"  self  ))

(defn module-help 
  "Describes the key flags of a module.

    Args:
      module: module|str, the module to describe the key flags for.

    Returns:
      str, describing the key flags of a module.
    "
  [ self module ]
  (py/call-attr self "module_help"  self module ))
(defn read-flags-from-files 
  "Processes command line args, but also allow args to be read from file.

    Args:
      argv: [str], a list of strings, usually sys.argv[1:], which may contain
          one or more flagfile directives of the form --flagfile=\"./filename\".
          Note that the name of the program (sys.argv[0]) should be omitted.
      force_gnu: bool, if False, --flagfile parsing obeys the
          FLAGS.is_gnu_getopt() value. If True, ignore the value and always
          follow gnu_getopt semantics.

    Returns:
      A new list which has the original list combined with what we read
      from any flagfile(s).

    Raises:
      IllegalFlagValueError: Raised when --flagfile is provided with no
          argument.

    This function is called by FLAGS(argv).
    It scans the input list for a flag that looks like:
    --flagfile=<somefile>. Then it opens <somefile>, reads all valid key
    and value pairs and inserts them into the input list in exactly the
    place where the --flagfile arg is found.

    Note that your application's flags are still defined the usual way
    using absl.flags DEFINE_flag() type functions.

    Notes (assuming we're getting a commandline of some sort as our input):
    --> For duplicate flags, the last one we hit should \"win\".
    --> Since flags that appear later win, a flagfile's settings can be \"weak\"
        if the --flagfile comes at the beginning of the argument sequence,
        and it can be \"strong\" if the --flagfile comes at the end.
    --> A further \"--flagfile=<otherfile.cfg>\" CAN be nested in a flagfile.
        It will be expanded in exactly the spot where it is found.
    --> In a flagfile, a line beginning with # or // is a comment.
    --> Entirely blank lines _should_ be ignored.
    "
  [self argv  & {:keys [force_gnu]} ]
    (py/call-attr-kw self "read_flags_from_files" [argv] {:force_gnu force_gnu }))

(defn register-flag-by-module 
  "Records the module that defines a specific flag.

    We keep track of which flag is defined by which module so that we
    can later sort the flags by module.

    Args:
      module_name: str, the name of a Python module.
      flag: Flag, the Flag instance that is key to the module.
    "
  [ self module_name flag ]
  (py/call-attr self "register_flag_by_module"  self module_name flag ))

(defn register-flag-by-module-id 
  "Records the module that defines a specific flag.

    Args:
      module_id: int, the ID of the Python module.
      flag: Flag, the Flag instance that is key to the module.
    "
  [ self module_id flag ]
  (py/call-attr self "register_flag_by_module_id"  self module_id flag ))

(defn register-key-flag-for-module 
  "Specifies that a flag is a key flag for a module.

    Args:
      module_name: str, the name of a Python module.
      flag: Flag, the Flag instance that is key to the module.
    "
  [ self module_name flag ]
  (py/call-attr self "register_key_flag_for_module"  self module_name flag ))

(defn remove-flag-values 
  "Remove flags that were previously appended from another FlagValues.

    Args:
      flag_values: FlagValues, the FlagValues instance containing flags to
          remove.
    "
  [ self flag_values ]
  (py/call-attr self "remove_flag_values"  self flag_values ))

(defn set-default 
  "Changes the default value of the named flag object.

    The flag's current value is also updated if the flag is currently using
    the default value, i.e. not specified in the command line, and not set
    by FLAGS.name = value.

    Args:
      name: str, the name of the flag to modify.
      value: The new default value.

    Raises:
      UnrecognizedFlagError: Raised when there is no registered flag named name.
      IllegalFlagValueError: Raised when value is not valid.
    "
  [ self name value ]
  (py/call-attr self "set_default"  self name value ))
(defn set-gnu-getopt 
  "Sets whether or not to use GNU style scanning.

    GNU style allows mixing of flag and non-flag arguments. See
    http://docs.python.org/library/getopt.html#getopt.gnu_getopt

    Args:
      gnu_getopt: bool, whether or not to use GNU style scanning.
    "
  [self   & {:keys [gnu_getopt]} ]
    (py/call-attr-kw self "set_gnu_getopt" [] {:gnu_getopt gnu_getopt }))

(defn unparse-flags 
  "Unparses all flags to the point before any FLAGS(argv) was called."
  [ self  ]
  (py/call-attr self "unparse_flags"  self  ))

(defn write-help-in-xml-format 
  "Outputs flag documentation in XML format.

    NOTE: We use element names that are consistent with those used by
    the C++ command-line flag library, from
    https://github.com/gflags/gflags.
    We also use a few new elements (e.g., <key>), but we do not
    interfere / overlap with existing XML elements used by the C++
    library.  Please maintain this consistency.

    Args:
      outfile: File object we write to.  Default None means sys.stdout.
    "
  [ self outfile ]
  (py/call-attr self "write_help_in_xml_format"  self outfile ))
