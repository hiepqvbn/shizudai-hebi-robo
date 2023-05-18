from hebi._internal.utils import intern_string
from weakref import ref, ReferenceType

from ..utils import Counter

import typing
from typing import Generic, TypeVar
if typing.TYPE_CHECKING:
  from typing import Any, Callable

################################################################################
# Unmanaged class wrappers
################################################################################


class UnmanagedObject(object):
  """Base class for an object created by the HEBI C API."""

  __slots__ = ['_finalized', '_internal', '_lock', '_on_delete']

  def __init__(self, internal, on_delete=None):
    self._finalized = False
    self._internal = internal
    from threading import Lock
    self._lock = Lock()
    self._on_delete = on_delete

  @property
  def _as_parameter_(self):
    with self._lock:
      if self._finalized:
        import sys
        sys.stderr.write('Warning: Attempting to reference finalized HEBI object\n')
        raise RuntimeError('Object has already been finalized')
      return self._internal

  @property
  def finalized(self):
    with self._lock:
      return self._finalized

  def force_delete(self):
    with self._lock:
      if self._finalized:
        return
      if self._on_delete and self._internal:
        self._on_delete(self._internal)
      self._internal = None
      self._finalized = True

  def __del__(self):
    self.force_delete()


class UnmanagedSharedObject(object):
  """Base class for an object from the HEBI C API which requires reference
  counting."""

  __slots__ = ['_holder']

  def __init__(self, internal: 'Any' = None, on_delete: 'Callable[[Any], None]' = (lambda _: None), existing: 'UnmanagedSharedObject | None' = None, isdummy: bool = False):
    class Proxy(object):
      def __init__(self, internal=None, on_delete: 'Callable[[Any], None]' = (lambda _: None), proxy: 'Proxy | None' = None):
        if proxy is not None:
          with proxy._lock:
            if proxy._refcount.count == 0:
              raise RuntimeError('Object has already been finalized')
            self._lock = proxy._lock
            self._internal: 'Any' = proxy._internal
            self._on_delete = proxy._on_delete
            proxy._refcount.increment()
            self._refcount = proxy._refcount
        else:
          from threading import Lock
          self._lock = Lock()
          self._internal = internal
          self._refcount = Counter()
          self._on_delete: 'Callable[[Any], None]' = on_delete

      def __del__(self):
        with self._lock:
          if self._refcount.count > 0:
            # Object has not been explicitly deleted
            self._refcount.decrement()
          if self._refcount.count == 0 and self._on_delete is not None:
            self._on_delete(self._internal)

      def force_delete(self):
        with self._lock:
          if self._refcount.count == 0:
            # Already deleted - return
            return
          self._refcount.clear()
          self._on_delete(self._internal)
          self._internal = None

      def get(self):
        with self._lock:
          if self._refcount.count == 0:
            raise RuntimeError('Object has already been finalized')
          return self._internal

    if existing is not None:
      if not isinstance(existing, UnmanagedSharedObject):
        raise TypeError('existing parameter must be an UnmanagedSharedObject instance')
      self._holder = Proxy(proxy=existing._holder)
    else:
      self._holder = Proxy(internal=internal, on_delete=on_delete)

  @property
  def _as_parameter_(self):
    return self._holder.get()


ObjectT = TypeVar('ObjectT')


class WeakReferenceContainer(Generic[ObjectT]):
  """Small wrapper around a weak reference.

  For internal use - do not use directly.
  """

  __slots__ = ['_weak_ref']

  def _get_ref(self):
    ref = self._weak_ref()
    if ref is not None:
      return ref
    raise RuntimeError('Reference no longer valid due to finalization')

  def __init__(self, reference: 'ObjectT'):
    self._weak_ref: 'ReferenceType[ObjectT]' = ref(reference)


################################################################################
# Enum wrappers
################################################################################


class FieldDetails:
  """
  TODO: Document
  """

  __slots__ = ['_units', '_camel_case', '_pascal_case', '_snake_case']

  def __init__(self, units: 'Any', camel: str, pascal: str, snake: str):
    if units is None:
      self._units = None
    else:
      self._units = intern_string(units)

    self._camel_case = intern_string(camel)
    self._pascal_case = intern_string(pascal)
    self._snake_case = intern_string(snake)

  @property
  def aliases(self):
    return {self._snake_case, self._pascal_case, self._camel_case}

  @property
  def units(self):
    return self._units

  @property
  def is_scalar(self):
    return True

  @property
  def camel_case(self):
    return self._camel_case

  @property
  def pascal_case(self):
    return self._pascal_case

  @property
  def snake_case(self):
    return self._snake_case

  def get_field(self, message_obj):
    """Retrieve the field value(s) from the given message object.

    The message object can be a single or group message (e.g.: Feedback, GroupFeedback, Command, GroupCommand, etc)

    :param message_obj: Message object which has the field represented by `self.snake_case`

    :raises AttributeError: If the field does not exist on the provided object
    """
    return getattr(message_obj, self._snake_case)


class NumberedFieldDetails:
  __slots__ = ['_getter_factory', '_units', '_base_camel_case', '_base_pascal_case', '_base_snake_case', '_numbered_fields']

  class Subfield:
    __slots__ = ['_numbered_field_details', '_sub_field_str', '_getter_factory']

    def __init__(self, numbered_field_details: 'NumberedFieldDetails', sub_field):

      self._numbered_field_details = numbered_field_details
      self._sub_field_str = str(sub_field)
      self._getter_factory = numbered_field_details._getter_factory(sub_field)

    @property
    def aliases(self):
      return {self._numbered_field_details._base_camel_case + self._sub_field_str,
              self._numbered_field_details._base_pascal_case + self._sub_field_str,
              self._numbered_field_details._base_snake_case + self._sub_field_str}

    @property
    def units(self):
      return self._numbered_field_details.units

    def get_field(self, message_obj):
      return self._getter_factory(message_obj)

  def __init__(self, units, camel: str, pascal: str, snake: str, getter_factory, numbered_fields):
    if units is None:
      self._units = None
    else:
      self._units = intern_string(units)

    self._base_camel_case = snake
    self._base_pascal_case = pascal
    self._base_snake_case = camel
    self._getter_factory = getter_factory
    self._numbered_fields = numbered_fields

  @property
  def units(self):
    return self._units

  @property
  def is_scalar(self):
    return False

  @property
  def scalars(self):
    """Retrieve all scalar fields (in a dictionary) encapsulated in this field.

    All elements within the dictionary will be a class similar to
    :class:`FieldDetails`.
    """
    num_fields = len(self._numbered_fields)

    ret: 'dict[Any, NumberedFieldDetails.Subfield]' = dict()
    for i in range(num_fields):
      sub_field = self._numbered_fields[i]
      ret[sub_field] = self.Subfield(self, sub_field)

    return ret


class EnumTraits(object):
  """
  TODO: Document
  """

  __slots__ = ['_as_parameter_', '_name', '_value', '_type']

  def __init__(self, value: int, name: str):
    self._value = value
    self._name = intern_string(name)
    self._as_parameter_ = value
    self._type: Any = None

  @property
  def value(self):
    return self._value

  @property
  def name(self):
    return self._name

  @property
  def type(self):
    return self._type

  def __int__(self):
    return self._value

  def __hash__(self):
    return self._value + (31 * hash(self._name))

  def __eq__(self, value):
    if type(value) is int:
      return value == self._value
    elif isinstance(value, EnumTraits):
      return (value._name is self._name) and (value._value == self._value)
    elif type(value) is str:
      return intern_string(value) is self._name
    else:
      return value == self._value

  def __ne__(self, value):
    if type(value) is int:
      return value != self._value
    elif isinstance(value, EnumTraits):
      return (value._name is not self._name) or (value._value != self._value)
    elif type(value) is str:
      return value is not self._name
    else:
      return value != self._value

  def __str__(self):
    return self._name

  def __repr__(self):
    return '{} (value={}, type={})'.format(self._name, self._value, self.type)


class MessageEnumTraits(EnumTraits):
  """
  TODO: Document
  """

  __slots__ = ['__products', '__substrates', '__not_broadcastable_reason', '__field_details']

  def __init__(self, value: int, name: str, field_details: 'FieldDetails | NumberedFieldDetails | None' = None, not_bcastable_reason: 'str | None' = None, yields_dict=None):
    super(MessageEnumTraits, self).__init__(value, name)
    self.__not_broadcastable_reason = not_bcastable_reason
    self.__products = yields_dict
    self.__field_details = field_details

    if yields_dict is not None:
      substrate = dict()
      for key in yields_dict.keys():
        value = yields_dict[key].lower()
        substrate[value] = key
      self.__substrates = substrate
    else:
      self.__substrates = None

  @property
  def allow_broadcast(self):
    """Determines whether the given enum has the ability to broadcast the same
    value to all modules in a group."""
    return self.__not_broadcastable_reason is not None

  @property
  def substrates(self):
    return self.__substrates

  @property
  def products(self):
    return self.__products

  @property
  def not_broadcastable_reason(self):
    if self.allow_broadcast:
      return ''
    return self.__not_broadcastable_reason

  @property
  def field_details(self):
    return self.__field_details

  def __repr__(self):
    if self.allow_broadcast:
      return '{} (value={}, type={}), broadcastable'.format(self._name, self._value, self.type)
    return '{} (value={}, type={}), not broadcastable ({})'.format(self._name, self._value, self.type, self.__not_broadcastable_reason)


T = TypeVar('T')


class EnumType(Generic[T]):
  __slots__ = ["__name", "__values", "__int_to_value"]

  def __init__(self, name: str, values: 'list[T]'):
    self.__name = name

    int_to_value: 'dict[int, T]' = dict()

    for value in values:
      if not isinstance(value, EnumTraits):
        raise TypeError("values must all be EnumTraits")
      else:
        value._type = self
        int_to_value[int(value)] = value

    self.__int_to_value = int_to_value
    self.__values: 'list[T]' = values

  @property
  def name(self):
    return self.__name

  @property
  def values(self):
    return self.__values

  def get_enum_value_by_int(self, value: int):
    if value in self.__int_to_value:
      return self.__int_to_value[value]
    else:
      raise KeyError("'{}' is not a valid enum value".format(value))
