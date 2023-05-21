from os.path import join, dirname, realpath
from platform import architecture, machine
from ctypes import CDLL

from .. import debug

import typing
if typing.TYPE_CHECKING:
  pass


class HEBICoreLibrary(object):
  """Loads the C API functions.

  Do not use.
  """
  __slots__ = ['__core_api_version', '__core_lib', '__core_location', '__wrapper_lib', '__wrapper_location']

  def __init__(self, core_lib: CDLL, core_location: str, wrapper_lib: CDLL, wrapper_locaton: str):
    self.__core_lib = core_lib
    self.__core_location = core_location
    self.__wrapper_lib = wrapper_lib
    self.__wrapper_location = wrapper_locaton

    from .ctypes_loader import populate_hebi_funcs, populate_hebi_wrapper_funcs
    populate_hebi_funcs(core_lib)
    populate_hebi_wrapper_funcs(wrapper_lib)

    from .ctypes_func_defs import set_api_handle, set_api_wrapper_handle
    set_api_handle(core_lib)
    set_api_wrapper_handle(wrapper_lib)

    from .. import Version, min_c_api_version
    self.__core_api_version = Version(-1, -1, -1)

    from ctypes import byref, c_int32
    from .. import debug

    # Make sure to check for a compatible version of the C API first
    if hasattr(core_lib, 'hebiGetLibraryVersion'):
      c_api_version = min_c_api_version()
      c_min_major = c_api_version.major_version
      c_min_minor = c_api_version.minor_version
      c_min_patch = c_api_version.patch_version

      major_version = c_int32(0)
      minor_version = c_int32(0)
      patch_version = c_int32(0)
      core_lib.hebiGetLibraryVersion(byref(major_version), byref(minor_version), byref(patch_version))

      req_str = '{}.{}.{}'.format(c_min_major, c_min_minor, c_min_patch)
      cur_str = '{}.{}.{}'.format(major_version.value, minor_version.value, patch_version.value)
      debug.debug_log('hebiGetLibraryVersion() ==> {}'.format(cur_str))

      if major_version.value != 2:
        # Refuse to load anything which is not a 2.x.x binary
        raise RuntimeError("C API library must be a 2.x.x release (loaded version {})".format(cur_str))

      if cur_str < req_str:
        print('Warning: loaded C library may be incompatible, as it is an outdated library. ' +
              'Loaded library version is {}, required minimum is {}'.format(cur_str, req_str))

      self.__core_api_version._major_version = major_version.value
      self.__core_api_version._minor_version = minor_version.value
      self.__core_api_version._patch_version = patch_version.value

    from .ctypes_defs import HebiCommandMetadata, HebiFeedbackMetadata, HebiInfoMetadata

    command_metadata = HebiCommandMetadata()
    feedback_metadata = HebiFeedbackMetadata()
    info_metadata = HebiInfoMetadata()

    core_lib.hebiCommandGetMetadata(byref(command_metadata))
    core_lib.hebiFeedbackGetMetadata(byref(feedback_metadata))
    core_lib.hebiInfoGetMetadata(byref(info_metadata))
    wrapper_lib.hwInitialize(byref(command_metadata), byref(feedback_metadata), byref(info_metadata))

  def __del__(self):
    self.__core_lib.hebiCleanup()

  @property
  def core_lib_location(self) -> str:
    return self.__core_location

  @property
  def wrapper_lib_location(self) -> str:
    return self.__wrapper_location

  @property
  def version(self):
    from .. import Version
    maj_v = self.__core_api_version.major_version
    min_v = self.__core_api_version.minor_version
    pat_v = self.__core_api_version.patch_version
    return Version(maj_v, min_v, pat_v)


class SharedLibraryLoader(object):
  def __init__(self, library: str):
    self._candidates: 'list[str]' = list()
    self._library = library

  def add_candidate(self, loc: str):
    self._candidates.append(loc)

  @property
  def candidates(self):
    return self._candidates

  def try_load_library(self):
    """Goes through all candidate libraries in priority order.

    :return: a tuple (pair): `ctypes` object representing the library, along with the file path of the library
    """
    candidates = self.candidates
    candidate_count = len(candidates)
    debug.debug_log("Loading library {}.".format(self._library))
    debug.debug_log("Loader(library={}): {} candidate binaries:".format(self._library, candidate_count))
    debug.debug_log("Loader(library={}): Candidate binaries, in order of priority:".format(self._library))

    # Loop twice so all candidate libraries can be printed to debug stream in case of debugging enabled
    for i, candidate in enumerate(candidates):
      debug.debug_log("Candidate {}: {}".format(i+1, candidate))

    from ctypes import cdll
    for candidate in candidates:
      try:
        lib = cdll.LoadLibrary(candidate)
        debug.debug_log('Successfully loaded library at {}'.format(candidate))
        return lib, candidate
      except Exception as e:
        debug.debug_log("Attempting to load library {} raised exception:\n{}".format(candidate, e))

    debug.warn_log("Unable to load (library={}).\nCandidate libraries attempted (in order):".format(self._library))
    for i, candidate in enumerate(candidates):
      debug.warn_log("Candidate {}: {}".format(i+1, candidate))
    return None, None


def _get_library_load_candidates(library: str, readable_name: str, env_var: 'str | None' = None):
  ret: 'list[str]' = list()
  from os import environ
  import sys

  # Top priority: environment variables (if set)
  if env_var is not None:
    environ_val = environ.get(env_var)
    if environ_val is not None:
      debug.debug_log("{} library candidate '{}' from {} environment variable".format(readable_name, environ_val, env_var))
      ret.append(environ_val)

  # Lower priority: load from installed package
  lib_base_path = join(join(dirname(realpath(__file__)), '..', '..'), 'lib')

  from hebi import version
  c_api_version = version.min_c_api_version()
  maj_ver = c_api_version.major_version
  min_ver = c_api_version.minor_version

  if sys.platform.startswith('linux'):
    _find_linux_candidates(ret, lib_base_path, maj_ver, min_ver, library)
  elif sys.platform == 'darwin':
    _find_mac_candidates(ret, lib_base_path, maj_ver, library)
  elif sys.platform == 'win32':
    _find_win_candidates(ret, lib_base_path, library)

  return ret


def _load_shared_library(name: str, readable_name: str, env_var: 'str | None' = None):
  loader = SharedLibraryLoader(name)

  for entry in _get_library_load_candidates(name, readable_name, env_var):
    loader.add_candidate(entry)

  loaded_c_lib, loaded_loc = loader.try_load_library()
  if loaded_c_lib is None:
    raise RuntimeError('{} library not found'.format(readable_name))

  assert loaded_loc is not None
  return loaded_c_lib, loaded_loc


def _load_core_shared_library(): return _load_shared_library('hebi', 'HEBI Core', 'HEBI_C_LIB')
def _load_wrapper_shared_library(): return _load_shared_library('hebiWrapper', 'HEBI Core wrapper', 'HEBI_WRAPPER_LIB')


def _find_linux_candidates(output: 'list[str]', lib_base_path: str, maj_ver: int, min_ver: int, library: str):
  import re
  cpu = machine()
  py_exec_arch = architecture()[0]
  lib_str = 'lib{}.so'.format(library)

  if cpu == 'x86_64' and ('64' in py_exec_arch):
    # 64 bit x86 CPU with 64 bit python
    lib_path = join(lib_base_path, 'linux_x86_64', lib_str)

  elif ((re.match('i[3-6]86', cpu) is not None)
        or (cpu == 'x86_64') and ('32' in py_exec_arch)):
    raise RuntimeError('i686 is no longer supported. If you are on a 64 bit kernel, install and run an x86_64 instance of Python.')

  elif (re.match('arm.*', cpu) is not None) and ('32' in py_exec_arch):
    # 32 bit armhf with 32 bit python
    lib_path = join(lib_base_path, 'linux_armhf', lib_str)

  elif ((re.match('arm.*', cpu) is not None)
        or 'aarch64' in cpu and ('64' in py_exec_arch)):
    lib_path = join(lib_base_path, 'linux_aarch64', lib_str)
  else:
    raise RuntimeError('Unknown architecture {0}'.format(cpu))

  output.append('{}.{}.{}'.format(lib_path, maj_ver, min_ver))
  output.append('{}.{}'.format(lib_path, maj_ver))
  output.append(lib_path)


def _find_mac_candidates(output: 'list[str]', lib_base_path: str, maj_ver: str, library: str):
  cpu = machine()
  if cpu == 'arm64':
    output.append(join(lib_base_path, 'osx_arm64', 'lib{}.{}.dylib'.format(library, maj_ver)))
    output.append(join(lib_base_path, 'osx_arm64', 'lib{}.dylib'.format(library)))
  else: # if cpu == 'amd64' or cpu == 'x86' ?
    output.append(join(lib_base_path, 'osx_amd64', 'lib{}.{}.dylib'.format(library, maj_ver)))
    output.append(join(lib_base_path, 'osx_amd64', 'lib{}.dylib'.format(library)))


def _find_win_candidates(output: 'list[str]', lib_base_path: str, library: str):
  cpu = machine()
  py_exec_arch = architecture()[0]

  if cpu == 'AMD64' or cpu == 'x86':
    # Windows doesn't like to make it easy to detect which architecture the process is running in (x86 vs x64)
    # You can use `ctypes` to detect this, but this is a more terse way.
    output.append(join(lib_base_path, 'win_x64', '{}.dll'.format(library)))
    output.append(join(lib_base_path, 'win_x86', '{}.dll'.format(library)))
  elif cpu == 'ARM':
    # XXX Not yet supported :(
    # 32 bit ARM on Windows
    raise RuntimeError('ARM is not yet supported on Windows')
  elif cpu == 'ARM64':
    # XXX Not yet supported :(
    # 64 bit ARM on Windows
    raise RuntimeError('ARM64 is not yet supported on Windows')
  else:
    raise RuntimeError('Unknown architecture {}'.format(cpu))


def _init_libraries():
  core_lib, core_loc = _load_core_shared_library()
  wrapper_lib, wrapper_loc = _load_wrapper_shared_library()

  return HEBICoreLibrary(core_lib, core_loc, wrapper_lib, wrapper_loc)


# Load library on import
_handle = _init_libraries()
