# - Find LIBUSB Installation
# This module tries to find the libftdi installation on your system.
# Once done this will define
#
# LIBUSB_FOUND - system has ftdi
# LIBUSB_INCLUDE_DIR - the ftdi include directory
# LIBUSB_LIBRARY - link these to use ftdi

find_package(PkgConfig)
pkg_check_modules(LIBUSB REQUIRED libusb-1.0)

set(LIBUSB_INCLUDE_DIR ${LIBUSB_INCLUDE_DIRS})
set(LIBUSB_LIBRARY ${LIBUSB_LIBRARIES})


IF (LIBUSB_LIBRARY)
  IF (LIBUSB_INCLUDE_DIR)
    set(LIBUSB_FOUND TRUE)
    MESSAGE(STATUS "Found libUSB: ${LIBUSB_INCLUDE_DIR}, ${LIBUSB_LIBRARY}")
  ELSE (LIBUSB_INCLUDE_DIR)
    set(LIBUSB_FOUND FALSE)
    MESSAGE(STATUS "libUSB headers NOT FOUND. Make sure to install the development headers!")
  ENDIF (LIBUSB_INCLUDE_DIR)
ELSE (LIBUSB_LIBRARY)
  set(LIBUSB_FOUND FALSE)
  MESSAGE(STATUS "libUSB NOT FOUND.")
ENDIF (LIBUSB_LIBRARY)

set(LIBUSB_INCLUDE_DIR ${LIBUSB_INCLUDE_DIR})
