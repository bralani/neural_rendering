set(ASSETIMPORT_SRCS
  assetimport_read.cpp
  assetimport_write.cpp
  )

if (BRLCAD_COMPONENTS OR NOT EXISTS ${CMAKE_SOURCE_DIR}/src/other)
  BRLCAD_Find_Package(ASSETIMPORT)
else (BRLCAD_COMPONENTS OR NOT EXISTS ${CMAKE_SOURCE_DIR}/src/other)
  find_package(ASSETIMPORT)
endif (BRLCAD_COMPONENTS OR NOT EXISTS ${CMAKE_SOURCE_DIR}/src/other)

if(BRLCAD_ENABLE_ASSETIMPORT AND ASSETIMPORT_LIBRARY AND ASSETIMPORT_INCLUDE_DIR)
  message("assetimport plugin FOUND and ENABLED")
  include_directories(SYSTEM ${ASSETIMPORT_INCLUDE_DIR})

  gcv_plugin_library(gcv-assetimport SHARED ${ASSETIMPORT_SRCS})
  target_link_libraries(gcv-assetimport libgcv libwdb librt libbu ${ASSETIMPORT_LIBRARY})
  set_property(TARGET gcv-assetimport APPEND PROPERTY COMPILE_DEFINITIONS BRLCADBUILD HAVE_CONFIG_H)
  VALIDATE_STYLE(gcv-assetimport "${ASSETIMPORT_SRCS}")
  PLUGIN_SETUP(gcv-assetimport gcv)

endif(BRLCAD_ENABLE_ASSETIMPORT AND ASSETIMPORT_LIBRARY AND ASSETIMPORT_INCLUDE_DIR)

CMAKEFILES(
  ${ASSETIMPORT_SRCS}
  CMakeLists.txt
  )

# Local Variables:
# tab-width: 8
# mode: cmake
# indent-tabs-mode: t
# End:
# ex: shiftwidth=2 tabstop=8
