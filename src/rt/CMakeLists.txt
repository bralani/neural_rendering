if(BRLCAD_ENABLE_OPENCL)
  add_definitions(-DUSE_OPENCL)
endif (BRLCAD_ENABLE_OPENCL)

set(LIBRTUIF_SOURCES
  do.c
  grid.c
  heatgraph.c
  main.c
  opt.c
  scanline.c
  usage.cpp
  worker.c
  )

BRLCAD_ADDLIB(librtuif "${LIBRTUIF_SOURCES}" "librt;${M_LIBRARY}" "" "${CMAKE_CURRENT_SOURCE_DIR}" STATIC NO_INSTALL)

if(BRLCAD_ENABLE_OPENCL)
  set(OPENCL_LIBS ${OPENCL_LIBRARIES})
endif(BRLCAD_ENABLE_OPENCL)

set(RT_STD_LIBS
  Threads::Threads
  )
if(MPI_C_LIBRARIES AND BRLCAD_ENABLE_MPI)
  set(RT_STD_LIBS ${RT_STD_LIBS}
    ${MPI_C_LIBRARIES}
    )
endif(MPI_C_LIBRARIES AND BRLCAD_ENABLE_MPI)
set(RT_STD_LIBS ${RT_STD_LIBS}
  librt
  libnmg
  libdm
  liboptical
  libicv
  ${M_LIBRARY}
  )

BRLCAD_ADDDATA(rtexample.c sample_applications)

BRLCAD_ADDEXEC(rtexample "rtexample.c" "librt;libdm;liboptical;libicv" NO_INSTALL)

# Check for and install the torch package
execute_process(COMMAND ${Python_EXECUTABLE} -m pip show torch RESULT_VARIABLE TORCH_FOUND)
if(NOT TORCH_FOUND)
    execute_process(COMMAND ${Python_EXECUTABLE} -m pip install torch)
endif()

# Check for and install the torchvision package
execute_process(COMMAND ${Python_EXECUTABLE} -m pip show torchvision RESULT_VARIABLE TORCHVISION_FOUND)
if(NOT TORCHVISION_FOUND)
    execute_process(COMMAND ${Python_EXECUTABLE} -m pip install torchvision)
endif()


# Pytorch
set(Torch_DIR "C:/lib/VS/libtorch/share/cmake/Torch")
set(Torch_INCLUDE_DIRS "C:/lib/VS/libtorch/include")
set(Torch_INCLUDE_LIBS "C:/lib/VS/libtorch/lib")
include_directories(${Torch_INCLUDE_DIRS})
find_package(Torch REQUIRED)
add_library(torch_runner STATIC torch_runner.cpp)

target_link_libraries(torch_runner "${TORCH_LIBRARIES}")
target_include_directories(torch_runner BEFORE PRIVATE
  ${CMAKE_SOURCE_DIR}/include/analyze
  ${CMAKE_CURRENT_SOURCE_DIR}
)
message(STATUS "Found Torch version: ${TORCH_VERSION}")
message(STATUS "Torch include directories: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET torch_runner
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:torch_runner>)
endif (MSVC)
target_include_directories(torch_runner PRIVATE
    ${CMAKE_SOURCE_DIR}/include/analyze
    ${CMAKE_CURRENT_SOURCE_DIR}
)

BRLCAD_ADDEXEC(rt_trainneural "view.c;grid.c;heatgraph.c;scanline.c;opt.c;do.c;worker.c;usage.cpp;rt_neural.cpp;rt_trainer.cpp;train_neural.cpp;neural.c;../libanalyze/util.cpp;" "Threads::Threads;librt;libdm;libnmg;libdm;liboptical;libicv;liboptical;libicv;${Python_LIBRARIES};${RT_STD_LIBS};${OPEN_CL_LIBS}" FOLDER Rt)
target_include_directories(rt_trainneural BEFORE PRIVATE
    ${CMAKE_SOURCE_DIR}/include/analyze
    ${CMAKE_CURRENT_SOURCE_DIR}
)
ADD_TARGET_DEPS(rt_trainneural dm_plugins dm-txt)
target_link_libraries(rt_trainneural torch_runner "${TORCH_LIBRARIES}" )

BRLCAD_ADDEXEC(reshoot "reshoot.c" "Threads::Threads;librt;libdm;liboptical;libicv" FOLDER Rt)
target_include_directories(reshoot BEFORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

BRLCAD_ADDEXEC(rtshot "rtshot.c" "${RT_STD_LIBS}" FOLDER Rt)
target_include_directories(rtshot BEFORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
ADD_TARGET_DEPS(rtshot dm_plugins)

BRLCAD_ADDEXEC(rtscale "read-rtlog.c;rtscale.c" "Threads::Threads;librt;libdm;liboptical;libicv;${M_LIBRARY}" FOLDER Rt)
target_include_directories(rtscale BEFORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
ADD_TARGET_DEPS(rtscale dm_plugins)

BRLCAD_ADDEXEC(rt "${LIBRTUIF_SOURCES};view.c" "${OPENCL_LIBS};${RT_STD_LIBS}" FOLDER Rt)
target_include_directories(rt BEFORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
ADD_TARGET_DEPS(rt dm_plugins dm-txt)

BRLCAD_ADDEXEC(rtarea "${LIBRTUIF_SOURCES};viewarea.c" "${RT_STD_LIBS}" FOLDER Rt)
target_include_directories(rtarea BEFORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_property(TARGET rtarea APPEND PROPERTY COMPILE_DEFINITIONS "RT_TXT_OUTPUT")

BRLCAD_ADDEXEC(rtcheck "${LIBRTUIF_SOURCES};viewcheck.c" "${RT_STD_LIBS}" FOLDER Rt)
target_include_directories(rtcheck BEFORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_property(TARGET rtcheck APPEND PROPERTY COMPILE_DEFINITIONS "RT_TXT_OUTPUT")

BRLCAD_ADDEXEC(rtedge "${LIBRTUIF_SOURCES};viewedge.c" "${RT_STD_LIBS}" FOLDER Rt)
target_include_directories(rtedge BEFORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
ADD_TARGET_DEPS(rtedge dm_plugins)

BRLCAD_ADDEXEC(rtg3 "${LIBRTUIF_SOURCES};viewg3.c" "${RT_STD_LIBS}" FOLDER Rt)
target_include_directories(rtg3 BEFORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_property(TARGET rtg3 APPEND PROPERTY COMPILE_DEFINITIONS "RT_TXT_OUTPUT")

BRLCAD_ADDEXEC(rthide "${LIBRTUIF_SOURCES};viewhide.c" "${RT_STD_LIBS}" FOLDER Rt)
target_include_directories(rthide BEFORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_property(TARGET rthide APPEND PROPERTY COMPILE_DEFINITIONS "RT_TXT_OUTPUT")

BRLCAD_ADDEXEC(rtsil "${LIBRTUIF_SOURCES};viewsil.c" "${RT_STD_LIBS}" FOLDER Rt)
target_include_directories(rtsil BEFORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
ADD_TARGET_DEPS(rtxray dm_plugins)

BRLCAD_ADDEXEC(rtsurf "rtsurf.c;rtsurf_hits.cpp" "libanalyze;librt;libdm;liboptical;libicv")
target_include_directories(rtsurf BEFORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

BRLCAD_ADDEXEC(rtweight "${LIBRTUIF_SOURCES};viewweight.c" "libanalyze;${RT_STD_LIBS}" FOLDER Rt)
target_include_directories(rtweight BEFORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
set_property(TARGET rtweight APPEND PROPERTY COMPILE_DEFINITIONS "RT_TXT_OUTPUT")

BRLCAD_ADDEXEC(rtxray "${LIBRTUIF_SOURCES};viewxray.c" "${RT_STD_LIBS}" FOLDER Rt)
target_include_directories(rtxray BEFORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
ADD_TARGET_DEPS(rtxray dm_plugins)

BRLCAD_ADDEXEC(rtdepth "${LIBRTUIF_SOURCES};viewdepth.c" "${RT_STD_LIBS}" FOLDER Rt)
target_include_directories(rtdepth BEFORE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
ADD_TARGET_DEPS(rtdepth dm_plugins)

set(rt_ignore_files
  CMakeLists.txt
  ext.h
  libfb-dummy.c
  main.c
  rtuif.h
  rtsurf_hits.h
  scanline.h
  viewdir.c
  viewdummy.c
  )
CMAKEFILES(${rt_ignore_files})

# Local Variables:
# tab-width: 8
# mode: cmake
# indent-tabs-mode: t
# End:
# ex: shiftwidth=2 tabstop=8
