cmake_minimum_required(VERSION 2.8.7)
project( Weazelball )

#------------------------------------------------------------------------------
## Dependencies
include( FindPkgConfig )
if( PKG_CONFIG_FOUND )
  pkg_check_modules( GAZEBO REQUIRED gazebo )
endif()
include_directories( ${GAZEBO_INCLUDE_DIRS} )
link_directories( ${GAZEBO_LIBRARY_DIRS} )

if( PKG_CONFIG_FOUND )
  pkg_check_modules( RAVELIN ravelin )
endif()
include_directories( ${RAVELIN_INCLUDE_DIRS} )
link_directories( ${RAVELIN_LIBRARY_DIRS} )

include (FindPkgConfig)
if (PKG_CONFIG_FOUND)
  pkg_check_modules(GAZEBO gazebo)
  pkg_check_modules(OGRE OGRE)
  pkg_check_modules(OGRE-Terrain OGRE-Terrain)
endif()
include_directories(${GAZEBO_INCLUDE_DIRS} ${OGRE_INCLUDE_DIRS} ${OGRE-Terrain_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS} ${OGRE_LIBRARY_DIRS})


#------------------------------------------------------------------------------
## Code
set( LIBS 
  ${GAZEBO_LIBRARIES} 
  ${RAVELIN_LIBRARIES}
  libRavelin.so
) 

set( HDRS 
  common.h
  math.h
  log.h
  state.h
  virtual_time.h
  vicon.h
  video.h
  models.h
  weazelball.h
  movie.h
  gazebo_log.h
)

set( SRCS
)

#------------------------------------------------------------------------------
## Build
#add_library( controller SHARED controller.cpp ${HDRS} ${SRCS} )
#target_link_libraries( controller ${LIBS} )

add_library( vicon-visualization-controller SHARED vicon_visualization_controller.cpp ${HDRS} ${SRCS} )
target_link_libraries( vicon-visualization-controller ${LIBS} )

add_library( synchronization-controller SHARED synchronization_controller.cpp ${HDRS} ${SRCS} )
target_link_libraries( synchronization-controller ${LIBS} )

add_library( filtering-controller SHARED filtering_controller.cpp ${HDRS} ${SRCS} )
target_link_libraries( filtering-controller ${LIBS} )

add_library( height-controller SHARED height_correction_controller.cpp ${HDRS} ${SRCS} )
target_link_libraries( height-controller ${LIBS} )

#add_library( simulation-controller SHARED simulation_controller.cpp ${HDRS} ${SRCS} )
add_library( simulation-controller SHARED randomization_controller.cpp ${HDRS} ${SRCS} )
target_link_libraries( simulation-controller ${LIBS} )


add_library( render-controller SHARED render_controller.cpp ${HDRS} ${SRCS} )
target_link_libraries( render-controller ${LIBS} )

add_library( movie-plugin SHARED camera_plugin.cpp ${HDRS} ${SRCS} )
target_link_libraries( movie-plugin ${LIBS} )

#add_library( block-on-plane-controller SHARED block_on_plane_controller.cpp ${HDRS} ${SRCS} )
#target_link_libraries( block-on-plane-controller ${LIBS} )

#add_library( movie-plugin SHARED render_plugin.cpp ${HDRS} ${SRCS} )
#target_link_libraries( movie-plugin ${LIBS} )

#add_library( postprocess-controller SHARED postprocess_controller.cpp ${HDRS} ${SRCS} )
#target_link_libraries( postprocess-controller ${LIBS} )

#add_library( world-plugin SHARED world_plugin.cpp ${HDRS} )
#target_link_libraries( world-plugin ${LIBS} )

#------------------------------------------------------------------------------
## Copy data files
file( COPY ${CMAKE_CURRENT_SOURCE_DIR}/models DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )

# designed to work in conjunction with the vicon-visualization-controller
file( COPY ${CMAKE_CURRENT_SOURCE_DIR}/visualize.world DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )

# designed to work in conjunction with the synchronization-controller
file( COPY ${CMAKE_CURRENT_SOURCE_DIR}/synchronize.world DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )

# designed to work in conjunction with the filtering-controller
file( COPY ${CMAKE_CURRENT_SOURCE_DIR}/filter.world DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )

# designed to work in conjunction with the height-controller
file( COPY ${CMAKE_CURRENT_SOURCE_DIR}/height.world DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )

# designed to work in conjunction with the postprocess-controller
file( COPY ${CMAKE_CURRENT_SOURCE_DIR}/postprocess.world DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )

# designed to work in conjunction with the simulation-controller
file( COPY ${CMAKE_CURRENT_SOURCE_DIR}/simulate.world DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
file( COPY ${CMAKE_CURRENT_SOURCE_DIR}/simulate-offset-4.world DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
file( COPY ${CMAKE_CURRENT_SOURCE_DIR}/simulate-offset-3.world DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
file( COPY ${CMAKE_CURRENT_SOURCE_DIR}/simulate-offset-2.world DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
file( COPY ${CMAKE_CURRENT_SOURCE_DIR}/simulate-offset-1.world DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
file( COPY ${CMAKE_CURRENT_SOURCE_DIR}/simulate-offset+1.world DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
file( COPY ${CMAKE_CURRENT_SOURCE_DIR}/simulate-offset+2.world DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
file( COPY ${CMAKE_CURRENT_SOURCE_DIR}/simulate-offset+3.world DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
file( COPY ${CMAKE_CURRENT_SOURCE_DIR}/simulate-offset+4.world DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )

# designed to work in conjunction with the friction-controller
file( COPY ${CMAKE_CURRENT_SOURCE_DIR}/block-on-plane.world DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )
