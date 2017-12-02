# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(SALGO_SRC "${SplitterALGO_SOURCE_DIR}")

include_directories( 
  ${OpenCV_INCLUDE_DIRS}
)

set(SOURCE_FILES
  Splitter.cpp
  ImageHandler.cpp
  GroundTruthParser.cpp
)

set(HEADER_FILES
  Splitter.h
  ImageHandler.h
  GroundTruthParser.h
)

add_library(SplitterALGO
  ${SOURCE_FILES}
  ${HEADER_FILES}
  ${WIN_RESOURCE_FILE}
)

target_link_libraries(SplitterALGO LINK_PUBLIC
  ${OpenCV_LIBS}
)
