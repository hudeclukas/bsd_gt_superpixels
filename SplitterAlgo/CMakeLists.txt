# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories( 
  ${OpenCV_INCLUDE_DIRS}
)

set(SOURCE_FILES
  Splitter.cpp
)

set(HEADER_FILES
  Splitter.h
)

add_library(SplitterALGO STATIC
  ${SOURCE_FILES}
  ${HEADER_FILES}
  ${WIN_RESOURCE_FILE}
)

target_link_libraries(SplitterALGO LINK_PUBLIC
  ${OpenCV_LIBS}
)
