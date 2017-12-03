cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
cmake_policy(SET CMP0020 NEW)
cmake_policy(SET CMP0071 NEW)

# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)


project(ImageSplitterSuperpixels)

set(ISS_SRC "${ImageSplitter_SOURCE_DIR}")
set(OpenCV_DIR "D:/OpenCV/OpenCV-build" CACHE PATH "OpenCV directory override" FORCE)
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.0" CACHE PATH "Cuda v9 directory override v8" FORCE)

find_package(OpenCV REQUIRED)
find_package(Qt5Core REQUIRED)
find_package(Qt5Widgets REQUIRED)
find_package(Qt5Gui REQUIRED)

include_directories( 
  ${OpenCV_INCLUDE_DIRS}
)

add_subdirectory(SplitterGUI)
add_subdirectory(SplitterAlgo)

add_executable(ImageSplitterSuperpixels 
  SplitterMain.cpp
  ${WIN_RESOURCE_FILE}
)

target_link_libraries(ImageSplitterSuperpixels 
  ${OpenCV_LIBS}
  SplitterGUI
  SplitterALGO
  Qt5::Widgets
  Qt5::Gui
  Qt5::Core
)

add_custom_command(
  TARGET ImageSplitterSuperpixels POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:Qt5::Widgets>
	$<TARGET_FILE:Qt5::Core>
	$<TARGET_FILE:Qt5::Gui>
	$<TARGET_FILE:${OPENCV_DIR}/$<CONFIGURATION>
	$<TARGET_FILE_DIR:ImageSplitterSuperpixels>
)