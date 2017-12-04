# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

message("SGUI_SRC ${SGUI_SRC}")
message("SALGO_SRC ${SALGO_SRC}")

include_directories( 
  ${CMAKE_SOURCE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

set(SOURCE_FILES
  MainSplitterGUI.cpp
  Berkeley.cpp
  DirIO.cpp
)

set(HEADER_FILES
  MainSplitterGUI.h
  Dataset.h
  Converter.h
  Berkeley.h
  DirIO.h
)

set(UI_FORMS
  SplitterGUI.ui
)

qt5_wrap_ui(UI_HEADERS ${UI_FORMS})


add_library(SplitterGUI STATIC
  ${SOURCE_FILES}
  ${HEADER_FILES}
  ${UI_HEADERS}  
  ${WIN_RESOURCE_FILE}
)

target_link_libraries(SplitterGUI PUBLIC
  ${OpenCV_LIBS}
  SplitterALGO
  Qt5::Widgets
  Qt5::Gui
  Qt5::Core
)
