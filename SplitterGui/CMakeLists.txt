# Find includes in corresponding build directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# Instruct CMake to run moc automatically when needed.
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

set(SGUI_SRC "${SplitterGUI_SOURCE_DIR}")

include_directories( 
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
  Berkeley.h
  DirIO.h
)

set(UI_FORMS
  SplitterGUI.ui
)

qt5_wrap_ui(UI_HEADERS ${UI_FORMS})


add_library(SplitterGUI
  ${SOURCE_FILES}
  ${HEADER_FILES}
  ${UI_HEADERS}  
  ${WIN_RESOURCE_FILE}
)

target_link_libraries(SplitterGUI LINK_PUBLIC
  ${OpenCV_LIBS}
  Qt5::Widgets
  Qt5::Gui
  Qt5::Core
)
