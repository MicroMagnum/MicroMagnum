set(SRC
  Magneto.cpp
  Benchmark.cpp
  Logger.cpp
)

if(UNIX)
  set(SRC ${SRC} "os_posix.cpp")
elseif(WINDOWS)
  set(SRC ${SRC} "os_windows.cpp")
endif()

append_sources(${CMAKE_CURRENT_SOURCE_DIR} "${SRC}" "" "")
