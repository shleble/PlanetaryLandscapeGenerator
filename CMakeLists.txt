cmake_minimum_required(VERSION 3.20)
project(DiplomaWork)

set(CMAKE_CXX_STANDARD 17)

# 1. Dependencies via Vcpkg
#    Vcpkg toolchain handles finding these.
find_package(TBB CONFIG REQUIRED)
find_package(ZLIB REQUIRED)
# blosc might need specific handling or be transitive, but usually OpenVDB links it.

# 2. Manual OpenVDB Configuration (since it's a local source build)
set(OPENVDB_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/openvdb")
set(OPENVDB_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/openvdb/build")

# Create an imported target for OpenVDB so usage is clean
add_library(OpenVDB::openvdb SHARED IMPORTED)

# Set the location of the .lib (Import Library) and .dll (Runtime)
set_target_properties(OpenVDB::openvdb PROPERTIES
    IMPORTED_IMPLIB "${OPENVDB_BUILD_DIR}/openvdb/openvdb/Release/openvdb.lib"
    IMPORTED_LOCATION "${OPENVDB_BUILD_DIR}/openvdb/openvdb/Release/openvdb.dll"
)

# Set include directories for OpenVDB users
target_include_directories(OpenVDB::openvdb INTERFACE
    "${OPENVDB_SOURCE_DIR}"
    "${OPENVDB_SOURCE_DIR}/openvdb"
    "${OPENVDB_BUILD_DIR}/openvdb"
    "${OPENVDB_BUILD_DIR}/openvdb/openvdb"
    "${OPENVDB_BUILD_DIR}/openvdb/openvdb/openvdb"
    "${OPENVDB_SOURCE_DIR}/ext/imath" 
)

# 3. Define the Executable
add_executable(DiplomaWork
    DiplomaWork/DiplomaWork.cpp
        Dependencies/FastNoiseLite.h
)

# 4. Link Libraries
#    Link against our manual OpenVDB target and the vcpkg dependencies
target_link_libraries(DiplomaWork PRIVATE 
    OpenVDB::openvdb
    TBB::tbb
    ZLIB::ZLIB
)

# 5. Post-Build: Copy DLLs
#    We need to copy OpenVDB DLLs. Vcpkg DLLs (TBB/Zlib) might be handled auto-magically 
#    by the VS generator + vcpkg, but let's be safe.
set(DLL_DIR "${OPENVDB_BUILD_DIR}/openvdb/openvdb/Release")
add_custom_command(TARGET DiplomaWork POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${DLL_DIR}/openvdb.dll"
    $<TARGET_FILE_DIR:DiplomaWork>
    COMMENT "Copying openvdb.dll..."
)
