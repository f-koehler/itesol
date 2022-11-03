find_package(Doxygen REQUIRED)
set(DOXYGEN_GENERATE_XML YES)
set(DOXYGEN_EXTRACT_ALL YES)
set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doxygen)
doxygen_add_docs(
    Doxygen
    ${CMAKE_SOURCE_DIR}/include
    ALL
)

find_package(Sphinx REQUIRED)
add_custom_target(Sphinx ALL
    COMMAND
    ${SPHINX_EXECUTABLE} -b html
    -Dbreathe_projects.itesol=${CMAKE_CURRENT_BINARY_DIR}/doxygen/xml
    ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/sphinx
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating documentation with Sphinx")
add_dependencies(Sphinx Doxygen)