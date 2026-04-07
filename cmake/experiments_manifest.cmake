set(EXPERIMENT_MANIFEST_JSON_PATH "${CMAKE_SOURCE_DIR}/config/experiment_manifest.json")
file(READ "${EXPERIMENT_MANIFEST_JSON_PATH}" _experiment_manifest_json)

# CMake consumes the same manifest JSON as the Python tooling so new experiments
# only need one registration edit before build and script paths stay aligned.

string(JSON _experiment_count LENGTH "${_experiment_manifest_json}" experiments)
if(_experiment_count EQUAL 0)
    message(FATAL_ERROR "Experiment manifest at ${EXPERIMENT_MANIFEST_JSON_PATH} is empty.")
endif()

set(EXPERIMENT_MANIFEST_ENTRIES "")
set(EXPERIMENT_SOURCE_FILES "")
set(EXPERIMENT_ADAPTER_SOURCE_FILES "")

math(EXPR _experiment_last_index "${_experiment_count} - 1")
foreach(_index RANGE ${_experiment_last_index})
    string(JSON _experiment_id GET "${_experiment_manifest_json}" experiments ${_index} id)
    string(JSON _display_name GET "${_experiment_manifest_json}" experiments ${_index} display_name)
    string(JSON _category GET "${_experiment_manifest_json}" experiments ${_index} category)
    string(JSON _adapter_symbol GET "${_experiment_manifest_json}" experiments ${_index} adapter_symbol)
    string(JSON _enabled_value GET "${_experiment_manifest_json}" experiments ${_index} enabled)
    string(JSON _source_path GET "${_experiment_manifest_json}" experiments ${_index} source)
    string(JSON _adapter_source_path GET "${_experiment_manifest_json}" experiments ${_index} adapter_source)

    if(_experiment_id STREQUAL "" OR _display_name STREQUAL "" OR _category STREQUAL "" OR _adapter_symbol STREQUAL "")
        message(FATAL_ERROR "Experiment manifest entry ${_index} is missing required registry fields.")
    endif()
    if(_source_path STREQUAL "" OR _adapter_source_path STREQUAL "")
        message(FATAL_ERROR "Experiment manifest entry '${_experiment_id}' is missing source paths.")
    endif()

    if(_enabled_value)
        set(_enabled_literal "ON")
    else()
        set(_enabled_literal "OFF")
    endif()

    list(APPEND EXPERIMENT_MANIFEST_ENTRIES
        "${_experiment_id}|${_display_name}|${_category}|${_adapter_symbol}|${_enabled_literal}"
    )
    list(APPEND EXPERIMENT_SOURCE_FILES "${CMAKE_SOURCE_DIR}/${_source_path}")
    list(APPEND EXPERIMENT_ADAPTER_SOURCE_FILES "${CMAKE_SOURCE_DIR}/${_adapter_source_path}")
endforeach()
