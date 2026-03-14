function(generate_experiment_registry)
    set(options)
    set(one_value_args
        MANIFEST_ENTRIES_VAR
        HEADER_TEMPLATE
        SOURCE_TEMPLATE
        OUTPUT_HEADER
        OUTPUT_SOURCE
    )
    cmake_parse_arguments(REG "${options}" "${one_value_args}" "" ${ARGN})

    if(NOT REG_MANIFEST_ENTRIES_VAR)
        message(FATAL_ERROR "generate_experiment_registry: MANIFEST_ENTRIES_VAR is required.")
    endif()
    if(NOT REG_HEADER_TEMPLATE)
        message(FATAL_ERROR "generate_experiment_registry: HEADER_TEMPLATE is required.")
    endif()
    if(NOT REG_SOURCE_TEMPLATE)
        message(FATAL_ERROR "generate_experiment_registry: SOURCE_TEMPLATE is required.")
    endif()
    if(NOT REG_OUTPUT_HEADER)
        message(FATAL_ERROR "generate_experiment_registry: OUTPUT_HEADER is required.")
    endif()
    if(NOT REG_OUTPUT_SOURCE)
        message(FATAL_ERROR "generate_experiment_registry: OUTPUT_SOURCE is required.")
    endif()

    set(entries "${${REG_MANIFEST_ENTRIES_VAR}}")
    if(NOT entries)
        message(FATAL_ERROR "generate_experiment_registry: manifest is empty.")
    endif()

    set(seen_ids "")
    set(experiment_forward_decls "")
    set(experiment_descriptor_initializers "")

    foreach(entry IN LISTS entries)
        string(REPLACE "|" ";" parts "${entry}")
        list(LENGTH parts part_count)
        if(NOT part_count EQUAL 5)
            message(FATAL_ERROR "Invalid manifest entry '${entry}'. Expected 5 fields: id|display_name|category|adapter_symbol|enabled")
        endif()

        list(GET parts 0 experiment_id)
        list(GET parts 1 display_name)
        list(GET parts 2 category)
        list(GET parts 3 adapter_symbol)
        list(GET parts 4 enabled_raw)

        string(STRIP "${experiment_id}" experiment_id)
        string(STRIP "${display_name}" display_name)
        string(STRIP "${category}" category)
        string(STRIP "${adapter_symbol}" adapter_symbol)
        string(STRIP "${enabled_raw}" enabled_raw)

        if(experiment_id STREQUAL "")
            message(FATAL_ERROR "Invalid manifest entry '${entry}': experiment id cannot be empty.")
        endif()
        if(display_name STREQUAL "")
            message(FATAL_ERROR "Invalid manifest entry '${entry}': display_name cannot be empty.")
        endif()
        if(category STREQUAL "")
            message(FATAL_ERROR "Invalid manifest entry '${entry}': category cannot be empty.")
        endif()
        if(adapter_symbol STREQUAL "")
            message(FATAL_ERROR "Invalid manifest entry '${entry}': adapter_symbol cannot be empty.")
        endif()

        list(FIND seen_ids "${experiment_id}" existing_id_index)
        if(NOT existing_id_index EQUAL -1)
            message(FATAL_ERROR "Duplicate experiment id '${experiment_id}' in experiment manifest.")
        endif()
        list(APPEND seen_ids "${experiment_id}")

        string(TOUPPER "${enabled_raw}" enabled_upper)
        if(enabled_upper STREQUAL "ON" OR enabled_upper STREQUAL "TRUE" OR enabled_upper STREQUAL "1" OR enabled_upper STREQUAL "YES")
            set(enabled_literal "true")
        elseif(enabled_upper STREQUAL "OFF" OR enabled_upper STREQUAL "FALSE" OR enabled_upper STREQUAL "0" OR enabled_upper STREQUAL "NO")
            set(enabled_literal "false")
        else()
            message(FATAL_ERROR "Invalid enabled value '${enabled_raw}' for experiment id '${experiment_id}'. Use ON/OFF.")
        endif()

        string(APPEND experiment_forward_decls
            "bool ${adapter_symbol}(VulkanContext& context, const BenchmarkRunner& runner, const AppOptions& options, ExperimentRunOutput& output);\n")
        string(APPEND experiment_descriptor_initializers
            "    ExperimentDescriptor{\"${experiment_id}\", \"${display_name}\", \"${category}\", ${enabled_literal}, &${adapter_symbol}},\n")
    endforeach()

    list(LENGTH seen_ids experiment_count)
    set(EXPERIMENT_FORWARD_DECLS "${experiment_forward_decls}")
    set(EXPERIMENT_DESCRIPTOR_INITIALIZERS "${experiment_descriptor_initializers}")
    set(EXPERIMENT_COUNT "${experiment_count}")

    configure_file("${REG_HEADER_TEMPLATE}" "${REG_OUTPUT_HEADER}" @ONLY)
    configure_file("${REG_SOURCE_TEMPLATE}" "${REG_OUTPUT_SOURCE}" @ONLY)
endfunction()
