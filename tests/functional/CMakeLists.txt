if(NOT APPLE)
    message(STATUS "Skipping functional tests: Apple platform required")
    return()
endif()

add_executable(cumetal_runtime_vector_add_test
    runtime_vector_add_test.cpp
)

add_executable(cumetal_runtime_matrix_mul_test
    runtime_matrix_mul_test.cpp
)

add_executable(cumetal_runtime_matrix_mul_tiled_test
    runtime_matrix_mul_tiled_test.cpp
)

add_executable(cumetal_runtime_printf_test
    runtime_printf_test.cpp
)

add_executable(cumetal_runtime_dynamic_shared_test
    runtime_dynamic_shared_test.cpp
)

add_executable(cumetal_runtime_struct_arg_test
    runtime_struct_arg_test.cpp
)

add_executable(cumetal_runtime_barrier_order_test
    runtime_barrier_order_test.cpp
)

add_executable(cumetal_runtime_cp_async_emul_test
    runtime_cp_async_emul_test.cpp
)

add_executable(cumetal_runtime_warp_partial_mask_test
    runtime_warp_partial_mask_test.cpp
)

add_executable(cumetal_runtime_warp_size_lane_test
    runtime_warp_size_lane_test.cpp
)

add_executable(cumetal_runtime_stream_vector_add_test
    runtime_stream_vector_add_test.cpp
)

add_executable(cumetal_runtime_null_stream_sync_test
    runtime_null_stream_sync_test.cpp
)

add_executable(cumetal_runtime_stream_per_thread_test
    runtime_stream_per_thread_test.cpp
)

add_executable(cumetal_runtime_async_memops_test
    runtime_async_memops_test.cpp
)

add_executable(cumetal_runtime_event_test
    runtime_event_test.cpp
)

add_executable(cumetal_runtime_stream_wait_event_test
    runtime_stream_wait_event_test.cpp
)

add_executable(cumetal_runtime_stream_query_test
    runtime_stream_query_test.cpp
)

add_executable(cumetal_runtime_memcpy_kind_test
    runtime_memcpy_kind_test.cpp
)

add_executable(cumetal_runtime_memcpy2d_test
    runtime_memcpy2d_test.cpp
)

add_executable(cumetal_runtime_symbol_memcpy_test
    runtime_symbol_memcpy_test.cpp
)

add_executable(cumetal_runtime_mem_get_info_test
    runtime_mem_get_info_test.cpp
)

add_executable(cumetal_runtime_host_alloc_test
    runtime_host_alloc_test.cpp
)

add_executable(cumetal_runtime_host_pointer_api_test
    runtime_host_pointer_api_test.cpp
)

add_executable(cumetal_runtime_stream_flags_test
    runtime_stream_flags_test.cpp
)

add_executable(cumetal_runtime_stream_callback_test
    runtime_stream_callback_test.cpp
)

add_executable(cumetal_runtime_device_api_test
    runtime_device_api_test.cpp
)

add_executable(cumetal_runtime_device_properties_test
    runtime_device_properties_test.cpp
)

add_executable(cumetal_runtime_device_attribute_test
    runtime_device_attribute_test.cpp
)

add_executable(cumetal_runtime_device_reset_test
    runtime_device_reset_test.cpp
)

add_executable(cumetal_runtime_device_flags_test
    runtime_device_flags_test.cpp
)

add_executable(cumetal_runtime_error_api_test
    runtime_error_api_test.cpp
)

add_executable(cumetal_runtime_profiler_api_test
    runtime_profiler_api_test.cpp
)

add_executable(cumetal_curand_uniform_test
    curand_uniform_test.cpp
)

add_executable(cumetal_cublas_api_test
    cublas_api_test.cpp
)

add_executable(cumetal_cufft_c2c_test
    cufft_c2c_test.cpp
)

add_executable(cumetal_runtime_occupancy_test
    runtime_occupancy_test.cpp
)

add_executable(cumetal_runtime_device_limits_test
    runtime_device_limits_test.cpp
)

add_executable(cumetal_driver_extended_api_test
    driver_extended_api_test.cpp
)

add_executable(cumetal_driver_vector_add_test
    driver_vector_add_test.cpp
)

add_executable(cumetal_driver_null_stream_sync_test
    driver_null_stream_sync_test.cpp
)

add_executable(cumetal_driver_matrix_mul_test
    driver_matrix_mul_test.cpp
)

add_executable(cumetal_driver_device_api_test
    driver_device_api_test.cpp
)

add_executable(cumetal_driver_error_api_test
    driver_error_api_test.cpp
)

add_executable(cumetal_driver_profiler_api_test
    driver_profiler_api_test.cpp
)

add_executable(cumetal_driver_device_query_test
    driver_device_query_test.cpp
)

add_executable(cumetal_driver_device_attribute_test
    driver_device_attribute_test.cpp
)

add_executable(cumetal_driver_stream_flags_test
    driver_stream_flags_test.cpp
)

add_executable(cumetal_driver_stream_per_thread_test
    driver_stream_per_thread_test.cpp
)

add_executable(cumetal_driver_stream_callback_test
    driver_stream_callback_test.cpp
)

add_executable(cumetal_driver_context_switch_test
    driver_context_switch_test.cpp
)

add_executable(cumetal_driver_context_requirements_test
    driver_context_requirements_test.cpp
)

add_executable(cumetal_driver_async_memcpy_test
    driver_async_memcpy_test.cpp
)

add_executable(cumetal_driver_memset_test
    driver_memset_test.cpp
)

add_executable(cumetal_driver_mem_get_info_test
    driver_mem_get_info_test.cpp
)

add_executable(cumetal_driver_mem_alloc_managed_test
    driver_mem_alloc_managed_test.cpp
)

add_executable(cumetal_driver_host_alloc_test
    driver_host_alloc_test.cpp
)

add_executable(cumetal_driver_host_pointer_api_test
    driver_host_pointer_api_test.cpp
)

add_executable(cumetal_driver_module_load_data_test
    driver_module_load_data_test.cpp
)

add_executable(cumetal_driver_module_load_data_ptx_test
    driver_module_load_data_ptx_test.cpp
)

add_executable(cumetal_driver_launch_extra_test
    driver_launch_extra_test.cpp
)

add_executable(cumetal_driver_launch_extra_scalar_test
    driver_launch_extra_scalar_test.cpp
)

add_executable(cumetal_driver_stream_wait_event_test
    driver_stream_wait_event_test.cpp
)

add_executable(cumetal_driver_launch_host_func_test
    driver_launch_host_func_test.cpp
)

add_executable(cumetal_runtime_device_attribute_ext_test
    runtime_device_attribute_ext_test.cpp
)

add_executable(cumetal_runtime_device_properties_ext_test
    runtime_device_properties_ext_test.cpp
)

add_executable(cumetal_runtime_malloc_pitch_test
    runtime_malloc_pitch_test.cpp
)

add_executable(cumetal_runtime_heap_auto_threshold_test
    runtime_heap_auto_threshold_test.cpp
)

add_executable(cumetal_runtime_peer_access_test
    runtime_peer_access_test.cpp
)

add_executable(cumetal_runtime_stream_priority_test
    runtime_stream_priority_test.cpp
)

add_executable(cumetal_runtime_cooperative_groups_grid_test
    runtime_cooperative_groups_grid_test.cpp
)

add_executable(cumetal_runtime_axpy_offset_test
    runtime_axpy_offset_test.cpp
)

add_executable(cumetal_runtime_atomic_test
    runtime_atomic_test.cpp
)

add_executable(cumetal_runtime_atomic_shared_test
    runtime_atomic_shared_test.cpp
)

add_executable(cumetal_runtime_warp_shuffle_test
    runtime_warp_shuffle_test.cpp
)

add_executable(cumetal_runtime_warp_vote_test
    runtime_warp_vote_test.cpp
)

add_executable(cumetal_runtime_fp16_ops_test
    runtime_fp16_ops_test.cpp
)

add_executable(cumetal_runtime_shared_reduce_test
    runtime_shared_reduce_test.cpp
)

add_executable(cumetal_runtime_grid_2d_test
    runtime_grid_2d_test.cpp
)

add_executable(cumetal_runtime_grid_3d_test
    runtime_grid_3d_test.cpp
)

add_executable(cumetal_runtime_fp64_ops_test
    runtime_fp64_ops_test.cpp
)

add_executable(cumetal_runtime_cooperative_launch_test
    runtime_cooperative_launch_test.cpp
)

if(CUMETAL_ENABLE_BINARY_SHIM)
    add_executable(cumetal_runtime_registration_path_test
        runtime_registration_path_test.cpp
    )

    add_executable(cumetal_runtime_call_config_registration_test
        runtime_call_config_registration_test.cpp
    )

    add_executable(cumetal_runtime_registration_fatbin_ptx_test
        runtime_registration_fatbin_ptx_test.cpp
    )

    add_executable(cumetal_runtime_legacy_launch_registration_test
        runtime_legacy_launch_registration_test.cpp
    )

    add_executable(cumetal_runtime_registration_fatbinary2_symbols_test
        runtime_registration_fatbinary2_symbols_test.cpp
    )

    add_executable(cumetal_runtime_registration_var_symbol_test
        runtime_registration_var_symbol_test.cpp
    )

    add_executable(cumetal_runtime_registration_printf_test
        runtime_registration_printf_test.cpp
    )
endif()

target_link_libraries(cumetal_runtime_vector_add_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_matrix_mul_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_matrix_mul_tiled_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_printf_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_dynamic_shared_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_struct_arg_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_barrier_order_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_cp_async_emul_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_warp_partial_mask_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_warp_size_lane_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_stream_vector_add_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_null_stream_sync_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_stream_per_thread_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_async_memops_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_event_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_stream_wait_event_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_stream_query_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_memcpy_kind_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_memcpy2d_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_symbol_memcpy_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_mem_get_info_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_host_alloc_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_host_pointer_api_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_stream_flags_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_stream_callback_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_device_api_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_device_properties_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_device_attribute_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_device_reset_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_device_flags_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_error_api_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_profiler_api_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_curand_uniform_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_cublas_api_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_cufft_c2c_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_occupancy_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_device_limits_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_extended_api_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_vector_add_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_null_stream_sync_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_matrix_mul_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_device_api_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_error_api_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_profiler_api_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_device_query_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_device_attribute_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_stream_flags_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_stream_per_thread_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_stream_callback_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_context_switch_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_context_requirements_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_async_memcpy_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_memset_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_mem_get_info_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_mem_alloc_managed_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_host_alloc_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_host_pointer_api_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_module_load_data_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_module_load_data_ptx_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_launch_extra_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_launch_extra_scalar_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_stream_wait_event_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_driver_launch_host_func_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_device_attribute_ext_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_device_properties_ext_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_malloc_pitch_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_heap_auto_threshold_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_peer_access_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_stream_priority_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_cooperative_groups_grid_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_axpy_offset_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_atomic_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_atomic_shared_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_warp_shuffle_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_warp_vote_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_fp16_ops_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_shared_reduce_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_grid_2d_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_grid_3d_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_fp64_ops_test
    PRIVATE
        cumetal_runtime
)

target_link_libraries(cumetal_runtime_cooperative_launch_test
    PRIVATE
        cumetal_runtime
)

if(CUMETAL_ENABLE_BINARY_SHIM)
    target_link_libraries(cumetal_runtime_registration_path_test
        PRIVATE
            cumetal_runtime
    )

    target_link_libraries(cumetal_runtime_call_config_registration_test
        PRIVATE
            cumetal_runtime
    )

    target_link_libraries(cumetal_runtime_registration_fatbin_ptx_test
        PRIVATE
            cumetal_runtime
    )

    target_link_libraries(cumetal_runtime_legacy_launch_registration_test
        PRIVATE
            cumetal_runtime
    )

    target_link_libraries(cumetal_runtime_registration_fatbinary2_symbols_test
        PRIVATE
            cumetal_runtime
    )

    target_link_libraries(cumetal_runtime_registration_var_symbol_test
        PRIVATE
            cumetal_runtime
    )

    target_link_libraries(cumetal_runtime_registration_printf_test
        PRIVATE
            cumetal_runtime
    )
endif()

target_compile_options(cumetal_runtime_vector_add_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_matrix_mul_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_matrix_mul_tiled_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_printf_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_dynamic_shared_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_struct_arg_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_barrier_order_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_cp_async_emul_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_warp_partial_mask_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_warp_size_lane_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_stream_vector_add_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_null_stream_sync_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_stream_per_thread_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_async_memops_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_event_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_stream_wait_event_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_stream_query_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_memcpy_kind_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_memcpy2d_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_symbol_memcpy_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_mem_get_info_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_host_alloc_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_host_pointer_api_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_stream_flags_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_stream_callback_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_device_api_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_device_properties_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_device_attribute_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_device_reset_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_device_flags_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_error_api_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_profiler_api_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_curand_uniform_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_cublas_api_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_cufft_c2c_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_occupancy_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_device_limits_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_extended_api_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_vector_add_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_null_stream_sync_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_matrix_mul_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_device_api_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_error_api_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_profiler_api_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_device_query_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_device_attribute_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_stream_flags_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_stream_per_thread_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_stream_callback_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_context_switch_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_context_requirements_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_async_memcpy_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_memset_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_mem_get_info_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_mem_alloc_managed_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_host_alloc_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_host_pointer_api_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_module_load_data_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_module_load_data_ptx_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_launch_extra_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_launch_extra_scalar_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_stream_wait_event_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_driver_launch_host_func_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_device_attribute_ext_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_device_properties_ext_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_malloc_pitch_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_heap_auto_threshold_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_peer_access_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_stream_priority_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_cooperative_groups_grid_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_axpy_offset_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_atomic_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_atomic_shared_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_warp_shuffle_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_warp_vote_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_fp16_ops_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_shared_reduce_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_grid_2d_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_grid_3d_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_fp64_ops_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

target_compile_options(cumetal_runtime_cooperative_launch_test
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

if(CUMETAL_ENABLE_BINARY_SHIM)
    target_compile_options(cumetal_runtime_registration_path_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )

    target_compile_options(cumetal_runtime_call_config_registration_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )

    target_compile_options(cumetal_runtime_registration_fatbin_ptx_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )

    target_compile_options(cumetal_runtime_legacy_launch_registration_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )

    target_compile_options(cumetal_runtime_registration_fatbinary2_symbols_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )

    target_compile_options(cumetal_runtime_registration_var_symbol_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )

    target_compile_options(cumetal_runtime_registration_printf_test
        PRIVATE
            -Wall
            -Wextra
            -Wpedantic
    )
endif()

add_test(
    NAME functional_runtime_vector_add
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_runtime_vector_add_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_runtime_vector_add PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_vector_add_heap_alloc
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_runtime_vector_add_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(
    functional_runtime_vector_add_heap_alloc
    PROPERTIES
        SKIP_RETURN_CODE 77
        ENVIRONMENT "CUMETAL_MTLHEAP_ALLOC=1;CUMETAL_MTLHEAP_CHUNK_BYTES=1048576"
)

add_test(
    NAME functional_runtime_vector_add_cu
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_cumetalc_cu_runtime_vector_add.sh
            $<TARGET_FILE:cumetalc>
            $<TARGET_FILE:cumetal_runtime_vector_add_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/vector_add.cu
            ${CMAKE_CURRENT_BINARY_DIR}/vector_add.from_cu.xcrun.metallib
)

set_tests_properties(functional_runtime_vector_add_cu PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_sample_vector_add
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_samples_vector_add.sh
            $<TARGET_FILE:cumetalc>
            ${CMAKE_SOURCE_DIR}/samples/vectorAdd/vectorAdd.cu
            ${CMAKE_SOURCE_DIR}/samples/vectorAdd/vectorAdd.cpp
            ${CMAKE_SOURCE_DIR}/runtime/api
            $<TARGET_FILE_DIR:cumetal_runtime>
            ${CMAKE_CURRENT_BINARY_DIR}/sample_vector_add.from_cu.xcrun.metallib
            ${CMAKE_CURRENT_BINARY_DIR}/sample_vector_add
)

set_tests_properties(functional_sample_vector_add PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_matrix_mul
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_axpy.sh
            $<TARGET_FILE:cumetal_runtime_matrix_mul_test>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/matrix_mul.metal
            ${CMAKE_CURRENT_BINARY_DIR}/matrix_mul.metallib
)

set_tests_properties(functional_runtime_matrix_mul PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_matrix_mul_tiled
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_axpy.sh
            $<TARGET_FILE:cumetal_runtime_matrix_mul_tiled_test>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/matrix_mul_tiled.metal
            ${CMAKE_CURRENT_BINARY_DIR}/matrix_mul_tiled.metallib
)

set_tests_properties(functional_runtime_matrix_mul_tiled PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_printf
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_axpy.sh
            $<TARGET_FILE:cumetal_runtime_printf_test>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/printf_kernel.metal
            ${CMAKE_CURRENT_BINARY_DIR}/printf_kernel.metallib
)

set_tests_properties(functional_runtime_printf PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_dynamic_shared
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_axpy.sh
            $<TARGET_FILE:cumetal_runtime_dynamic_shared_test>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/dynamic_shared_reduce.metal
            ${CMAKE_CURRENT_BINARY_DIR}/dynamic_shared_reduce.metallib
)

set_tests_properties(functional_runtime_dynamic_shared PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_struct_arg
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_axpy.sh
            $<TARGET_FILE:cumetal_runtime_struct_arg_test>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/struct_arg.metal
            ${CMAKE_CURRENT_BINARY_DIR}/struct_arg.metallib
)

set_tests_properties(functional_runtime_struct_arg PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_barrier_order
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_axpy.sh
            $<TARGET_FILE:cumetal_runtime_barrier_order_test>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/barrier_order.metal
            ${CMAKE_CURRENT_BINARY_DIR}/barrier_order.metallib
)

set_tests_properties(functional_runtime_barrier_order PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_cp_async_emul
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_axpy.sh
            $<TARGET_FILE:cumetal_runtime_cp_async_emul_test>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/cp_async_emul.metal
            ${CMAKE_CURRENT_BINARY_DIR}/cp_async_emul.metallib
)

set_tests_properties(functional_runtime_cp_async_emul PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_warp_partial_mask
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_axpy.sh
            $<TARGET_FILE:cumetal_runtime_warp_partial_mask_test>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/warp_partial_mask.metal
            ${CMAKE_CURRENT_BINARY_DIR}/warp_partial_mask.metallib
)

set_tests_properties(functional_runtime_warp_partial_mask PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_warp_size_lane
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_axpy.sh
            $<TARGET_FILE:cumetal_runtime_warp_size_lane_test>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/warp_size_lane.metal
            ${CMAKE_CURRENT_BINARY_DIR}/warp_size_lane.metallib
)

set_tests_properties(functional_runtime_warp_size_lane PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_stream_vector_add
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_runtime_stream_vector_add_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_runtime_stream_vector_add PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_null_stream_sync
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_runtime_null_stream_sync_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_runtime_null_stream_sync PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_stream_per_thread
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_runtime_stream_per_thread_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_runtime_stream_per_thread PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_async_memops
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_runtime_async_memops_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_runtime_async_memops PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_event
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_runtime_event_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_runtime_event PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_stream_wait_event
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_runtime_stream_wait_event_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_runtime_stream_wait_event PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_stream_query
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_runtime_stream_query_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_runtime_stream_query PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_memcpy_kind
    COMMAND cumetal_runtime_memcpy_kind_test
)

add_test(
    NAME functional_runtime_memcpy2d
    COMMAND cumetal_runtime_memcpy2d_test
)


add_test(
    NAME functional_runtime_symbol_memcpy
    COMMAND cumetal_runtime_symbol_memcpy_test
)

add_test(
    NAME functional_runtime_mem_get_info
    COMMAND cumetal_runtime_mem_get_info_test
)

add_test(
    NAME functional_runtime_host_alloc
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_runtime_host_alloc_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_runtime_host_alloc PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_host_pointer_api
    COMMAND cumetal_runtime_host_pointer_api_test
)

add_test(
    NAME functional_runtime_stream_flags
    COMMAND cumetal_runtime_stream_flags_test
)

add_test(
    NAME functional_runtime_stream_callback
    COMMAND cumetal_runtime_stream_callback_test
)

add_test(
    NAME functional_runtime_device_api
    COMMAND cumetal_runtime_device_api_test
)

add_test(
    NAME functional_runtime_device_properties
    COMMAND cumetal_runtime_device_properties_test
)

add_test(
    NAME functional_runtime_device_attribute
    COMMAND cumetal_runtime_device_attribute_test
)

add_test(
    NAME functional_runtime_device_reset
    COMMAND cumetal_runtime_device_reset_test
)

add_test(
    NAME functional_runtime_device_flags
    COMMAND cumetal_runtime_device_flags_test
)

add_test(
    NAME functional_runtime_error_api
    COMMAND cumetal_runtime_error_api_test
)

add_test(
    NAME functional_runtime_profiler_api
    COMMAND cumetal_runtime_profiler_api_test
)

add_test(
    NAME functional_curand_uniform
    COMMAND cumetal_curand_uniform_test
)

add_test(
    NAME functional_cublas_api
    COMMAND cumetal_cublas_api_test
)

add_test(
    NAME functional_cufft_c2c
    COMMAND cumetal_cufft_c2c_test
)

add_test(
    NAME functional_runtime_occupancy
    COMMAND cumetal_runtime_occupancy_test
)

add_test(
    NAME functional_runtime_device_limits
    COMMAND cumetal_runtime_device_limits_test
)

add_test(
    NAME functional_driver_extended_api
    COMMAND cumetal_driver_extended_api_test
)

add_test(
    NAME functional_driver_vector_add
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_driver_vector_add_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_driver_vector_add PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_driver_null_stream_sync
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_driver_null_stream_sync_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_driver_null_stream_sync PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_driver_matrix_mul
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_axpy.sh
            $<TARGET_FILE:cumetal_driver_matrix_mul_test>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/matrix_mul.metal
            ${CMAKE_CURRENT_BINARY_DIR}/driver_matrix_mul.metallib
)

set_tests_properties(functional_driver_matrix_mul PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_driver_device_api
    COMMAND cumetal_driver_device_api_test
)

add_test(
    NAME functional_driver_error_api
    COMMAND cumetal_driver_error_api_test
)

add_test(
    NAME functional_driver_profiler_api
    COMMAND cumetal_driver_profiler_api_test
)

add_test(
    NAME functional_driver_device_query
    COMMAND cumetal_driver_device_query_test
)

add_test(
    NAME functional_driver_device_attribute
    COMMAND cumetal_driver_device_attribute_test
)

add_test(
    NAME functional_driver_stream_flags
    COMMAND cumetal_driver_stream_flags_test
)

add_test(
    NAME functional_driver_stream_per_thread
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_driver_stream_per_thread_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_driver_stream_per_thread PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_driver_stream_callback
    COMMAND cumetal_driver_stream_callback_test
)

add_test(
    NAME functional_driver_launch_host_func
    COMMAND cumetal_driver_launch_host_func_test
)

add_test(
    NAME functional_runtime_device_attribute_ext
    COMMAND cumetal_runtime_device_attribute_ext_test
)

add_test(
    NAME functional_runtime_device_properties_ext
    COMMAND cumetal_runtime_device_properties_ext_test
)

add_test(
    NAME functional_runtime_malloc_pitch
    COMMAND cumetal_runtime_malloc_pitch_test
)

# heap auto-threshold: lower threshold to 64 KiB so the 128 KiB allocation uses heap.
add_test(
    NAME functional_runtime_heap_auto_threshold
    COMMAND cumetal_runtime_heap_auto_threshold_test
)
set_tests_properties(functional_runtime_heap_auto_threshold
    PROPERTIES ENVIRONMENT "CUMETAL_MTLHEAP_THRESHOLD_BYTES=65536"
)

# heap disabled: CUMETAL_MTLHEAP_ALLOC=0 must not crash even for large allocations.
add_test(
    NAME functional_runtime_heap_disabled
    COMMAND cumetal_runtime_heap_auto_threshold_test
)
set_tests_properties(functional_runtime_heap_disabled
    PROPERTIES ENVIRONMENT "CUMETAL_MTLHEAP_ALLOC=0"
)

add_test(
    NAME functional_runtime_peer_access
    COMMAND cumetal_runtime_peer_access_test
)

add_test(
    NAME functional_runtime_stream_priority
    COMMAND cumetal_runtime_stream_priority_test
)

add_test(
    NAME functional_runtime_cooperative_groups_grid
    COMMAND cumetal_runtime_cooperative_groups_grid_test
)

add_test(
    NAME functional_driver_context_switch
    COMMAND cumetal_driver_context_switch_test
)

add_test(
    NAME functional_driver_context_requirements
    COMMAND cumetal_driver_context_requirements_test
)

add_test(
    NAME functional_driver_async_memcpy
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_driver_async_memcpy_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_driver_async_memcpy PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_driver_memset
    COMMAND cumetal_driver_memset_test
)

add_test(
    NAME functional_driver_mem_get_info
    COMMAND cumetal_driver_mem_get_info_test
)

add_test(
    NAME functional_driver_mem_alloc_managed
    COMMAND cumetal_driver_mem_alloc_managed_test
)

add_test(
    NAME functional_driver_host_alloc
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_driver_host_alloc_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_driver_host_alloc PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_driver_host_pointer_api
    COMMAND cumetal_driver_host_pointer_api_test
)

add_test(
    NAME functional_driver_module_load_data
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_driver_module_load_data_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_driver_module_load_data PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_driver_module_load_data_ptx
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_driver_module_load_data_ptx.sh
            $<TARGET_FILE:cumetal_driver_module_load_data_ptx_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/vector_add.ptx
)

set_tests_properties(functional_driver_module_load_data_ptx PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_driver_launch_extra
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_driver_launch_extra_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_driver_launch_extra PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_driver_launch_extra_scalar
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_axpy.sh
            $<TARGET_FILE:cumetal_driver_launch_extra_scalar_test>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/matrix_mul_scalar.metal
            ${CMAKE_CURRENT_BINARY_DIR}/matrix_mul_scalar.metallib
)

set_tests_properties(functional_driver_launch_extra_scalar PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_driver_stream_wait_event
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_driver_stream_wait_event_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_driver_stream_wait_event PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_axpy_offset
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_axpy.sh
            $<TARGET_FILE:cumetal_runtime_axpy_offset_test>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/axpy.metal
            ${CMAKE_CURRENT_BINARY_DIR}/axpy.metallib
)

set_tests_properties(functional_runtime_axpy_offset PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_atomic
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_axpy.sh
            $<TARGET_FILE:cumetal_runtime_atomic_test>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/atomic_inc.metal
            ${CMAKE_CURRENT_BINARY_DIR}/atomic_inc.metallib
)

set_tests_properties(functional_runtime_atomic PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_atomic_shared
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_axpy.sh
            $<TARGET_FILE:cumetal_runtime_atomic_shared_test>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/atomic_shared.metal
            ${CMAKE_CURRENT_BINARY_DIR}/atomic_shared.metallib
)

set_tests_properties(functional_runtime_atomic_shared PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_warp_shuffle
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_axpy.sh
            $<TARGET_FILE:cumetal_runtime_warp_shuffle_test>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/warp_shuffle.metal
            ${CMAKE_CURRENT_BINARY_DIR}/warp_shuffle.metallib
)

set_tests_properties(functional_runtime_warp_shuffle PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_warp_vote
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_axpy.sh
            $<TARGET_FILE:cumetal_runtime_warp_vote_test>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/warp_vote.metal
            ${CMAKE_CURRENT_BINARY_DIR}/warp_vote.metallib
)

set_tests_properties(functional_runtime_warp_vote PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_fp16_ops
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_axpy.sh
            $<TARGET_FILE:cumetal_runtime_fp16_ops_test>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/fp16_ops.metal
            ${CMAKE_CURRENT_BINARY_DIR}/fp16_ops.metallib
)

set_tests_properties(functional_runtime_fp16_ops PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_shared_reduce
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_axpy.sh
            $<TARGET_FILE:cumetal_runtime_shared_reduce_test>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/shared_reduce.metal
            ${CMAKE_CURRENT_BINARY_DIR}/shared_reduce.metallib
)

set_tests_properties(functional_runtime_shared_reduce PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_grid_2d
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_axpy.sh
            $<TARGET_FILE:cumetal_runtime_grid_2d_test>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/grid_2d.metal
            ${CMAKE_CURRENT_BINARY_DIR}/grid_2d.metallib
)

set_tests_properties(functional_runtime_grid_2d PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_grid_3d
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_axpy.sh
            $<TARGET_FILE:cumetal_runtime_grid_3d_test>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/grid_3d.metal
            ${CMAKE_CURRENT_BINARY_DIR}/grid_3d.metallib
)

set_tests_properties(functional_runtime_grid_3d PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_fp64_ops
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_driver_module_load_data_ptx.sh
            $<TARGET_FILE:cumetal_runtime_fp64_ops_test>
            ${CMAKE_CURRENT_SOURCE_DIR}/reference/fp64_mul_add.ptx
)

set_tests_properties(functional_runtime_fp64_ops PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_ptx_lowering_regression
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_ptx_lowering_regression.sh
            $<TARGET_FILE:cumetalc>
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_BINARY_DIR}
)

set_tests_properties(functional_runtime_ptx_lowering_regression PROPERTIES SKIP_RETURN_CODE 77)

add_test(
    NAME functional_runtime_cooperative_launch
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
            ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
            $<TARGET_FILE:cumetal_runtime_cooperative_launch_test>
            ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
)

set_tests_properties(functional_runtime_cooperative_launch PROPERTIES SKIP_RETURN_CODE 77)

if(CUMETAL_ENABLE_BINARY_SHIM)
    add_test(
        NAME functional_runtime_registration_path
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
                ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
                $<TARGET_FILE:cumetal_runtime_registration_path_test>
                ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
    )

    set_tests_properties(functional_runtime_registration_path PROPERTIES SKIP_RETURN_CODE 77)

    add_test(
        NAME functional_runtime_call_config_registration
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
                ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
                $<TARGET_FILE:cumetal_runtime_call_config_registration_test>
                ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
    )

    set_tests_properties(functional_runtime_call_config_registration PROPERTIES SKIP_RETURN_CODE 77)

    add_test(
        NAME functional_runtime_registration_fatbin_ptx
        COMMAND bash
                ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_registration_fatbin_ptx.sh
                $<TARGET_FILE:cumetal_runtime_registration_fatbin_ptx_test>
                ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/vector_add.ptx
    )

    set_tests_properties(functional_runtime_registration_fatbin_ptx PROPERTIES SKIP_RETURN_CODE 77)

    add_test(
        NAME functional_runtime_legacy_launch_registration
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
                ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
                $<TARGET_FILE:cumetal_runtime_legacy_launch_registration_test>
                ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
    )

    set_tests_properties(functional_runtime_legacy_launch_registration PROPERTIES SKIP_RETURN_CODE 77)

    add_test(
        NAME functional_runtime_registration_fatbinary2_symbols
        COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_vector_add.sh
                ${CMAKE_SOURCE_DIR}/scripts/generate_reference_metallib.sh
                $<TARGET_FILE:cumetal_runtime_registration_fatbinary2_symbols_test>
                ${CMAKE_SOURCE_DIR}/tests/air_abi/reference/reference.metallib
    )

    set_tests_properties(functional_runtime_registration_fatbinary2_symbols PROPERTIES SKIP_RETURN_CODE 77)

    add_test(
        NAME functional_runtime_registration_var_symbol
        COMMAND cumetal_runtime_registration_var_symbol_test
    )

    add_test(
        NAME functional_runtime_registration_printf
        COMMAND bash
                ${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_registration_fatbin_ptx.sh
                $<TARGET_FILE:cumetal_runtime_registration_printf_test>
                ${CMAKE_CURRENT_SOURCE_DIR}/reference/printf_scale.ptx
    )

    set_tests_properties(functional_runtime_registration_printf PROPERTIES SKIP_RETURN_CODE 77)
endif()
