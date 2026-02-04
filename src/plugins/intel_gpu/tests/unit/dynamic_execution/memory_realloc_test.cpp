// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/concatenation.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/softmax.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/reshape.hpp>
#include <intel_gpu/primitives/data.hpp>
#include "intel_gpu/runtime/memory_pool.hpp"
#include "intel_gpu/runtime/engine.hpp"

#include "softmax_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>
#include <cstdint>

using namespace cldnn;
using namespace ::tests;

namespace memory_realloc_tests {
TEST(memory_reuse_realloc_reset_test, basic_conv_with_padding) {
    auto& engine = get_test_engine();

    layout weight_layout = layout{ov::PartialShape{1, 3, 3, 3}, data_types::f16, format::bfyx};

    auto weights = engine.allocate_memory(weight_layout);
    set_values<ov::float16>(weights, {
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            //
            2.0f, 2.0f, 2.0f,
            2.0f, 2.0f, 2.0f,
            2.0f, 2.0f, 2.0f,
            //
            3.0f, 3.0f, 3.0f,
            3.0f, 3.0f, 3.0f,
            3.0f, 3.0f, 3.0f,
    });

    layout input_layout_1 = layout{ov::PartialShape{1, 3, 5, 5}, data_types::f32, format::bfyx};
    auto input_mem_1 = engine.allocate_memory(input_layout_1);
    set_values(input_mem_1, {
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         //
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         //
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                        });

    std::vector<float> ref_output_1 = {6,   18,  36, 54,  72,  54,  30,  12,  36, 72, 108, 144, 108,
                                       60,  18,  54, 108, 162, 216, 162, 90,  18, 54, 108, 162, 216,
                                       162, 90,  18, 54,  108, 162, 216, 162, 90, 12, 36,  72,  108,
                                       144, 108, 60, 6,   18,  36,  54,  72,  54, 30};

    layout input_layout_2 = layout{ov::PartialShape{1, 3, 2, 2}, data_types::f32, format::bfyx};
    auto input_mem_2 = engine.allocate_memory(input_layout_2);
    set_values(input_mem_2, {11.0f,  11.0f, 11.0f, 11.0f,
                             11.0f,  11.0f, 11.0f, 11.0f,
                             11.0f,  11.0f, 11.0f, 11.0f});
    std::vector<float> ref_output_2 = { 66, 132, 132, 66, 132, 264, 264, 132, 132, 264, 264, 132, 66, 132, 132, 66};
     std::vector<float> values_to_subtract = {};
    auto input_l = layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx};
    topology topology(input_layout("input", input_l),
                      data("weights", weights),
                      reorder("reorder", input_info("input"), format::bfyx, data_types::f16,
                      values_to_subtract, reorder_mean_mode::subtract, padding{{0, 0, 2, 2}, 0}),
                      convolution("conv",
                                  input_info("reorder"),
                                  "weights",
                                  "",     /*bias*/
                                  1,
                                  {1, 1}, /*stride*/
                                  {1, 1}, /*dilation*/
                                  {2, 2},  /*pad_above*/
                                  {2, 2},  /*pad_below*/
                                  false,
                                  ov::op::PadType::EXPLICIT),
                      reorder("output", input_info("conv"), format::bfyx, data_types::f32)); /*output padding*/

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"conv", {format::any, "", impl_types::ocl}}}));

    network network(engine, topology, config);
    network.set_input_data("input", input_mem_1);
    auto outputs_1 = network.execute();
    network.set_input_data("input", input_mem_2);
    auto outputs_2 = network.execute();
    auto output_mem_2 = outputs_2.begin()->second.get_memory();
    cldnn::mem_lock<float> output_mem_2_ptr(output_mem_2, get_test_stream());
    for (size_t i = 0; i < output_mem_2->get_layout().get_linear_size(); ++i) {
        ASSERT_EQ(output_mem_2_ptr[i], ref_output_2[i]);
    }
    // check padding of second run of reorder
    // 0, 0, 0,  0,  0, 0,
    // 0, 0, 0,  0,  0, 0,
    // 0, 0, 11, 11, 0, 0,
    // 0, 0, 11, 11, 0, 0,
    // 0, 0,"0","0","0","0", // !! check pad_after
    // 0, 0,"0","0","0","0", // !! check pad_after
    auto reorder_mem = network.get_primitive("reorder")->output_memory_ptr();
    cldnn::mem_lock<ov::float16, mem_lock_type::read> reorder_mem_ptr(reorder_mem, get_test_stream());
    for (size_t i = 26; i < 29; ++i) {
        ASSERT_EQ((float)reorder_mem_ptr[i], 0.f);
    }
    for (size_t i = 32; i < 35; ++i) {
        ASSERT_EQ((float)reorder_mem_ptr[i], 0.f);
    }
    // Mem should be reallocate when request size is bigger than existing buffer size
    ASSERT_TRUE(reorder_mem->size() <= reorder_mem->get_mem_tracker()->size())
                << "reorder mem buffer size: " <<  reorder_mem->size() << "bytes is bigger than original size of allocated mem: "
                << reorder_mem->get_mem_tracker()->size() << "bytes.";
}

TEST(memory_reuse_realloc_reset_test, basic_conv_with_memory_get_from_padded_pool) {
    auto& engine = get_test_engine();

    layout weight_layout = layout{ov::PartialShape{1, 4, 3, 3}, data_types::f32, format::bfyx};
    auto weights = engine.allocate_memory(weight_layout);
    set_values<float>(weights, {
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,

        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,

        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,

        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f
    });

    layout weight_layout2 = layout{ov::PartialShape{1, 3, 3, 3}, data_types::f32, format::bfyx};
    auto weights2 = engine.allocate_memory(weight_layout2);
    set_values<float>(weights2, {
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,

        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,

        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f
    });

    layout elt_layout1 = layout{ov::PartialShape{1, 2, 4, 4}, data_types::f32, format::bfyx};
    auto elt_mem1 = engine.allocate_memory(elt_layout1);
    set_values<float>(elt_mem1, {
        10.f, 10.f, 10.f, 10.f,
        10.f, 10.f, 10.f, 10.f,
        10.f, 10.f, 10.f, 10.f,
        10.f, 10.f, 10.f, 10.f,

        10.f, 10.f, 10.f, 10.f,
        10.f, 10.f, 10.f, 10.f,
        10.f, 10.f, 10.f, 10.f,
        10.f, 10.f, 10.f, 10.f
    });

    std::vector<float> ref_output = {
        1080, 1720, 1720, 1080,
        1720, 2740, 2740, 1720,
        1720, 2740, 2740, 1720,
        1080, 1720, 1720, 1080
    };

    std::vector<float> subtract_val = {0.f, };
    auto input_l = layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx};
    auto elt_input_l = layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx};

    topology topology(input_layout("elt_input", elt_input_l),
                      data("weights", weights),
                      data("weights2", weights2),
                      reorder("reorder1-1", input_info("elt_input"), format::bfyx, data_types::f32, subtract_val, reorder_mean_mode::subtract),
                      reorder("reorder1-2", input_info("elt_input"), format::bfyx, data_types::f32, subtract_val, reorder_mean_mode::subtract),
                      concatenation("concat1", {input_info("reorder1-1"), input_info("reorder1-2")}, 1),
                      convolution("conv1",
                                  input_info("concat1"),
                                  "weights",
                                  "",     /*bias*/
                                  1,
                                  {1, 1}, /*stride*/
                                  {1, 1}, /*dilation*/
                                  {1, 1}, /*pad_above*/
                                  {1, 1}, /*pad_below*/
                                  false,
                                  ov::op::PadType::EXPLICIT),
                      reorder("reorder2-1", input_info("conv1"), format::bfyx, data_types::f32, subtract_val, reorder_mean_mode::subtract),
                      concatenation("concat2", {input_info("reorder1-1"), input_info("reorder2-1")}, 1),
                      convolution("conv2",
                                  input_info("concat2"),
                                  "weights2",
                                  "",     /*bias*/
                                  1,
                                  {1, 1}, /*stride*/
                                  {1, 1}, /*dilation*/
                                  {1, 1}, /*pad_above*/
                                  {1, 1}, /*pad_below*/
                                  false,
                                  ov::op::PadType::EXPLICIT),
                      reorder("output", input_info("conv2"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::queue_type(QueueTypes::in_order));

    network network(engine, topology, config);
    network.set_input_data("elt_input", elt_mem1);
    auto outputs = network.execute();
    auto output_mem = outputs.begin()->second.get_memory();
    cldnn::mem_lock<float> output_mem_ptr(output_mem, get_test_stream());

    for (size_t i = 0; i < output_mem->get_layout().get_linear_size(); ++i) {
        ASSERT_EQ(output_mem_ptr[i], ref_output[i]);
    }
}

TEST(softmax_gpu_dynamic_f32_test_upper_bound, input_same_values) {
    static const int32_t
        output_x_1  = 10, output_b_1  = 8,
        input_x_1   = 10, input_b_1   = 8,
        out_size_1  = output_x_1 * output_b_1,
        output_x_2  = 10, output_b_2  = 4,
        input_x_2   = 10, input_b_2  = 4,
        out_size_2  = output_x_2 * output_b_2,
        output_x_3  = 10, output_b_3  = 16,
        input_x_3   = 10, input_b_3  = 16,
        out_size_3  = output_x_3 * output_b_3;

    cldnn::engine& engine = get_test_engine();

    auto compare_out_buffer_with_expected = [&](float* out_buffer, std::vector<float>& expected_buffer, size_t size) {
        for(size_t i = 0; i < size; ++i) {
            // does output have expected values
            ASSERT_TRUE(are_equal(out_buffer[i], expected_buffer[i]))
                << "At ["<< i <<  "] Expected : " << expected_buffer[i] << " actual : " << out_buffer[i];
        }
    };
    auto in_layout =
        layout(ov::PartialShape{ov::Dimension{1, 10}, ov::Dimension{1, 10}, ov::Dimension{1, 10}, ov::Dimension{1, 10}},
               data_types::f32,
               format::bfyx);
    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    ov::intel_gpu::ImplementationDesc softmax_impl = { format::bfyx, "softmax_gpu_ref" };
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "softmax", softmax_impl } }));
    network network(engine, topology(input_layout("input", in_layout),
                                     reorder("reorder", input_info("input"), format::bfyx, data_types::f16),
                                     softmax("softmax", input_info("reorder"), 3),
                                     reorder("reorder2", input_info("softmax"), format::bfyx, data_types::f32)),
                                     config);

    // First run
    float out_buffer_1[out_size_1];
    std::vector<float> in_b_1(out_size_1, 1.0f);
    std::vector<float> expected_buffer_1(out_size_1, 0.1f);
    cldnn::memory::ptr input_1 = engine.allocate_memory({ data_types::f32, format::bfyx, {input_b_1, 1, input_x_1, 1}});
    set_values(input_1, in_b_1);
    network.set_input_data("input", input_1);

    auto outputs_1 = network.execute();
    auto output_mem_1 = outputs_1.begin()->second.get_memory();
    auto internal_mems_1 = network.get_primitive("softmax")->get_intermediates_memories();
    cldnn::mem_lock<float> output_ptr_1(output_mem_1, get_test_stream());
    for (uint32_t i = 0; i < out_size_1; i++) {
        out_buffer_1[i] = output_ptr_1[i];
    }
    compare_out_buffer_with_expected(out_buffer_1, expected_buffer_1, out_size_1);

    // Second run
    float out_buffer_2[out_size_2];
    std::vector<float> in_b_2(out_size_2, 2.0f);
    std::vector<float> expected_buffer_2(out_size_2, 0.1f);
    cldnn::memory::ptr input_2 = engine.allocate_memory({ data_types::f32, format::bfyx, {input_b_2, 1, input_x_2, 1}});
    set_values(input_2, in_b_2);
    network.set_input_data("input", input_2);
    auto outputs_2 = network.execute();
    auto output_mem_2 = outputs_2.begin()->second.get_memory();
    auto internal_mems_2 = network.get_primitive("softmax")->get_intermediates_memories();
    cldnn::mem_lock<float> output_ptr_2(output_mem_2, get_test_stream());
    for (uint32_t i = 0; i < out_size_2; i++) {
        out_buffer_2[i] = output_ptr_2[i];
    }
    compare_out_buffer_with_expected(out_buffer_2, expected_buffer_2, out_size_2);

    // Check output is not reallocated
    ASSERT_EQ(output_ptr_1.data(), output_ptr_2.data());
    ASSERT_EQ(internal_mems_1.size(), internal_mems_2.size());
    for (size_t i = 0; i < internal_mems_1.size(); ++i) {
        ASSERT_EQ(internal_mems_1[i]->buffer_ptr(), internal_mems_2[i]->buffer_ptr());
        if (engine.get_device_info().supports_immad) {
            ASSERT_EQ(internal_mems_1[i]->get_allocation_type(), allocation_type::usm_device);
        }
    }
    // Third run
    float out_buffer_3[out_size_3];
    std::vector<float> in_b_3(out_size_3, 2.0f);
    std::vector<float> expected_buffer_3(out_size_3, 0.1f);
    cldnn::memory::ptr input_3 = engine.allocate_memory({ data_types::f32, format::bfyx, {input_b_3, 1, input_x_3, 1}});
    set_values(input_3, in_b_3);
    network.set_input_data("input", input_3);
    auto outputs_3 = network.execute();
    auto output_mem_3 = outputs_3.begin()->second.get_memory();
    cldnn::mem_lock<float> output_ptr_3(output_mem_3, get_test_stream());
    for (uint32_t i = 0; i < out_size_3; i++) {
        out_buffer_3[i] = output_ptr_3[i];
    }
    compare_out_buffer_with_expected(out_buffer_3, expected_buffer_3, out_size_3);
    auto internal_mems_3 = network.get_primitive("softmax")->get_intermediates_memories();
    for (size_t i = 0; i < internal_mems_3.size(); ++i) {
        if (engine.get_device_info().supports_immad) {
            ASSERT_EQ(internal_mems_3[i]->get_allocation_type(), allocation_type::usm_device);
        }
    }
    auto& pool = network.get_memory_pool();
    // check if previously allocated internal buffer is released
    ASSERT_EQ(pool.get_non_padded_pool_size(), 3);
}

TEST(dyn_shape_mem_test, igpu_shape_infer_dep_mem_type) {
    auto& engine = get_test_engine();
    auto input_lay_1 = layout{ov::PartialShape::dynamic(2), data_types::f32, format::bfyx};
    auto input_lay_2 = layout{ov::PartialShape::dynamic(2), data_types::i32, format::bfyx};
    topology topology(input_layout("input1", input_lay_1),
                      input_layout("pattern1", input_lay_2),
                      input_layout("pattern2", input_lay_2),
                      reorder("reorder", input_info("input1"), format::bfyx, data_types::f16),
                      eltwise("eltwise", {input_info("pattern1"), input_info("pattern2")}, eltwise_mode::sum, ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::NUMPY)),
                      reshape("reshape", input_info("reorder"), input_info("eltwise"), false, ov::PartialShape()));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);

    auto input_mem = engine.allocate_memory(layout{ov::PartialShape{6, 1}, data_types::f32, format::bfyx});
    set_values<float>(input_mem, {11.f, 22.f, 33.f, 44.f, 55.f, 66.f});;
    auto pattern_mem1 = engine.allocate_memory(layout{ov::PartialShape{4}, data_types::i32, format::bfyx});
    set_values<int32_t>(pattern_mem1, {2, 1, 1, 0});;
    auto pattern_mem2 = engine.allocate_memory(layout{ov::PartialShape{4}, data_types::i32, format::bfyx});
    set_values<int32_t>(pattern_mem2, {1, 1, 0, 1});;

    network.set_input_data("input1", input_mem);
    network.set_input_data("pattern1", pattern_mem1);
    network.set_input_data("pattern2", pattern_mem2);
    auto output = network.execute();
    const auto& reorder_mem = network.get_primitive("reorder")->output_memory();
    const auto& pattern_mem = network.get_primitive("eltwise")->output_memory();
    ASSERT_EQ(reorder_mem.get_allocation_type(), allocation_type::usm_device);
    if (engine.get_device_info().dev_type == device_type::integrated_gpu) {
        // for iGPU, allocating shape infer dep mem to usm_host improves shape_infer performance by preventing memcpy b/w device to host mem
        ASSERT_EQ(pattern_mem.get_allocation_type(), allocation_type::usm_host);
    } else {
        // if allocate shape infer dep mem to host && write result from device && read from host, cache coherence issue occurs
        ASSERT_EQ(pattern_mem.get_allocation_type(), allocation_type::usm_device);
    }
    auto expected_layout = layout{ov::PartialShape{3, 2, 1, 1}, data_types::f16, format::bfyx};
    ASSERT_EQ(output.begin()->second.get_memory()->get_layout(), expected_layout);
}

TEST(memory_reuse_realloc_reset_test, usm_subbuffer_multi_live_reuse) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_usm || !engine.use_unified_shared_memory()) {
        GTEST_SKIP() << "USM not supported";
    }

    ExecutionConfig config = get_test_default_config(engine);
    memory_pool pool(engine, config);

    std::unordered_set<uint32_t> restriction_set;
    memory_restricter<uint32_t> restrictions(&restriction_set);

    layout large_layout = layout{ov::PartialShape{1, 1, 1, 4096}, data_types::f32, format::bfyx};
    layout small_layout = layout{ov::PartialShape{1, 1, 1, 1024}, data_types::f32, format::bfyx};

    auto large = pool.get_from_non_padded_pool(large_layout, "large", 1, 0, restrictions, allocation_type::usm_device, true, false);
    pool.release_memory(large.get(), 1, "large", 0);

    auto small1 = pool.get_from_non_padded_pool(small_layout, "small1", 2, 0, restrictions, allocation_type::usm_device, true, false);
    auto small2 = pool.get_from_non_padded_pool(small_layout, "small2", 3, 0, restrictions, allocation_type::usm_device, true, false);

    ASSERT_EQ(small1->get_mem_tracker(), small2->get_mem_tracker());
    ASSERT_NE(small1->buffer_ptr(), small2->buffer_ptr());
}

TEST(memory_reuse_realloc_reset_test, ocl_subbuffer_multi_live_reuse) {
    auto& engine = get_test_engine();
    if (engine.runtime_type() != runtime_types::ocl) {
        GTEST_SKIP() << "OpenCL runtime required";
    }

    ExecutionConfig config = get_test_default_config(engine);
    memory_pool pool(engine, config);

    std::unordered_set<uint32_t> restriction_set;
    memory_restricter<uint32_t> restrictions(&restriction_set);

    layout large_layout = layout{ov::PartialShape{1, 1, 1, 4096}, data_types::f32, format::bfyx};
    layout small_layout = layout{ov::PartialShape{1, 1, 1, 1024}, data_types::f32, format::bfyx};

    if (!engine.check_allocatable(large_layout, allocation_type::cl_mem) ||
        !engine.check_allocatable(small_layout, allocation_type::cl_mem)) {
        GTEST_SKIP() << "cl_mem allocation not supported";
    }

    auto large = pool.get_from_non_padded_pool(large_layout, "large", 10, 0, restrictions, allocation_type::cl_mem, true, false);
    pool.release_memory(large.get(), 10, "large", 0);

    auto small1 = pool.get_from_non_padded_pool(small_layout, "small1", 11, 0, restrictions, allocation_type::cl_mem, true, false);
    auto small2 = pool.get_from_non_padded_pool(small_layout, "small2", 12, 0, restrictions, allocation_type::cl_mem, true, false);

    ASSERT_EQ(small1->get_mem_tracker(), small2->get_mem_tracker());
    ASSERT_NE(small1->buffer_ptr(), small2->buffer_ptr());
}

TEST(memory_reuse_realloc_reset_test, usm_subbuffer_shared_segment_split) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_usm || !engine.use_unified_shared_memory()) {
        GTEST_SKIP() << "USM not supported";
    }

    ExecutionConfig config = get_test_default_config(engine);
    memory_pool pool(engine, config);

    std::unordered_set<uint32_t> restriction_set;
    memory_restricter<uint32_t> restrictions(&restriction_set);

    layout layout_a = layout{ov::PartialShape{1, 1, 1, 256}, data_types::u8, format::bfyx};
    layout layout_b = layout{ov::PartialShape{1, 1, 1, 64}, data_types::u8, format::bfyx};
    layout layout_c = layout{ov::PartialShape{1, 1, 1, 192}, data_types::u8, format::bfyx};

    auto mem_a = pool.get_from_non_padded_pool(layout_a, "A", 100, 0, restrictions, allocation_type::usm_device, true, false);
    auto mem_b = pool.get_from_non_padded_pool(layout_b, "B", 101, 0, restrictions, allocation_type::usm_device, true, false);

    restriction_set.insert(101);
    auto mem_c = pool.get_from_non_padded_pool(layout_c, "C", 102, 0, restrictions, allocation_type::usm_device, true, false);

    ASSERT_EQ(mem_a->get_mem_tracker(), mem_b->get_mem_tracker());
    ASSERT_EQ(mem_a->get_mem_tracker(), mem_c->get_mem_tracker());

    auto a_ptr = reinterpret_cast<uint8_t*>(mem_a->buffer_ptr());
    auto b_ptr = reinterpret_cast<uint8_t*>(mem_b->buffer_ptr());
    auto c_ptr = reinterpret_cast<uint8_t*>(mem_c->buffer_ptr());

    ASSERT_EQ(b_ptr, a_ptr);
    ASSERT_NE(c_ptr, a_ptr);
    ASSERT_NE(c_ptr, b_ptr);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(c_ptr) - reinterpret_cast<uintptr_t>(a_ptr), 64u);
}

TEST(memory_reuse_realloc_reset_test, ocl_subbuffer_shared_segment_split) {
    auto& engine = get_test_engine();
    if (engine.runtime_type() != runtime_types::ocl) {
        GTEST_SKIP() << "OpenCL runtime required";
    }

    ExecutionConfig config = get_test_default_config(engine);
    memory_pool pool(engine, config);

    std::unordered_set<uint32_t> restriction_set;
    memory_restricter<uint32_t> restrictions(&restriction_set);

    layout layout_a = layout{ov::PartialShape{1, 1, 1, 256}, data_types::u8, format::bfyx};
    layout layout_b = layout{ov::PartialShape{1, 1, 1, 64}, data_types::u8, format::bfyx};
    layout layout_c = layout{ov::PartialShape{1, 1, 1, 192}, data_types::u8, format::bfyx};

    if (!engine.check_allocatable(layout_a, allocation_type::cl_mem) ||
        !engine.check_allocatable(layout_b, allocation_type::cl_mem) ||
        !engine.check_allocatable(layout_c, allocation_type::cl_mem)) {
        GTEST_SKIP() << "cl_mem allocation not supported";
    }

    auto mem_a = pool.get_from_non_padded_pool(layout_a, "A", 200, 0, restrictions, allocation_type::cl_mem, true, false);
    auto mem_b = pool.get_from_non_padded_pool(layout_b, "B", 201, 0, restrictions, allocation_type::cl_mem, true, false);

    restriction_set.insert(201);
    auto mem_c = pool.get_from_non_padded_pool(layout_c, "C", 202, 0, restrictions, allocation_type::cl_mem, true, false);

    ASSERT_EQ(mem_a->get_mem_tracker(), mem_b->get_mem_tracker());
    ASSERT_EQ(mem_a->get_mem_tracker(), mem_c->get_mem_tracker());
    ASSERT_NE(mem_b->buffer_ptr(), mem_c->buffer_ptr());
}

TEST(memory_reuse_realloc_reset_test, usm_subbuffer_shared_segment_split_five) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_usm || !engine.use_unified_shared_memory()) {
        GTEST_SKIP() << "USM not supported";
    }

    ExecutionConfig config = get_test_default_config(engine);
    memory_pool pool(engine, config);

    std::unordered_set<uint32_t> restriction_set;
    memory_restricter<uint32_t> restrictions(&restriction_set);

    layout layout_a = layout{ov::PartialShape{1, 1, 1, 320}, data_types::u8, format::bfyx};
    layout layout_b = layout{ov::PartialShape{1, 1, 1, 64}, data_types::u8, format::bfyx};

    auto mem_a = pool.get_from_non_padded_pool(layout_a, "A", 300, 0, restrictions, allocation_type::usm_device, true, false);

    std::vector<memory::ptr> blocks;
    blocks.reserve(5);
    for (size_t i = 0; i < 5; ++i) {
        if (i > 0)
            restriction_set.insert(static_cast<uint32_t>(300 + i));
        blocks.push_back(pool.get_from_non_padded_pool(layout_b,
                                                       "B" + std::to_string(i + 1),
                                                       301 + i,
                                                       0,
                                                       restrictions,
                                                       allocation_type::usm_device,
                                                       true,
                                                       false));
    }

    auto a_ptr = reinterpret_cast<uint8_t*>(mem_a->buffer_ptr());
    for (size_t i = 0; i < blocks.size(); ++i) {
        auto b_ptr = reinterpret_cast<uint8_t*>(blocks[i]->buffer_ptr());
        ASSERT_EQ(reinterpret_cast<uintptr_t>(b_ptr) - reinterpret_cast<uintptr_t>(a_ptr), 64u * i);
    }
}

TEST(memory_reuse_realloc_reset_test, ocl_subbuffer_shared_segment_split_five) {
    auto& engine = get_test_engine();
    if (engine.runtime_type() != runtime_types::ocl) {
        GTEST_SKIP() << "OpenCL runtime required";
    }

    ExecutionConfig config = get_test_default_config(engine);
    memory_pool pool(engine, config);

    std::unordered_set<uint32_t> restriction_set;
    memory_restricter<uint32_t> restrictions(&restriction_set);

    layout layout_a = layout{ov::PartialShape{1, 1, 1, 320}, data_types::u8, format::bfyx};
    layout layout_b = layout{ov::PartialShape{1, 1, 1, 64}, data_types::u8, format::bfyx};

    if (!engine.check_allocatable(layout_a, allocation_type::cl_mem) ||
        !engine.check_allocatable(layout_b, allocation_type::cl_mem)) {
        GTEST_SKIP() << "cl_mem allocation not supported";
    }

    auto mem_a = pool.get_from_non_padded_pool(layout_a, "A", 400, 0, restrictions, allocation_type::cl_mem, true, false);

    std::vector<memory::ptr> blocks;
    blocks.reserve(5);
    for (size_t i = 0; i < 5; ++i) {
        if (i > 0)
            restriction_set.insert(static_cast<uint32_t>(400 + i));
        blocks.push_back(pool.get_from_non_padded_pool(layout_b,
                                                       "B" + std::to_string(i + 1),
                                                       401 + i,
                                                       0,
                                                       restrictions,
                                                       allocation_type::cl_mem,
                                                       true,
                                                       false));
    }

    ASSERT_EQ(mem_a->get_mem_tracker(), blocks.front()->get_mem_tracker());
    for (size_t i = 1; i < blocks.size(); ++i) {
        ASSERT_NE(blocks[0]->buffer_ptr(), blocks[i]->buffer_ptr());
    }
}

TEST(memory_reuse_realloc_reset_test, usm_subbuffer_release_shared_segment) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_usm || !engine.use_unified_shared_memory()) {
        GTEST_SKIP() << "USM not supported";
    }

    ExecutionConfig config = get_test_default_config(engine);
    memory_pool pool(engine, config);

    std::unordered_set<uint32_t> restriction_set;
    memory_restricter<uint32_t> restrictions(&restriction_set);

    layout layout_a = layout{ov::PartialShape{1, 1, 1, 192}, data_types::u8, format::bfyx};
    layout layout_b = layout{ov::PartialShape{1, 1, 1, 64}, data_types::u8, format::bfyx};

    auto mem_a = pool.get_from_non_padded_pool(layout_a, "A", 500, 0, restrictions, allocation_type::usm_device, true, false);
    auto mem_b1 = pool.get_from_non_padded_pool(layout_b, "B1", 501, 0, restrictions, allocation_type::usm_device, true, false);
    restriction_set.insert(501);
    auto mem_b2 = pool.get_from_non_padded_pool(layout_b, "B2", 502, 0, restrictions, allocation_type::usm_device, true, false);
    restriction_set.insert(502);
    auto mem_b3 = pool.get_from_non_padded_pool(layout_b, "B3", 503, 0, restrictions, allocation_type::usm_device, true, false);

    pool.release_memory(mem_b2.get(), 502, "B2", 0);

    restriction_set.clear();
    restriction_set.insert(501);
    restriction_set.insert(503);
    auto mem_c = pool.get_from_non_padded_pool(layout_b, "C", 504, 0, restrictions, allocation_type::usm_device, true, false);

    ASSERT_EQ(mem_a->get_mem_tracker(), mem_c->get_mem_tracker());
    ASSERT_EQ(mem_b2->buffer_ptr(), mem_c->buffer_ptr());
}

TEST(memory_reuse_realloc_reset_test, ocl_subbuffer_release_shared_segment) {
    auto& engine = get_test_engine();
    if (engine.runtime_type() != runtime_types::ocl) {
        GTEST_SKIP() << "OpenCL runtime required";
    }

    ExecutionConfig config = get_test_default_config(engine);
    memory_pool pool(engine, config);

    std::unordered_set<uint32_t> restriction_set;
    memory_restricter<uint32_t> restrictions(&restriction_set);

    layout layout_a = layout{ov::PartialShape{1, 1, 1, 192}, data_types::u8, format::bfyx};
    layout layout_b = layout{ov::PartialShape{1, 1, 1, 64}, data_types::u8, format::bfyx};

    if (!engine.check_allocatable(layout_a, allocation_type::cl_mem) ||
        !engine.check_allocatable(layout_b, allocation_type::cl_mem)) {
        GTEST_SKIP() << "cl_mem allocation not supported";
    }

    auto mem_a = pool.get_from_non_padded_pool(layout_a, "A", 600, 0, restrictions, allocation_type::cl_mem, true, false);
    auto mem_b1 = pool.get_from_non_padded_pool(layout_b, "B1", 601, 0, restrictions, allocation_type::cl_mem, true, false);
    restriction_set.insert(601);
    auto mem_b2 = pool.get_from_non_padded_pool(layout_b, "B2", 602, 0, restrictions, allocation_type::cl_mem, true, false);
    restriction_set.insert(602);
    auto mem_b3 = pool.get_from_non_padded_pool(layout_b, "B3", 603, 0, restrictions, allocation_type::cl_mem, true, false);

    pool.release_memory(mem_b2.get(), 602, "B2", 0);

    restriction_set.clear();
    restriction_set.insert(601);
    restriction_set.insert(603);
    auto mem_c = pool.get_from_non_padded_pool(layout_b, "C", 604, 0, restrictions, allocation_type::cl_mem, true, false);

    ASSERT_EQ(mem_a->get_mem_tracker(), mem_c->get_mem_tracker());
    ASSERT_EQ(mem_b2->buffer_ptr(), mem_c->buffer_ptr());
}

TEST(memory_reuse_realloc_reset_test, basic_conv_with_padding_reorder) {
    auto& engine = get_test_engine();

    layout weight_layout = layout{ov::PartialShape{1, 3, 3, 3}, data_types::f16, format::bfyx};

    auto weights = engine.allocate_memory(weight_layout);
    set_values<ov::float16>(weights, {
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            1.0f, 1.0f, 1.0f,
            //
            2.0f, 2.0f, 2.0f,
            2.0f, 2.0f, 2.0f,
            2.0f, 2.0f, 2.0f,
            //
            3.0f, 3.0f, 3.0f,
            3.0f, 3.0f, 3.0f,
            3.0f, 3.0f, 3.0f,
    });

    layout input_layout_1 = layout{ov::PartialShape{1, 3, 5, 5}, data_types::f32, format::bfyx};
    auto input_mem_1 = engine.allocate_memory(input_layout_1);
    set_values(input_mem_1, {
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         //
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         //
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                         1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                        });

    std::vector<float> ref_output_1 = {6,   18,  36, 54,  72,  54,  30,  12,  36, 72, 108, 144, 108,
                                       60,  18,  54, 108, 162, 216, 162, 90,  18, 54, 108, 162, 216,
                                       162, 90,  18, 54,  108, 162, 216, 162, 90, 12, 36,  72,  108,
                                       144, 108, 60, 6,   18,  36,  54,  72,  54, 30};

    layout input_layout_2 = layout{ov::PartialShape{1, 3, 2, 2}, data_types::f32, format::bfyx};
    auto input_mem_2 = engine.allocate_memory(input_layout_2);
    set_values(input_mem_2, {11.0f,  11.0f, 11.0f, 11.0f,
                             11.0f,  11.0f, 11.0f, 11.0f,
                             11.0f,  11.0f, 11.0f, 11.0f});
    std::vector<float> ref_output_2 = { 66, 132, 132, 66, 132, 264, 264, 132, 132, 264, 264, 132, 66, 132, 132, 66};
     std::vector<float> values_to_subtract = {};
    auto input_l = layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx};
    topology topology(input_layout("input", input_l),
                      data("weights", weights),
                      reorder("reorder", input_info("input"), format::bfyx, data_types::f16,
                      values_to_subtract, reorder_mean_mode::subtract, padding{{0, 0, 2, 2}, 0}),
                      convolution("conv",
                                  input_info("reorder"),
                                  "weights",
                                  "",     /*bias*/
                                  1,
                                  {1, 1}, /*stride*/
                                  {1, 1}, /*dilation*/
                                  {2, 2},  /*pad_above*/
                                  {2, 2},  /*pad_below*/
                                  false,
                                  ov::op::PadType::EXPLICIT),
                      reorder("output", input_info("conv"), format::bfyx, data_types::f32)); /*output padding*/

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"conv", {format::any, "", impl_types::ocl}}}));

    network network(engine, topology, config);
    network.set_input_data("input", input_mem_2);
    auto outputs_1 = network.execute();
    network.set_input_data("input", input_mem_1);
    auto outputs_2 = network.execute();

    // check padding of second run of reorder
    // 0, 0, 0, ... 0,  0, 0,
    // 0, 0, 0, ... 0,  0, 0,
    // 0, 0, 1, ... 5,  0, 0,
    // .  .   .
    // 0, 0, 1, ... 5,  0, 0,
    // 0, 0,"0", .. "0","0","0", // !! check pad_after
    // 0, 0,"0", .. "0","0","0", // !! check pad_after
    auto reorder_mem = network.get_primitive("reorder")->output_memory_ptr();
    cldnn::mem_lock<ov::float16, mem_lock_type::read> reorder_mem_ptr(reorder_mem, get_test_stream());
    for (size_t i = (63 + 81 * 2); i < (71 + 81 * 2); ++i) {
        ASSERT_EQ((float)reorder_mem_ptr[i], 0.f);
    }
    for (size_t i = (72 + 81 * 2); i < (80 + 81 * 2); ++i) {
        ASSERT_EQ((float)reorder_mem_ptr[i], 0.f);
    }
    // Mem should be reallocate when request size is bigger than existing buffer size
    ASSERT_TRUE(reorder_mem->size() <= reorder_mem->get_mem_tracker()->size())
                << "reorder mem buffer size: " <<  reorder_mem->size() << "bytes is bigger than original size of allocated mem: "
                << reorder_mem->get_mem_tracker()->size() << "bytes.";
}
}  // memory_realloc_tests
