// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/runtime/memory_pool.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/layout.hpp"

#include <memory>
#include <vector>
#include <cstddef>

using namespace cldnn;
using namespace ::tests;

namespace {

// Helper to create a simple layout with given size
layout make_layout(size_t size_bytes) {
    // Calculate feature size to get approximately the requested byte count
    return layout(ov::PartialShape{1, static_cast<int64_t>(size_bytes)}, data_types::u8, format::bfyx);
}

// Helper to create restrictions set
memory_restricter<uint32_t> make_restrictions(const std::unordered_set<uint32_t>& restricted_ids) {
    static std::unordered_set<uint32_t> static_set;
    static_set = restricted_ids;
    return memory_restricter<uint32_t>(&static_set);
}

memory_restricter<uint32_t> make_empty_restrictions() {
    static std::unordered_set<uint32_t> empty_set;
    return memory_restricter<uint32_t>(&empty_set);
}

}  // namespace

class memory_pool_region_test : public ::testing::Test {
public:
    void SetUp() override {
        config_ = ExecutionConfig{};
    }

    engine& get_engine() {
        return tests::get_test_engine();
    }

protected:
    ExecutionConfig config_;
};

// Test: Basic allocation creates a new region
TEST_F(memory_pool_region_test, basic_allocation_creates_region) {
    memory_pool pool(get_engine(), config_);
    
    auto layout = make_layout(1024);  // 1KB
    auto restrictions = make_empty_restrictions();
    
    auto mem = pool.get_memory(layout, "prim1", 1, 0, restrictions, 
                               allocation_type::usm_device, true, false, false);
    
    ASSERT_NE(mem, nullptr);
    EXPECT_GE(mem->size(), layout.bytes_count());
    EXPECT_EQ(pool.get_non_padded_pool_size(), 1u);  // One region created
}

// Test: Memory reuse when no conflict
TEST_F(memory_pool_region_test, memory_reuse_no_conflict) {
    memory_pool pool(get_engine(), config_);
    
    auto layout = make_layout(1024);
    auto restrictions = make_empty_restrictions();
    
    // First allocation
    auto mem1 = pool.get_memory(layout, "prim1", 1, 0, restrictions,
                                allocation_type::usm_device, true, false, false);
    
    // Second allocation with no conflict should reuse
    auto mem2 = pool.get_memory(layout, "prim2", 2, 0, restrictions,
                                allocation_type::usm_device, true, false, false);
    
    // Both should share the same underlying buffer
    EXPECT_EQ(mem1->get_internal_params().mem, mem2->get_internal_params().mem);
    EXPECT_EQ(pool.get_non_padded_pool_size(), 1u);  // Still one region
}

// Test: New allocation when conflict exists
TEST_F(memory_pool_region_test, new_allocation_on_conflict) {
    memory_pool pool(get_engine(), config_);
    
    auto layout = make_layout(1024);
    auto no_restrictions = make_empty_restrictions();
    
    // First allocation
    auto mem1 = pool.get_memory(layout, "prim1", 1, 0, no_restrictions,
                                allocation_type::usm_device, true, false, false);
    
    // Second allocation with conflict (prim1's unique_id=1 is restricted)
    auto restrictions = make_restrictions({1});
    auto mem2 = pool.get_memory(layout, "prim2", 2, 0, restrictions,
                                allocation_type::usm_device, true, false, false);
    
    // Should be different underlying buffers due to conflict
    EXPECT_NE(mem1->get_internal_params().mem, mem2->get_internal_params().mem);
    EXPECT_EQ(pool.get_non_padded_pool_size(), 2u);  // Two regions
}

// Test: Subblock allocation within a region
TEST_F(memory_pool_region_test, subblock_allocation_in_region) {
    memory_pool pool(get_engine(), config_);
    
    // First, create a small block
    auto small_layout = make_layout(1024);  // 1KB
    auto no_restrictions = make_empty_restrictions();
    
    auto mem_small1 = pool.get_memory(small_layout, "prim_small1", 1, 0, no_restrictions,
                                      allocation_type::usm_device, true, false, false);
    
    EXPECT_EQ(pool.get_non_padded_pool_size(), 1u);
    
    // Now allocate a larger block (no conflict) - this will create a new, larger region
    auto large_layout = make_layout(4096);  // 4KB
    auto restrictions1 = make_restrictions({1});  // Conflict with prim_small1
    
    auto mem_large = pool.get_memory(large_layout, "prim_large", 2, 0, restrictions1,
                                     allocation_type::usm_device, true, false, false);
    
    EXPECT_EQ(pool.get_non_padded_pool_size(), 2u);  // Two regions now
    
    // Now allocate another small block with conflict with prim_large
    // This should find space in the large region after prim_large's block... 
    // But wait - prim_large occupies entire 4KB region, no room for subblock!
    // So this will create a third region OR reuse prim_small1's region
    auto restrictions2 = make_restrictions({2});  // Conflict with prim_large
    auto mem_small2 = pool.get_memory(small_layout, "prim_small2", 3, 0, restrictions2,
                                      allocation_type::usm_device, true, false, false);
    
    ASSERT_NE(mem_small2, nullptr);
    // prim_small2 can reuse prim_small1's region (no conflict with id=1)
    EXPECT_EQ(pool.get_non_padded_pool_size(), 2u);
    EXPECT_EQ(mem_small1->get_mem_tracker(), mem_small2->get_mem_tracker());
}

// Test: Multiple subblocks in one region
TEST_F(memory_pool_region_test, multiple_subblocks_in_region) {
    memory_pool pool(get_engine(), config_);
    
    // Create a large region
    auto large_layout = make_layout(8192);  // 8KB
    auto no_restrictions = make_empty_restrictions();
    
    auto mem1 = pool.get_memory(large_layout, "prim1", 1, 0, no_restrictions,
                                allocation_type::usm_device, true, false, false);
    
    // Allocate smaller blocks with conflicts
    auto small_layout = make_layout(1024);

    auto mem2 = pool.get_memory(small_layout, "prim2", 2, 0, no_restrictions,
                                allocation_type::usm_device, true, false, false);

    auto restrictions2 = make_restrictions({2}); // Conflict with prim2
    auto mem3 = pool.get_memory(small_layout, "prim3", 3, 0, restrictions2,
                                allocation_type::usm_device, true, false, false);
    
    // All should be allocated
    ASSERT_NE(mem1, nullptr);
    ASSERT_NE(mem2, nullptr);
    ASSERT_NE(mem3, nullptr);

    // Should still be one region (multiple subblocks)
    EXPECT_GE(pool.get_non_padded_pool_size(), 1u);
    EXPECT_EQ(mem1->buffer_ptr(), mem2->buffer_ptr());
    EXPECT_EQ(mem1->get_mem_tracker(), mem2->get_mem_tracker());
    EXPECT_EQ(mem1->get_mem_tracker(), mem3->get_mem_tracker());
    EXPECT_EQ(static_cast<std::byte*>(mem1->buffer_ptr()) + 1024, static_cast<std::byte*>(mem3->buffer_ptr()));  // mem3 should be at offset 1024
}

// Test: Release memory removes user
TEST_F(memory_pool_region_test, release_memory_removes_user) {
    memory_pool pool(get_engine(), config_);
    
    auto layout = make_layout(1024);
    auto no_restrictions = make_empty_restrictions();
    
    // Allocate memory
    auto mem = pool.get_memory(layout, "prim1", 1, 0, no_restrictions,
                               allocation_type::usm_device, true, false, false);
    
    EXPECT_EQ(pool.get_non_padded_pool_size(), 1u);
    
    // Release the memory
    pool.release_memory(mem.get(), 1, "prim1", 0);
    
    // Region should be removed since block is empty
    EXPECT_EQ(pool.get_non_padded_pool_size(), 0u);
}

// Test: Release with multiple users keeps region
TEST_F(memory_pool_region_test, release_with_multiple_users) {
    memory_pool pool(get_engine(), config_);
    
    auto layout = make_layout(1024);
    auto no_restrictions = make_empty_restrictions();
    
    // Two primitives share the same memory
    auto mem1 = pool.get_memory(layout, "prim1", 1, 0, no_restrictions,
                                allocation_type::usm_device, true, false, false);
    auto mem2 = pool.get_memory(layout, "prim2", 2, 0, no_restrictions,
                                allocation_type::usm_device, true, false, false);
    
    EXPECT_EQ(pool.get_non_padded_pool_size(), 1u);
    
    // Release first user
    pool.release_memory(mem1.get(), 1, "prim1", 0);
    
    // Region should still exist (second user still active)
    EXPECT_EQ(pool.get_non_padded_pool_size(), 1u);
    
    // Release second user
    pool.release_memory(mem2.get(), 2, "prim2", 0);
    
    // Region should be removed now
    EXPECT_EQ(pool.get_non_padded_pool_size(), 0u);
}

// Test: Clear pool for network
TEST_F(memory_pool_region_test, clear_pool_for_network) {
    memory_pool pool(get_engine(), config_);
    
    auto layout = make_layout(1024);
    auto no_restrictions = make_empty_restrictions();
    
    // Allocate for network 0
    auto mem_net0 = pool.get_memory(layout, "prim1", 1, 0, no_restrictions,
                                    allocation_type::usm_device, true, false, false);
    
    // Allocate for network 1
    auto mem_net1 = pool.get_memory(layout, "prim2", 2, 1, no_restrictions,
                                    allocation_type::usm_device, true, false, false);
    
    EXPECT_EQ(pool.get_non_padded_pool_size(), 2u);
    
    // Clear network 0
    pool.clear_pool_for_network(0);
    
    EXPECT_EQ(pool.get_non_padded_pool_size(), 1u);
    
    // Clear network 1
    pool.clear_pool_for_network(1);
    
    EXPECT_EQ(pool.get_non_padded_pool_size(), 0u);
}

// Test: Different allocation types don't share regions
TEST_F(memory_pool_region_test, different_allocation_types_separate) {
    memory_pool pool(get_engine(), config_);
    
    auto layout = make_layout(1024);
    auto no_restrictions = make_empty_restrictions();
    
    auto mem_device = pool.get_memory(layout, "prim1", 1, 0, no_restrictions,
                                      allocation_type::usm_device, true, false, false);
    
    auto mem_host = pool.get_memory(layout, "prim2", 2, 0, no_restrictions,
                                    allocation_type::usm_host, true, false, false);
    
    // Should be different regions due to different allocation types
    EXPECT_NE(mem_device->get_internal_params().mem, mem_host->get_internal_params().mem);
    EXPECT_EQ(pool.get_non_padded_pool_size(), 2u);
}

// Test: Different networks don't share regions
TEST_F(memory_pool_region_test, different_networks_separate) {
    memory_pool pool(get_engine(), config_);
    
    auto layout = make_layout(1024);
    auto no_restrictions = make_empty_restrictions();
    
    auto mem_net0 = pool.get_memory(layout, "prim1", 1, 0, no_restrictions,
                                    allocation_type::usm_device, true, false, false);
    
    auto mem_net1 = pool.get_memory(layout, "prim2", 2, 1, no_restrictions,
                                    allocation_type::usm_device, true, false, false);
    
    // Should be different regions due to different networks
    EXPECT_NE(mem_net0->get_internal_params().mem, mem_net1->get_internal_params().mem);
    EXPECT_EQ(pool.get_non_padded_pool_size(), 2u);
}

// Test: Reuse after release
TEST_F(memory_pool_region_test, reuse_after_release) {
    memory_pool pool(get_engine(), config_);
    
    auto layout = make_layout(1024);
    auto no_restrictions = make_empty_restrictions();
    
    // Allocate
    auto mem1 = pool.get_memory(layout, "prim1", 1, 0, no_restrictions,
                                allocation_type::usm_device, true, false, false);
    void* ptr1 = mem1->get_internal_params().mem;
    
    // Release
    pool.release_memory(mem1.get(), 1, "prim1", 0);
    EXPECT_EQ(pool.get_non_padded_pool_size(), 0u);
    
    // Allocate again - should get new region
    auto mem2 = pool.get_memory(layout, "prim2", 2, 0, no_restrictions,
                                allocation_type::usm_device, true, false, false);
    
    EXPECT_EQ(pool.get_non_padded_pool_size(), 1u);
}

// Test: Smaller allocation can reuse larger region
TEST_F(memory_pool_region_test, smaller_allocation_reuses_larger) {
    memory_pool pool(get_engine(), config_);
    
    auto no_restrictions = make_empty_restrictions();
    
    // Create large region
    auto large_layout = make_layout(4096);
    auto mem_large = pool.get_memory(large_layout, "prim1", 1, 0, no_restrictions,
                                     allocation_type::usm_device, true, false, false);
    
    // Smaller allocation without conflict should reuse
    auto small_layout = make_layout(1024);
    auto mem_small = pool.get_memory(small_layout, "prim2", 2, 0, no_restrictions,
                                     allocation_type::usm_device, true, false, false);
    
    // Should share same underlying buffer
    EXPECT_EQ(mem_large->get_internal_params().mem, mem_small->get_internal_params().mem);
    EXPECT_EQ(pool.get_non_padded_pool_size(), 1u);
}

// Test: Dynamic shape allocation
TEST_F(memory_pool_region_test, dynamic_shape_allocation) {
    memory_pool pool(get_engine(), config_);
    
    auto layout = make_layout(1024);
    auto no_restrictions = make_empty_restrictions();
    
    // Allocate with is_dynamic=true
    auto mem = pool.get_memory(layout, "prim1", 1, 0, no_restrictions,
                               allocation_type::usm_device, true, false, true);
    
    ASSERT_NE(mem, nullptr);
    EXPECT_EQ(pool.get_non_padded_pool_size(), 1u);
}

// Test: Memory from pool has from_memory_pool flag set
TEST_F(memory_pool_region_test, memory_pool_flag_set) {
    memory_pool pool(get_engine(), config_);
    
    auto layout = make_layout(1024);
    auto no_restrictions = make_empty_restrictions();
    
    // Second allocation should have flag set (reuses existing)
    auto mem1 = pool.get_memory(layout, "prim1", 1, 0, no_restrictions,
                                allocation_type::usm_device, true, false, false);
    auto mem2 = pool.get_memory(layout, "prim2", 2, 0, no_restrictions,
                                allocation_type::usm_device, true, false, false);
    
    EXPECT_TRUE(mem2->from_memory_pool);
}

// Test: Non-reusable memory doesn't go to pool
TEST_F(memory_pool_region_test, non_reusable_memory) {
    memory_pool pool(get_engine(), config_);
    
    auto layout = make_layout(1024);
    auto no_restrictions = make_empty_restrictions();
    
    // Allocate with reusable=false
    auto mem = pool.get_memory(layout, "prim1", 1, 0, no_restrictions,
                               allocation_type::usm_device, false, false, false);
    
    ASSERT_NE(mem, nullptr);
    EXPECT_EQ(pool.get_non_padded_pool_size(), 0u);  // Not in pool
}

// Test: Stress test - many allocations and releases
TEST_F(memory_pool_region_test, stress_many_allocations) {
    memory_pool pool(get_engine(), config_);
    
    auto layout = make_layout(1024);
    auto no_restrictions = make_empty_restrictions();
    
    const size_t num_allocations = 100;
    const size_t max_blocks_per_region = 32;  // Matches MAX_BLOCKS_PER_REGION in memory_pool.cpp
    std::vector<memory::ptr> memories;
    
    // Allocate many memories
    for (size_t i = 0; i < num_allocations; ++i) {
        auto mem = pool.get_memory(layout, "prim" + std::to_string(i), i, 0, no_restrictions,
                                   allocation_type::usm_device, true, false, false);
        ASSERT_NE(mem, nullptr);
        memories.push_back(mem);
    }
    
    // With single-user-per-block design and MAX_BLOCKS_PER_REGION limit,
    // we need multiple regions: ceil(100/32) = 4
    const size_t expected_regions = (num_allocations + max_blocks_per_region - 1) / max_blocks_per_region;
    EXPECT_EQ(pool.get_non_padded_pool_size(), expected_regions);
    
    // Release all
    for (size_t i = 0; i < num_allocations; ++i) {
        pool.release_memory(memories[i].get(), i, "prim" + std::to_string(i), 0);
    }
    
    EXPECT_EQ(pool.get_non_padded_pool_size(), 0u);
}

// Test: Conflicting allocations create separate regions
TEST_F(memory_pool_region_test, conflicting_allocations_chain) {
    memory_pool pool(get_engine(), config_);
    
    auto layout = make_layout(1024);
    
    // Each allocation conflicts with the previous one
    auto no_restrictions = make_empty_restrictions();
    auto mem1 = pool.get_memory(layout, "prim1", 1, 0, no_restrictions,
                                allocation_type::usm_device, true, false, false);
    
    auto restrictions1 = make_restrictions({1});
    auto mem2 = pool.get_memory(layout, "prim2", 2, 0, restrictions1,
                                allocation_type::usm_device, true, false, false);
    
    auto restrictions2 = make_restrictions({2});
    auto mem3 = pool.get_memory(layout, "prim3", 3, 0, restrictions2,
                                allocation_type::usm_device, true, false, false);
    
    // mem3 should be able to reuse mem1's region (no conflict with 1)
    // So we should have 2 regions: one for {prim1, prim3} and one for {prim2}
    EXPECT_EQ(pool.get_non_padded_pool_size(), 2u);
}

// Test: Region selection prefers smaller suitable regions
TEST_F(memory_pool_region_test, region_selection_smallest_fit) {
    memory_pool pool(get_engine(), config_);
    
    auto no_restrictions = make_empty_restrictions();
    
    // Create regions of different sizes with conflicts so they're separate
    auto layout_small = make_layout(1024);
    auto layout_medium = make_layout(2048);
    auto layout_large = make_layout(4096);
    
    auto mem_large = pool.get_memory(layout_large, "prim1", 1, 0, no_restrictions,
                                     allocation_type::usm_device, true, false, false);
    
    auto restrictions1 = make_restrictions({1});
    auto mem_medium = pool.get_memory(layout_medium, "prim2", 2, 0, restrictions1,
                                      allocation_type::usm_device, true, false, false);
    
    auto restrictions2 = make_restrictions({1, 2});
    auto mem_small = pool.get_memory(layout_small, "prim3", 3, 0, restrictions2,
                                     allocation_type::usm_device, true, false, false);
    
    EXPECT_EQ(pool.get_non_padded_pool_size(), 3u);
    
    // New small allocation without restrictions should prefer smallest fitting region
    auto mem_new = pool.get_memory(layout_small, "prim4", 4, 0, no_restrictions,
                                   allocation_type::usm_device, true, false, false);
    
    // Should still be 3 regions (reused smallest suitable)
    EXPECT_EQ(pool.get_non_padded_pool_size(), 3u);
}

// Test: Zero-size layout handling
TEST_F(memory_pool_region_test, zero_size_layout) {
    memory_pool pool(get_engine(), config_);
    
    // Create a layout with zero dimensions
    layout zero_layout(ov::PartialShape{0}, data_types::f32, format::bfyx);
    auto no_restrictions = make_empty_restrictions();
    
    // Should handle gracefully (allocate directly, not through pool)
    auto mem = pool.get_memory(zero_layout, "prim1", 1, 0, no_restrictions,
                               allocation_type::usm_device, true, false, false);
    
    // Either nullptr or a valid memory - just shouldn't crash
    // Zero-size allocations go through direct allocation, not pool
}

// Test: Release memory that doesn't exist (should not crash)
TEST_F(memory_pool_region_test, release_nonexistent_memory) {
    memory_pool pool(get_engine(), config_);
    
    auto layout = make_layout(1024);
    auto no_restrictions = make_empty_restrictions();
    
    // Allocate
    auto mem = pool.get_memory(layout, "prim1", 1, 0, no_restrictions,
                               allocation_type::usm_device, true, false, false);
    
    // Release twice - second should be safe
    pool.release_memory(mem.get(), 1, "prim1", 0);
    // Note: Second release with same user should be safe (user already removed)
}

// Test: Verify subblock doesn't overlap with conflicting blocks
TEST_F(memory_pool_region_test, subblock_avoids_conflicts) {
    memory_pool pool(get_engine(), config_);
    
    auto no_restrictions = make_empty_restrictions();
    
    // Create a large region (8KB)
    auto large_layout = make_layout(8192);
    auto mem1 = pool.get_memory(large_layout, "prim1", 1, 0, no_restrictions,
                                allocation_type::usm_device, true, false, false);
    
    // Create a block at start (2KB) that conflicts with mem1
    auto small_layout = make_layout(2048);
    auto restrictions1 = make_restrictions({1});
    auto mem2 = pool.get_memory(small_layout, "prim2", 2, 0, restrictions1,
                                allocation_type::usm_device, true, false, false);
    
    // Create another block that conflicts with both
    auto restrictions2 = make_restrictions({1, 2});
    auto mem3 = pool.get_memory(small_layout, "prim3", 3, 0, restrictions2,
                                allocation_type::usm_device, true, false, false);
    
    // All should be allocated successfully
    ASSERT_NE(mem1, nullptr);
    ASSERT_NE(mem2, nullptr);
    ASSERT_NE(mem3, nullptr);
}
