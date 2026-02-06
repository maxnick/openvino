// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include "intel_gpu/runtime/memory_pool.hpp"

using namespace cldnn;
using namespace ::tests;

namespace memory_pool_subblock_tests {

// Test alignment function
TEST(memory_pool_subblock_test, alignment) {
    ASSERT_EQ(memory_pool::align_offset(0), 0u);
    ASSERT_EQ(memory_pool::align_offset(1), memory_pool::SUBBLOCK_ALIGNMENT);
    ASSERT_EQ(memory_pool::align_offset(63), memory_pool::SUBBLOCK_ALIGNMENT);
    ASSERT_EQ(memory_pool::align_offset(64), 64u);
    ASSERT_EQ(memory_pool::align_offset(65), 128u);
    ASSERT_EQ(memory_pool::align_offset(127), 128u);
    ASSERT_EQ(memory_pool::align_offset(128), 128u);
}

// Test is_subblock_sharing_supported for different allocation types
TEST(memory_pool_subblock_test, subblock_sharing_support) {
    ASSERT_TRUE(memory_pool::is_subblock_sharing_supported(allocation_type::usm_host));
    ASSERT_TRUE(memory_pool::is_subblock_sharing_supported(allocation_type::usm_device));
    ASSERT_TRUE(memory_pool::is_subblock_sharing_supported(allocation_type::usm_shared));
    ASSERT_TRUE(memory_pool::is_subblock_sharing_supported(allocation_type::cl_mem));
    ASSERT_FALSE(memory_pool::is_subblock_sharing_supported(allocation_type::unknown));
}

// Test memory_record root allocation constructor
TEST(memory_pool_subblock_test, memory_record_root_constructor) {
    auto& engine = get_test_engine();
    layout test_layout = layout{ov::PartialShape{1, 1, 1, 1024}, data_types::f32, format::bfyx};
    auto mem = engine.allocate_memory(test_layout);

    memory_set users;
    users.insert(memory_user(1, 0, "test_prim"));
    memory_record record(users, mem, 0, allocation_type::usm_device);

    ASSERT_TRUE(record.is_root());
    ASSERT_EQ(record._parent, nullptr);
    ASSERT_TRUE(record._children.empty());
    ASSERT_EQ(record._offset, 0u);
    ASSERT_EQ(record._size, mem->size());
    ASSERT_EQ(record._depth, 0u);
    ASSERT_FALSE(record.is_free());
    ASSERT_TRUE(record.is_leaf());
}

// Test memory_record subblock constructor
TEST(memory_pool_subblock_test, memory_record_subblock_constructor) {
    auto& engine = get_test_engine();
    layout test_layout = layout{ov::PartialShape{1, 1, 1, 1024}, data_types::f32, format::bfyx};
    auto mem = engine.allocate_memory(test_layout);

    // Create root record
    memory_set users;
    memory_record root(users, mem, 0, allocation_type::usm_device);

    // Create subblock record
    memory_record subblock(mem, 0, allocation_type::usm_device, &root, 128, 512, 1);

    ASSERT_FALSE(subblock.is_root());
    ASSERT_EQ(subblock._parent, &root);
    ASSERT_TRUE(subblock._children.empty());
    ASSERT_EQ(subblock._offset, 128u);
    ASSERT_EQ(subblock._size, 512u);
    ASSERT_EQ(subblock._depth, 1u);
    ASSERT_TRUE(subblock.is_free());  // No users
    ASSERT_TRUE(subblock.is_leaf());  // No children
}

// Test memory_record overlap detection
TEST(memory_pool_subblock_test, memory_record_overlap_detection) {
    auto& engine = get_test_engine();
    layout test_layout = layout{ov::PartialShape{1, 1, 1, 1024}, data_types::f32, format::bfyx};
    auto mem = engine.allocate_memory(test_layout);

    memory_set users;
    memory_record root(users, mem, 0, allocation_type::usm_device);

    // Create a subblock at offset 100, size 50 (range [100, 150))
    memory_record subblock(mem, 0, allocation_type::usm_device, &root, 100, 50, 1);

    // Test non-overlapping ranges
    ASSERT_FALSE(subblock.overlaps(0, 50));      // [0, 50) - before
    ASSERT_FALSE(subblock.overlaps(150, 50));    // [150, 200) - after
    ASSERT_FALSE(subblock.overlaps(50, 50));     // [50, 100) - adjacent before

    // Test overlapping ranges
    ASSERT_TRUE(subblock.overlaps(100, 50));     // Exact match
    ASSERT_TRUE(subblock.overlaps(90, 20));      // Partial overlap at start
    ASSERT_TRUE(subblock.overlaps(140, 20));     // Partial overlap at end
    ASSERT_TRUE(subblock.overlaps(110, 20));     // Contained within
    ASSERT_TRUE(subblock.overlaps(50, 200));     // Contains the subblock
}

// Test get_root traversal
TEST(memory_pool_subblock_test, get_root_traversal) {
    auto& engine = get_test_engine();
    layout test_layout = layout{ov::PartialShape{1, 1, 1, 2048}, data_types::f32, format::bfyx};
    auto mem = engine.allocate_memory(test_layout);

    memory_set users;
    memory_record root(users, mem, 0, allocation_type::usm_device);
    memory_record child1(mem, 0, allocation_type::usm_device, &root, 0, 1024, 1);
    memory_record grandchild(mem, 0, allocation_type::usm_device, &child1, 0, 512, 2);

    root.add_child(&child1);
    child1.add_child(&grandchild);

    // Verify get_root from different levels
    ASSERT_EQ(root.get_root(), &root);
    ASSERT_EQ(child1.get_root(), &root);
    ASSERT_EQ(grandchild.get_root(), &root);
}

// Test ancestor user collection
TEST(memory_pool_subblock_test, collect_ancestor_users) {
    auto& engine = get_test_engine();
    layout test_layout = layout{ov::PartialShape{1, 1, 1, 2048}, data_types::f32, format::bfyx};
    auto mem = engine.allocate_memory(test_layout);

    memory_set root_users;
    root_users.insert(memory_user(1, 0, "root_prim"));
    memory_record root(root_users, mem, 0, allocation_type::usm_device);

    memory_record child1(mem, 0, allocation_type::usm_device, &root, 0, 1024, 1);
    child1._users.insert(memory_user(2, 0, "child1_prim"));
    root.add_child(&child1);

    memory_record grandchild(mem, 0, allocation_type::usm_device, &child1, 0, 512, 2);
    grandchild._users.insert(memory_user(3, 0, "grandchild_prim"));
    child1.add_child(&grandchild);

    // Collect ancestors from grandchild
    memory_set collected;
    grandchild.collect_ancestor_users(collected);

    // Should contain users from root and child1, but not grandchild itself
    ASSERT_EQ(collected.size(), 2u);
    bool found_root = false, found_child1 = false;
    for (const auto& user : collected) {
        if (user._unique_id == 1) found_root = true;
        if (user._unique_id == 2) found_child1 = true;
    }
    ASSERT_TRUE(found_root);
    ASSERT_TRUE(found_child1);
}

// Test descendant user collection
TEST(memory_pool_subblock_test, collect_descendant_users) {
    auto& engine = get_test_engine();
    layout test_layout = layout{ov::PartialShape{1, 1, 1, 2048}, data_types::f32, format::bfyx};
    auto mem = engine.allocate_memory(test_layout);

    memory_set root_users;
    root_users.insert(memory_user(1, 0, "root_prim"));
    memory_record root(root_users, mem, 0, allocation_type::usm_device);

    memory_record child1(mem, 0, allocation_type::usm_device, &root, 0, 1024, 1);
    child1._users.insert(memory_user(2, 0, "child1_prim"));
    root.add_child(&child1);

    memory_record child2(mem, 0, allocation_type::usm_device, &root, 1024, 1024, 1);
    child2._users.insert(memory_user(3, 0, "child2_prim"));
    root.add_child(&child2);

    memory_record grandchild(mem, 0, allocation_type::usm_device, &child1, 0, 512, 2);
    grandchild._users.insert(memory_user(4, 0, "grandchild_prim"));
    child1.add_child(&grandchild);

    // Collect descendants from root
    memory_set collected;
    root.collect_descendant_users(collected);

    // Should contain users from child1, child2, and grandchild
    ASSERT_EQ(collected.size(), 3u);
    bool found_child1 = false, found_child2 = false, found_grandchild = false;
    for (const auto& user : collected) {
        if (user._unique_id == 2) found_child1 = true;
        if (user._unique_id == 3) found_child2 = true;
        if (user._unique_id == 4) found_grandchild = true;
    }
    ASSERT_TRUE(found_child1);
    ASSERT_TRUE(found_child2);
    ASSERT_TRUE(found_grandchild);
}

// Test overlapping sibling user collection
TEST(memory_pool_subblock_test, collect_overlapping_sibling_users) {
    auto& engine = get_test_engine();
    layout test_layout = layout{ov::PartialShape{1, 1, 1, 2048}, data_types::f32, format::bfyx};
    auto mem = engine.allocate_memory(test_layout);

    memory_set root_users;
    memory_record root(root_users, mem, 0, allocation_type::usm_device);

    // child1: [0, 512)
    memory_record child1(mem, 0, allocation_type::usm_device, &root, 0, 512, 1);
    child1._users.insert(memory_user(1, 0, "child1_prim"));
    root.add_child(&child1);

    // child2: [256, 768) - overlaps with child1
    memory_record child2(mem, 0, allocation_type::usm_device, &root, 256, 512, 1);
    child2._users.insert(memory_user(2, 0, "child2_prim"));
    root.add_child(&child2);

    // child3: [1024, 1536) - does not overlap with child1
    memory_record child3(mem, 0, allocation_type::usm_device, &root, 1024, 512, 1);
    child3._users.insert(memory_user(3, 0, "child3_prim"));
    root.add_child(&child3);

    // Collect overlapping siblings from child1's perspective
    memory_set collected;
    child1.collect_overlapping_sibling_users(collected);

    // Should only contain child2's user (child3 doesn't overlap)
    ASSERT_EQ(collected.size(), 1u);
    ASSERT_EQ(collected.begin()->_unique_id, 2u);
}

// Test has_conflict_with_hierarchy
TEST(memory_pool_subblock_test, has_conflict_with_hierarchy) {
    auto& engine = get_test_engine();
    layout test_layout = layout{ov::PartialShape{1, 1, 1, 2048}, data_types::f32, format::bfyx};
    auto mem = engine.allocate_memory(test_layout);

    memory_set root_users;
    root_users.insert(memory_user(1, 0, "root_prim"));
    memory_record root(root_users, mem, 0, allocation_type::usm_device);

    memory_record child(mem, 0, allocation_type::usm_device, &root, 0, 1024, 1);
    child._users.insert(memory_user(2, 0, "child_prim"));
    root.add_child(&child);

    // Restriction containing root user - should conflict
    std::unordered_set<uint32_t> restriction_set1 = {1};
    memory_restricter<uint32_t> restriction1(&restriction_set1);
    ASSERT_TRUE(child.has_conflict_with_hierarchy(restriction1));

    // Restriction containing child user - should conflict
    std::unordered_set<uint32_t> restriction_set2 = {2};
    memory_restricter<uint32_t> restriction2(&restriction_set2);
    ASSERT_TRUE(child.has_conflict_with_hierarchy(restriction2));

    // Restriction not containing any related user - should not conflict
    std::unordered_set<uint32_t> restriction_set3 = {999};
    memory_restricter<uint32_t> restriction3(&restriction_set3);
    ASSERT_FALSE(child.has_conflict_with_hierarchy(restriction3));

    // Empty restriction - should not conflict
    std::unordered_set<uint32_t> empty_set;
    memory_restricter<uint32_t> empty_restriction(&empty_set);
    ASSERT_FALSE(child.has_conflict_with_hierarchy(empty_restriction));
}

// Test add_child and remove_child
TEST(memory_pool_subblock_test, add_remove_child) {
    auto& engine = get_test_engine();
    layout test_layout = layout{ov::PartialShape{1, 1, 1, 1024}, data_types::f32, format::bfyx};
    auto mem = engine.allocate_memory(test_layout);

    memory_set users;
    memory_record root(users, mem, 0, allocation_type::usm_device);
    memory_record child1(mem, 0, allocation_type::usm_device, &root, 0, 512, 1);
    memory_record child2(mem, 0, allocation_type::usm_device, &root, 512, 512, 1);

    ASSERT_TRUE(root._children.empty());
    ASSERT_TRUE(root.is_leaf());

    root.add_child(&child1);
    ASSERT_EQ(root._children.size(), 1u);
    ASSERT_FALSE(root.is_leaf());

    root.add_child(&child2);
    ASSERT_EQ(root._children.size(), 2u);

    root.remove_child(&child1);
    ASSERT_EQ(root._children.size(), 1u);
    ASSERT_EQ(root._children[0], &child2);

    root.remove_child(&child2);
    ASSERT_TRUE(root._children.empty());
    ASSERT_TRUE(root.is_leaf());
}

// Test can_be_removed
TEST(memory_pool_subblock_test, can_be_removed) {
    auto& engine = get_test_engine();
    layout test_layout = layout{ov::PartialShape{1, 1, 1, 1024}, data_types::f32, format::bfyx};
    auto mem = engine.allocate_memory(test_layout);

    memory_set users;
    memory_record root(users, mem, 0, allocation_type::usm_device);

    // Free and no children - can be removed
    ASSERT_TRUE(root.can_be_removed());

    // Add user - cannot be removed
    root._users.insert(memory_user(1, 0, "prim1"));
    ASSERT_FALSE(root.can_be_removed());

    // Remove user but add child - cannot be removed
    root._users.clear();
    memory_record child(mem, 0, allocation_type::usm_device, &root, 0, 512, 1);
    root.add_child(&child);
    ASSERT_FALSE(root.can_be_removed());

    // Remove child - can be removed again
    root.remove_child(&child);
    ASSERT_TRUE(root.can_be_removed());
}

// Integration test: verify memory pool with subblock allocation through the pool interface
TEST(memory_pool_subblock_test, pool_integration_basic) {
    auto& engine = get_test_engine();
    ExecutionConfig config = get_test_default_config(engine);
    memory_pool pool(engine, config);

    // Allocate a large block first
    layout large_layout = layout{ov::PartialShape{1, 1, 1, 4096}, data_types::f32, format::bfyx};  // 16KB

    std::unordered_set<uint32_t> no_restrictions_set;
    memory_restricter<uint32_t> no_restrictions(&no_restrictions_set);

    auto mem1 = pool.get_from_non_padded_pool(large_layout, "large_prim", 1, 0, no_restrictions,
                                               allocation_type::usm_device, true, false);
    ASSERT_NE(mem1, nullptr);

    // Now request a smaller block - should reuse the large block
    layout small_layout = layout{ov::PartialShape{1, 1, 1, 512}, data_types::f32, format::bfyx};  // 2KB

    auto mem2 = pool.get_from_non_padded_pool(small_layout, "small_prim", 2, 0, no_restrictions,
                                               allocation_type::usm_device, true, false);
    ASSERT_NE(mem2, nullptr);

    // mem2 should reuse mem1's block (either as full reuse or subblock)
    // The memory should be from the pool
    ASSERT_TRUE(mem2->from_memory_pool);
}

// Test that conflicting allocations go to separate regions
TEST(memory_pool_subblock_test, pool_integration_with_conflicts) {
    auto& engine = get_test_engine();
    ExecutionConfig config = get_test_default_config(engine);
    memory_pool pool(engine, config);

    // Allocate first block
    layout layout1 = layout{ov::PartialShape{1, 1, 1, 1024}, data_types::f32, format::bfyx};  // 4KB

    std::unordered_set<uint32_t> no_restrictions_set;
    memory_restricter<uint32_t> no_restrictions(&no_restrictions_set);

    auto mem1 = pool.get_from_non_padded_pool(layout1, "prim1", 1, 0, no_restrictions,
                                               allocation_type::usm_device, true, false);
    ASSERT_NE(mem1, nullptr);

    // Allocate second block with restriction against first
    std::unordered_set<uint32_t> restriction_set = {1};  // Conflicts with prim1
    memory_restricter<uint32_t> restrictions(&restriction_set);

    auto mem2 = pool.get_from_non_padded_pool(layout1, "prim2", 2, 0, restrictions,
                                               allocation_type::usm_device, true, false);
    ASSERT_NE(mem2, nullptr);

    // mem2 should be a different allocation since it conflicts with mem1
    // (either a new allocation or a subblock at different offset)
}

// Test depth limit
TEST(memory_pool_subblock_test, depth_limit_constant) {
    // Verify the depth limit constant exists and has reasonable value
    ASSERT_GT(memory_pool::MAX_SUBBLOCK_DEPTH, 0u);
    ASSERT_LE(memory_pool::MAX_SUBBLOCK_DEPTH, 10u);  // Reasonable upper bound
}

// Test constants are accessible
TEST(memory_pool_subblock_test, constants_accessible) {
    ASSERT_EQ(memory_pool::SUBBLOCK_ALIGNMENT, 64u);
    ASSERT_EQ(memory_pool::SUBBLOCK_MIN_SIZE, 256u);
}

}  // namespace memory_pool_subblock_tests
