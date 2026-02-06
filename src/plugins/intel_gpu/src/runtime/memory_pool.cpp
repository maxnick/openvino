// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <fstream>
#include <vector>

#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/memory_pool.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

#include <list>
#include <string>
#include <utility>
#include <set>
#include <stdexcept>


#ifdef GPU_DEBUG_CONFIG
#define MEM_USER(uid, nid, pid, cnt) uid, nid, pid, cnt
#else
#define MEM_USER(uid, nid, pid, cnt) uid, nid, pid
#endif
namespace cldnn {

// memory_record implementation
memory_record::memory_record(memory_set users,
                             std::shared_ptr<memory>& memory,
                             uint32_t net_id,
                             allocation_type type,
                             size_t offset,
                             size_t size)
    : _users(std::move(users))
    , _memory(memory)
    , _network_id(net_id)
    , _type(type)
    , _offset(offset)
    , _size(size == 0 ? (memory ? memory->size() : 0) : size) {}

// File-static utility functions for memory pool
static constexpr size_t SUBBLOCK_ALIGNMENT = 64;        // 64-byte alignment for subblock offsets
static constexpr size_t SUBBLOCK_MIN_SIZE = 256;        // Minimum size for subblock allocation
static constexpr size_t MAX_BLOCKS_PER_REGION = 32;     // Maximum blocks per region to prevent pathological cases

static bool is_subblock_sharing_supported(allocation_type type) {
    // Subblock sharing is supported for USM and OpenCL buffer memory types
    // Image formats are not supported
    switch (type) {
        case allocation_type::usm_host:
        case allocation_type::usm_device:
        case allocation_type::usm_shared:
        case allocation_type::cl_mem:
            return true;
        default:
            return false;
    }
}

static size_t align_offset(size_t offset) {
    return (offset + SUBBLOCK_ALIGNMENT - 1) & ~(SUBBLOCK_ALIGNMENT - 1);
}

static bool has_conflict(const memory_set& mem_cand,
                         const memory_restricter<uint32_t>& restrictions) {
    for (const auto& mem_usr : mem_cand) {
        if (restrictions.contains(static_cast<uint32_t>(mem_usr._unique_id)))
            return true;
    }
    return false;
}
memory::ptr memory_pool::alloc_memory(const layout& layout, allocation_type type, bool reset) {
    return _engine->allocate_memory(layout, type, reset);
}

memory_pool::~memory_pool() {}
std::optional<size_t> memory_pool::find_offset_in_region(const memory_region& region,
                                                         size_t required_size,
                                                         const memory_restricter<uint32_t>& restrictions) {
    const size_t region_size = region._memory->size();

    // Edge case: required size larger than region
    if (required_size > region_size) {
        return std::nullopt;
    }

    // Edge case: region too small after alignment
    if (align_offset(0) + required_size > region_size) {
        return std::nullopt;
    }

    size_t offset = 0;
    auto it = region._blocks.begin();

    while (offset + required_size <= region_size) {
        // Skip blocks that end before or at our offset (can't overlap)
        while (it != region._blocks.end()) {
            size_t block_end = it->first + it->second->_size;
            if (block_end <= offset) {
                ++it;
            } else {
                break;
            }
        }

        // No more blocks - we have space
        if (it == region._blocks.end()) {
            return offset;
        }

        // Block starts at or after our range end - found valid spot
        if (it->first >= offset + required_size) {
            return offset;
        }

        // Block overlaps our range - check for conflict
        if (has_conflict(it->second->_users, restrictions)) {
            // Conflict found - jump past this block
            size_t block_end = it->first + it->second->_size;
            offset = align_offset(block_end);
        }
        // Move to next block regardless (non-conflicting blocks don't change)
        ++it;
    }

    return std::nullopt;  // No space found
}

memory_ptr memory_pool::create_new_region(const layout& layout,
                                          const primitive_id& prim_id,
                                          size_t unique_id,
                                          uint32_t network_id,
                                          allocation_type type,
                                          bool reset) {
    const auto layout_bytes_count = layout.bytes_count();

    // Allocate GPU memory
    auto mem = alloc_memory(layout, type, reset);
    if (!mem) {
        return nullptr;
    }

    // Create memory region and insert into pool
    auto region_it = _non_padded_pool.emplace(layout_bytes_count, memory_region(mem));

    // Register in tracker map for O(1) reverse lookup
    MemoryTracker* tracker = mem->get_mem_tracker().get();
    OPENVINO_ASSERT(tracker, "[GPU] Memory tracker is null for newly allocated memory");
    _tracker_to_region[tracker] = region_it;

    // Create first memory record at offset 0
    memory_set users;
    users.insert(memory_user(MEM_USER(unique_id, network_id, prim_id, layout_bytes_count)));
    _records.emplace_back(users, mem, network_id, type, 0, layout_bytes_count);
    auto record_it = std::prev(_records.end());

    // Register block in region
    region_it->second._blocks[0] = record_it;

    GPU_DEBUG_TRACE_DETAIL << "[memory_pool] Created new region for " << prim_id
                           << " (id: " << unique_id << "), size: " << layout_bytes_count << std::endl;

#ifdef GPU_DEBUG_CONFIG
    GPU_DEBUG_IF(_config.get_dump_memory_pool()) {
        total_mem_size_non_padded_pool += layout_bytes_count;
        if (type == allocation_type::usm_host)
            mem_size_non_padded_pool_host += layout_bytes_count;
    }
#endif

    return mem;
}

memory_ptr memory_pool::add_block_to_region(region_iterator region_it,
                                            const layout& layout,
                                            const primitive_id& prim_id,
                                            size_t unique_id,
                                            uint32_t network_id,
                                            size_t offset) {
    auto& region = region_it->second;
    const auto layout_bytes_count = layout.bytes_count();
    const auto type = region._memory->get_allocation_type();

    // Create subbuffer or reinterpret buffer based on offset
    memory_ptr block_mem;
    try {
        if (offset == 0) {
            block_mem = _engine->reinterpret_buffer(*region._memory, layout);
        } else {
            block_mem = _engine->create_subbuffer(*region._memory, layout, offset);
        }
    } catch (...) {
        GPU_DEBUG_TRACE_DETAIL << "[memory_pool] Failed to create subbuffer at offset " << offset << std::endl;
        return nullptr;
    }

    if (!block_mem) {
        return nullptr;
    }

    // Create memory record
    memory_set users;
    users.insert(memory_user(MEM_USER(unique_id, network_id, prim_id, layout_bytes_count)));
    _records.emplace_back(users, block_mem, network_id, type, offset, layout_bytes_count);
    auto record_it = std::prev(_records.end());

    // Register block in region
    region._blocks[offset] = record_it;

    block_mem->from_memory_pool = true;

    GPU_DEBUG_TRACE_DETAIL << "[memory_pool] Added block to region for " << prim_id
                           << " (id: " << unique_id << ") at offset " << offset
                           << ", size: " << layout_bytes_count
                           << ", region size: " << region._memory->size() << std::endl;

    return block_mem;
}

void memory_pool::remove_block_from_region(region_iterator region_it, size_t offset) {
    auto& region = region_it->second;

    // Find and remove the block
    auto block_it = region._blocks.find(offset);
    if (block_it != region._blocks.end()) {
        // Remove from _records list
        _records.erase(block_it->second);
        // Remove from region's block map
        region._blocks.erase(block_it);
    }

    // If region is empty, remove it entirely
    if (region._blocks.empty()) {
        const auto region_size = region._memory->size();
        GPU_DEBUG_TRACE_DETAIL << "[memory_pool] Removing empty region of size " << region_size << std::endl;

#ifdef GPU_DEBUG_CONFIG
        GPU_DEBUG_IF(_config.get_dump_memory_pool()) {
            total_mem_size_non_padded_pool -= region_size;
            if (region._memory->get_allocation_type() == allocation_type::usm_host)
                mem_size_non_padded_pool_host -= region_size;
        }
#endif

        // Remove from tracker map
        MemoryTracker* tracker = region._memory->get_mem_tracker().get();
        if (tracker) {
            _tracker_to_region.erase(tracker);
        }

        // Remove region from pool (this releases the GPU memory when last ref is dropped)
        _non_padded_pool.erase(region_it);
    }
}

void memory_pool::release_memory(memory* mem, const size_t& unique_id, primitive_id prim_id, uint32_t network_id) {
    auto _layout = mem->get_layout();
    if (_layout.is_dynamic()) {
        const auto max_shape = _layout.get_partial_shape().get_max_shape();
        _layout = _layout.clone_with_other_shape(max_shape);
    }
    auto type = mem->get_allocation_type();
    const auto _layout_bytes_count = _layout.bytes_count();

    // Try to find in non-padded pool using MemoryTracker for O(1) region lookup
    MemoryTracker* tracker = mem->get_mem_tracker().get();
    if (tracker) {
        auto tracker_it = _tracker_to_region.find(tracker);
        if (tracker_it != _tracker_to_region.end()) {
            auto region_it = tracker_it->second;
            auto& region = region_it->second;

            // Find the block within this region
            for (auto& [offset, record_it] : region._blocks) {
                if (record_it->_memory.get() == mem ||
                    record_it->_memory->get_internal_params().mem == mem->get_internal_params().mem) {

                    // Remove user from block
                    auto user_it = record_it->_users.find({MEM_USER(unique_id, network_id, prim_id, _layout_bytes_count)});
                    if (user_it != record_it->_users.end()) {
                        record_it->_users.erase(user_it);
                        GPU_DEBUG_TRACE_DETAIL << "[memory_pool] Released user " << prim_id
                                               << " (id: " << unique_id << ") from block at offset "
                                               << offset << ", size " << record_it->_size << std::endl;

                        // If block becomes empty, remove it
                        if (record_it->_users.empty()) {
                            remove_block_from_region(region_it, offset);
                        }
                    }
                    return;
                }
            }
            // Memory has valid tracker pointing to a region, but block not found - this shouldn't happen
            OPENVINO_THROW("[GPU] Memory block not found in region during release. Memory: ", prim_id,
                          " (id: ", unique_id, "), tracker found but block missing.");
        }
    }

    // Check padded pool
    {
        auto itr = _padded_pool.find(_layout);

        if (itr != _padded_pool.end()) {
            auto& list = itr->second;
            auto list_itr = list.begin();

            while (list_itr != list.end()) {
                if (list_itr->_memory.get()->get_internal_params().mem == mem->get_internal_params().mem &&
                    list_itr->_network_id == network_id &&
                    list_itr->_type == type) {
                    auto user_it = list_itr->_users.find({MEM_USER(unique_id, network_id, prim_id, _layout_bytes_count)});

                    if (user_it != list_itr->_users.end()) {
                        user_it = list_itr->_users.erase(user_it);
                    }
                    if (list_itr->_users.empty()) {
#ifdef GPU_DEBUG_CONFIG
                        GPU_DEBUG_IF(_config.get_dump_memory_pool()) {
                            auto released_mem_size = mem->size();
                            total_mem_size_padded_pool -= released_mem_size;
                            if (type == allocation_type::usm_host)
                                mem_size_padded_pool_host -= released_mem_size;
                        }
#endif
                        list.erase(list_itr);
                    }
                    break;
                } else {
                    list_itr++;
                }
            }

            if (list.empty()) {
                _padded_pool.erase(itr);
            }
        }
    }

#ifdef GPU_DEBUG_CONFIG
    GPU_DEBUG_IF(_config.get_dump_memory_pool()) {
        auto iter = std::find_if(_no_reusable_mems.begin(), _no_reusable_mems.end(), [&](const cldnn::memory_record& r) {
            return (network_id == r._network_id
                && type == r._type
                && mem->get_internal_params().mem == r._memory->get_internal_params().mem);
        });
        if (iter != _no_reusable_mems.end()) {
            GPU_DEBUG_IF(_config.get_dump_memory_pool()) {
                auto released_mem_size = iter->_users.begin()->_mem_size;
                total_mem_size_no_reusable -= released_mem_size;
                if (type == allocation_type::usm_host)
                    mem_size_no_reusable_host -= released_mem_size;
            }
            iter->_users.clear();
            _no_reusable_mems.erase(iter);
        }
    }
#endif
}

memory::ptr memory_pool::get_from_non_padded_pool(const layout& layout,
                                                  const primitive_id& prim_id,
                                                  size_t unique_id,
                                                  uint32_t network_id,
                                                  const memory_restricter<uint32_t>& restrictions,
                                                  allocation_type type,
                                                  bool reset,
                                                  bool is_dynamic) {
    const auto layout_bytes_count = layout.bytes_count();

    // Validate: reject zero-size allocations
    if (layout_bytes_count == 0) {
        GPU_DEBUG_TRACE_DETAIL << "[memory_pool] Rejecting zero-size allocation for " << prim_id << std::endl;
        return alloc_memory(layout, type, reset);
    }

    const bool subblock_supported = is_subblock_sharing_supported(type);

    // Search regions by size (smallest suitable first)
    auto it = _non_padded_pool.lower_bound(layout_bytes_count);
    while (it != _non_padded_pool.end()) {
        auto& region = it->second;
        const auto region_size = it->first;

        // Get network_id and type from the region's root memory or first block
        // All blocks in a region share the same network_id and allocation type
        if (region._blocks.empty()) {
            ++it;
            continue;
        }
        auto first_record_it = region._blocks.begin()->second;
        if (first_record_it->_network_id != network_id ||
            region._memory->get_allocation_type() != type) {
            ++it;
            continue;
        }

        // Format compatibility checks
        if (region._memory->get_layout().format == format::fs_b_yx_fsv32 ||
            layout.format == format::fs_b_yx_fsv32 ||
            ((layout.format == format::b_fs_yx_fsv32 || layout.format == format::b_fs_zyx_fsv32) &&
             (layout.feature() % 32 != 0))) {
            ++it;
            continue;
        }

        // Dynamic shape utilization threshold check
        if (is_dynamic && layout_bytes_count <= region_size * _mem_pool_util_threshold) {
            ++it;
            continue;
        }

        // Skip regions that have too many blocks (prevent pathological cases)
        if (region._blocks.size() >= MAX_BLOCKS_PER_REGION) {
            ++it;
            continue;
        }

        // Optimization: if single block fills entire region and has conflict, skip
        if (region._blocks.size() == 1) {
            auto& [offset, record_it] = *region._blocks.begin();
            if (offset == 0 && record_it->_size == region_size &&
                has_conflict(record_it->_users, restrictions)) {
                ++it;
                continue;
            }
        }

        // Try to find existing block we can reuse (exact match at some offset)
        for (auto& [offset, record_it] : region._blocks) {
            if (record_it->_size >= layout_bytes_count &&
                !has_conflict(record_it->_users, restrictions)) {
                // Can reuse this block
                record_it->_users.insert(memory_user(MEM_USER(unique_id, network_id, prim_id, layout_bytes_count)));

                memory::ptr ret_mem;
                if (offset == 0) {
                    ret_mem = _engine->reinterpret_buffer(*region._memory, layout);
                } else {
                    ret_mem = _engine->create_subbuffer(*region._memory, layout, offset);
                }
                ret_mem->from_memory_pool = true;

                GPU_DEBUG_TRACE_DETAIL << "[memory_pool] Reusing block for " << prim_id
                                       << " (id: " << unique_id << "), block size: " << record_it->_size
                                       << ", offset: " << offset << std::endl;
                return ret_mem;
            }
        }

        // Try to allocate a new block at a valid offset within this region
        if (subblock_supported && region_size >= layout_bytes_count + SUBBLOCK_MIN_SIZE) {
            auto offset_opt = find_offset_in_region(region, layout_bytes_count, restrictions);
            if (offset_opt.has_value()) {
                auto ret_mem = add_block_to_region(it, layout, prim_id, unique_id, network_id, offset_opt.value());
                if (ret_mem) {
                    return ret_mem;
                }
            }
        }

        ++it;
    }

    GPU_DEBUG_LOG << "[" << prim_id << "(" << unique_id << "): output]" << std::endl;
    // Didn't find anything suitable - create new region
    return create_new_region(layout, prim_id, unique_id, network_id, type, reset);
}

memory::ptr memory_pool::get_from_padded_pool(const layout& layout,
                                              const primitive_id& prim_id,
                                              size_t unique_id,
                                              uint32_t network_id,
                                              const memory_restricter<uint32_t>& restrictions,
                                              allocation_type type) {
    auto first_level_cache = _padded_pool.find(layout);
    if (first_level_cache != _padded_pool.end()) {
        for (auto& rec_list : first_level_cache->second) {
            if (rec_list._network_id == network_id &&
                rec_list._type == type &&
                ((layout.format != format::b_fs_yx_fsv32 && layout.format != format::b_fs_zyx_fsv32) ||
                 (layout.feature() % 32 == 0)) &&
                layout.feature() <= rec_list._memory->get_layout().feature() &&
                layout.batch() <= rec_list._memory->get_layout().batch() &&
                rec_list._memory->get_layout().format != format::fs_b_yx_fsv32 &&
                layout.format != format::fs_b_yx_fsv32 &&
                !has_conflict(rec_list._users, restrictions)) {
                auto ret_mem = _engine->reinterpret_buffer(*(rec_list._memory), layout);
                rec_list._users.insert({MEM_USER(unique_id, network_id, prim_id, ret_mem->size())});
                ret_mem->from_memory_pool = true;
                return ret_mem;
            }
        }
        auto mem = alloc_memory(layout, type);
        first_level_cache->second.emplace_back(
            memory_record({{MEM_USER(unique_id, network_id, prim_id, mem->size())}}, mem, network_id, type));
#ifdef GPU_DEBUG_CONFIG
        {
            GPU_DEBUG_IF(_config.get_dump_memory_pool()) {
                const auto allocated_mem_size = mem->size();
                total_mem_size_padded_pool += allocated_mem_size;
                if (type == allocation_type::usm_host)
                    mem_size_padded_pool_host += allocated_mem_size;
            }
        }
#endif
        return mem;
    }
    GPU_DEBUG_LOG << "[" << prim_id << "(" << unique_id << ")" << ": output]" << std::endl;
    auto mem = alloc_memory(layout, type);
    std::list<memory_record> list = {memory_record({{MEM_USER(unique_id, network_id, prim_id, mem->size())}}, mem, network_id, type)};
    _padded_pool.emplace(layout, std::move(list));
#ifdef GPU_DEBUG_CONFIG
    {
        GPU_DEBUG_IF(_config.get_dump_memory_pool()) {
            const auto allocated_mem_size = mem->size();
            total_mem_size_padded_pool += allocated_mem_size;
            if (type == allocation_type::usm_host)
                mem_size_padded_pool_host += allocated_mem_size;
        }
    }
#endif
    return mem;
}

memory::ptr memory_pool::get_memory(const layout& layout, allocation_type type, bool reset) {
    return alloc_memory(layout, type, reset);
}

memory::ptr memory_pool::get_memory(const layout& layout,
                                    const primitive_id& prim_id,
                                    const size_t unique_id,
                                    uint32_t network_id,
                                    const memory_restricter<uint32_t>& restrictions,
                                    allocation_type type,
                                    bool reusable_across_network,
                                    bool reset,
                                    bool is_dynamic) {
    bool do_reuse = reusable_across_network;
    GPU_DEBUG_IF(_config.get_disable_memory_reuse()) {
        do_reuse = false;
    }

    if (!do_reuse || layout.format.is_image()) {
        // images (reuse not yet implemented)
        auto mem = alloc_memory(layout, type, reset);
#ifdef GPU_DEBUG_CONFIG
        GPU_DEBUG_IF(_config.get_dump_memory_pool()) {
            auto allocated_mem_size = mem->size();
            _no_reusable_mems.push_back(
                                    memory_record({{MEM_USER(unique_id, network_id, prim_id, allocated_mem_size)}}, mem, network_id, type));
            total_mem_size_no_reusable += allocated_mem_size;
            if (type == allocation_type::usm_host)
                mem_size_no_reusable_host += allocated_mem_size;
        }
#endif
        return mem;
    } else if (!layout.data_padding || is_dynamic) {
        // non-padded buffers. For dynamic shape, use non-padded pool even if it has padding because we will reset the buffer if it is reused
        return get_from_non_padded_pool(layout, prim_id, unique_id, network_id, restrictions, type, reset, is_dynamic);
    } else {
        // padded buffers
        return get_from_padded_pool(layout, prim_id, unique_id, network_id, restrictions, type);
    }
}

void memory_pool::clear_pool_for_network(uint32_t network_id) {
    // Free up _non_padded_pool for this network
    for (auto region_it = _non_padded_pool.begin(); region_it != _non_padded_pool.end(); ) {
        auto& region = region_it->second;

        // Check if region belongs to this network (via first block's network_id)
        bool belongs_to_network = false;
        if (!region._blocks.empty()) {
            belongs_to_network = (region._blocks.begin()->second->_network_id == network_id);
        }

        if (belongs_to_network) {
            const auto region_size = region._memory->size();
#ifdef GPU_DEBUG_CONFIG
            GPU_DEBUG_IF(_config.get_dump_memory_pool()) {
                total_mem_size_non_padded_pool -= region_size;
                if (region._memory->get_allocation_type() == allocation_type::usm_host)
                    mem_size_non_padded_pool_host -= region_size;
            }
#endif
            // Remove all records belonging to this region
            for (auto& [offset, record_it] : region._blocks) {
                _records.erase(record_it);
            }

            // Remove from tracker map
            MemoryTracker* tracker = region._memory->get_mem_tracker().get();
            if (tracker) {
                _tracker_to_region.erase(tracker);
            }

            region_it = _non_padded_pool.erase(region_it);
        } else {
            ++region_it;
        }
    }

    // Free up _padded_pool for this network
    {
        auto itr = _padded_pool.begin();

        while (itr != _padded_pool.end()) {
            auto& list = itr->second;
            auto list_itr = list.begin();
#ifdef GPU_DEBUG_CONFIG
            auto type = list_itr->_type;
#endif
            while (list_itr != list.end()) {
                if (list_itr->_network_id == network_id) {
                    list_itr = list.erase(list_itr);
                } else {
                    list_itr++;
                }
            }

            if (list.empty()) {
#ifdef GPU_DEBUG_CONFIG
                GPU_DEBUG_IF(_config.get_dump_memory_pool()) {
                    auto released_mem_size = itr->first.bytes_count();
                    total_mem_size_padded_pool -= released_mem_size;
                    if (type == allocation_type::usm_host)
                        mem_size_padded_pool_host -= released_mem_size;
                }
#endif
                itr = _padded_pool.erase(itr);
            } else {
                itr++;
            }
        }
    }

#ifdef GPU_DEBUG_CONFIG
    // Free up _no_reusable_mems for this network
    GPU_DEBUG_IF(_config.get_dump_memory_pool()) {
        auto itr = _no_reusable_mems.begin();
        while (itr != _no_reusable_mems.end()) {
            auto& record = *itr;
            if (itr->_network_id == network_id) {
                GPU_DEBUG_IF(_config.get_dump_memory_pool()) {
                    auto released_mem_size = itr->_users.begin()->_mem_size;
                    total_mem_size_no_reusable -= released_mem_size;
                    if (record._type == allocation_type::usm_host)
                        mem_size_no_reusable_host -= released_mem_size;
                }
                itr = _no_reusable_mems.erase(itr);
            } else {
                itr++;
            }
        }
    }
#endif
}

memory_pool::memory_pool(engine& engine, const ExecutionConfig& config) : _engine(&engine), _config(config) {
    _mem_pool_util_threshold = _config.get_mem_pool_util_threshold();
    if (_mem_pool_util_threshold < 0.f || _mem_pool_util_threshold > 1.f) {
        _mem_pool_util_threshold = std::clamp(_mem_pool_util_threshold, 0.f, 1.f);
        GPU_DEBUG_INFO << "[WARNING] mem_pool_util_threshold should be in range [0.f, 1.f]. Reset to "
            << _mem_pool_util_threshold << std::endl;
    }
    GPU_DEBUG_TRACE_DETAIL << "mem_pool_util_threshold set to " << _mem_pool_util_threshold << std::endl;
}

#ifdef GPU_DEBUG_CONFIG
inline std::string get_mb_size(size_t size) {
    if (size == 0)
        return "0 MB";
    return std::to_string(static_cast<float>(size) / (1024 * 1024)) + " MB";
}

inline float get_utilization(size_t size, size_t total_size) {
    return (static_cast<float>(size) * 100.0f / total_size);
}
#endif

size_t memory_pool::get_total_mem_pool_size(allocation_type type) {
#ifdef GPU_DEBUG_CONFIG
    const auto host_mem_size = mem_size_no_reusable_host + mem_size_non_padded_pool_host + mem_size_padded_pool_host;
    const auto total_mem_size = total_mem_size_no_reusable + total_mem_size_non_padded_pool + total_mem_size_padded_pool;
    if (type == allocation_type::usm_host) {
        return host_mem_size;
    } else {
        return (total_mem_size - host_mem_size);
    }
#else
    return 0;
#endif
}

void memory_pool::dump(uint32_t net_id, uint32_t iter, std::string dump_dir_path) {
    dump_to_screen(net_id, iter);
    if (!dump_dir_path.empty())
        dump_to_file(net_id, iter, dump_dir_path);
}

void memory_pool::dump_to_file(uint32_t net_id, uint32_t iter, std::string dump_dir_path) {
#ifdef GPU_DEBUG_CONFIG
    const std::string dump_file_name = "dump_runtime_memory_pool_net_" + std::to_string(net_id) + "_iter_" + std::to_string(iter) + ".csv";
    const std::string desc = "pool_type,layout,mem_ptr,mem_type,region_size,prim_id,unique_id,mem_size,block_offset,is_free";
    const std::string dump_path = dump_dir_path + dump_file_name;
    std::ofstream of(dump_path);
    if (of.is_open()) {
        of << desc << std::endl;

        for (auto& [region_size, region] : _non_padded_pool) {
            const auto region_type = region._memory->get_allocation_type();
            for (auto& [offset, record_it] : region._blocks) {
                const bool is_free = record_it->is_free();

                if (!record_it->_users.empty()) {
                    for (const auto& user : record_it->_users) {
                        of << "non_padded_pool,," << region._memory->buffer_ptr() << "," << region_type << ","
                            << region_size << "," << user._prim_id << "," << user._unique_id << "," << user._mem_size
                            << "," << offset << ",0" << std::endl;
                    }
                } else {
                    of << "non_padded_pool_free,," << region._memory->buffer_ptr() << "," << region_type << ","
                        << region_size << ",free,0,0," << offset << ",1" << std::endl;
                }
            }
        }

        for (auto& mem : _padded_pool) {
            for (auto& record : mem.second) {
                const size_t mem_pool_size = record._memory->size();
                for (const auto& user : record._users) {
                    of << "padded_pool," << mem.first.to_short_string() << "," << record._memory->buffer_ptr() << "," << record._type << ","
                        << mem_pool_size << "," << user._prim_id << "," << user._unique_id << "," << user._mem_size << ",0,0" << std::endl;
                }
            }
        }
        for (auto& mem : _no_reusable_mems) {
            for (const auto& user : mem._users) {
                of << "no_reusable_pool,," << mem._memory->buffer_ptr() << "," << mem._type << ","
                    << user._mem_size << "," << user._prim_id << "," << user._unique_id << "," << user._mem_size << ",0,0" << std::endl;
            }
        }
        std::cout << "Dump file to " << dump_path << std::endl;
    }
#endif
}

void memory_pool::dump_to_screen(uint32_t net_id, uint32_t iter) {
#ifdef GPU_DEBUG_CONFIG
    GPU_DEBUG_COUT << "Dump memory pool of network (net_id : " << net_id << ", iter : " << iter << ")" << std::endl;
    float total_requested_mem_non_padded_pool    = 0.f;
    float total_requested_mem_padded_pool        = 0.f;
    size_t total_subblocks = 0;
    size_t total_free_blocks = 0;

    {
        GPU_DEBUG_COUT << "========== non-padded pool ( " << _non_padded_pool.size() << " regions) ==========" << std::endl;
        for (auto& [region_size, region] : _non_padded_pool) {
            GPU_DEBUG_COUT << "Region: " << region._memory->buffer_ptr() << " (size: " << get_mb_size(region_size)
                << ", type: " << region._memory->get_allocation_type()
                << ", blocks: " << region._blocks.size() << ")" << std::endl;

            for (auto& [offset, record_it] : region._blocks) {
                const bool is_subblock = (offset > 0 || record_it->_size < region_size);
                const bool is_free = record_it->is_free();

                if (is_subblock) {
                    total_subblocks++;
                }
                if (is_free) {
                    total_free_blocks++;
                }

                GPU_DEBUG_COUT << "  Block at offset " << offset
                    << " (size: " << get_mb_size(record_it->_size)
                    << ", free: " << (is_free ? "yes" : "no") << ")" << std::endl;

                if (!record_it->_users.empty()) {
                    float min_utilization = 100.0f;
                    float max_utilization = 0.f;
                    for (const auto& user : record_it->_users) {
                        float utilization = get_utilization(user._mem_size, record_it->_size);
                        min_utilization = std::min(utilization, min_utilization);
                        max_utilization = std::max(utilization, max_utilization);
                        total_requested_mem_non_padded_pool += static_cast<float>(user._mem_size);
                        GPU_DEBUG_COUT << "    --- " << user._prim_id << " (" << user._unique_id << "), "
                            << get_mb_size(user._mem_size) << ", " << utilization << "%" << std::endl;
                    }
                    GPU_DEBUG_COUT <<  "   - min utilization: " << min_utilization << " %" << std::endl;
                    GPU_DEBUG_COUT <<  "   - max utilization: " << max_utilization << " %" << std::endl;
                }
            }
        }
    }

    {
        GPU_DEBUG_COUT << "========== padded pool (" << _padded_pool.size() << " records) ==========" << std::endl;
        for (auto& mem : _padded_pool) {
            GPU_DEBUG_COUT << " layout: " << mem.first.to_short_string() << ", records(" << mem.second.size() << ")" << std::endl;
            for (auto& record : mem.second) {
                size_t mem_size = record._memory->size();
                GPU_DEBUG_COUT << "  " << record._memory->buffer_ptr() << " (size:" << get_mb_size(mem_size)
                                << "MB, type: " << record._type << ")'s users : " << std::endl;
                float min_utilization = 100.0f;
                float max_utilization = 0.f;
                for (const auto& user : record._users) {
                    float utilization = get_utilization(user._mem_size, mem_size);
                    min_utilization = std::min(utilization, min_utilization);
                    max_utilization = std::max(utilization, max_utilization);
                    total_requested_mem_padded_pool += static_cast<float>(user._mem_size);
                    GPU_DEBUG_COUT << "    --- " << user._prim_id << " (" << user._unique_id << "), "
                        << get_mb_size(user._mem_size) << ", " << utilization << "%" << std::endl;
                }
                GPU_DEBUG_COUT << "   - min of the memory pool entry: " << min_utilization << std::endl;
                GPU_DEBUG_COUT << "   - max of the memory pool entry: " << max_utilization << std::endl;
            }
        }
    }

    {
        GPU_DEBUG_COUT << "========== no reusable memory (" << _no_reusable_mems.size() << " records) ==========" << std::endl;
        for (auto& mem : _no_reusable_mems) {
            GPU_DEBUG_COUT << mem._memory->buffer_ptr() << " (type: " << mem._type << ")'s user: " << std::endl;
            for (const auto& user : mem._users) {
                GPU_DEBUG_COUT << "    --- " << user._prim_id << " (" << user._unique_id << "), "
                    << get_mb_size(user._mem_size) << std::endl;
            }
        }
    }

    GPU_DEBUG_COUT << "************************************************************************" << std::endl;
    GPU_DEBUG_COUT << "Memory pool footprint of the network (net_id : " << net_id << ", iter : " << iter << ")" << std::endl;
    GPU_DEBUG_COUT << "Total memory size of non_padded_pool     : " << get_mb_size(total_mem_size_non_padded_pool) << std::endl;
    GPU_DEBUG_COUT << "Total regions                            : " << _non_padded_pool.size() << std::endl;
    GPU_DEBUG_COUT << "Total blocks                             : " << _records.size() << std::endl;
    GPU_DEBUG_COUT << "Total subblock allocations               : " << total_subblocks << std::endl;
    GPU_DEBUG_COUT << "Total free blocks                        : " << total_free_blocks << std::endl;
    if (total_mem_size_non_padded_pool > 0.f) {
        GPU_DEBUG_COUT << " * Efficiency        : "
            << std::to_string(static_cast<float>(total_requested_mem_non_padded_pool / total_mem_size_non_padded_pool))
            << " (total mem requested : " << get_mb_size(total_requested_mem_non_padded_pool)
            << " / total mem pool size : " << get_mb_size(total_mem_size_non_padded_pool) << ")" << std::endl;
        GPU_DEBUG_COUT << " * host mem size     : " << get_mb_size(mem_size_non_padded_pool_host) << std::endl;
        GPU_DEBUG_COUT << " * device mem size   : "
                            << get_mb_size(total_mem_size_non_padded_pool - mem_size_non_padded_pool_host) << std::endl;
    }
    GPU_DEBUG_COUT << "Total memory size of padded_pool memory  : " << get_mb_size(total_mem_size_padded_pool) << std::endl;
    if (total_mem_size_padded_pool > 0.f) {
        GPU_DEBUG_COUT << " * Efficiency        : "
            << std::to_string(static_cast<float>(total_requested_mem_padded_pool / total_mem_size_padded_pool))
            << " (total mem requested : " << get_mb_size(total_requested_mem_padded_pool)
            << " / total mem pool size : " << get_mb_size(total_mem_size_padded_pool) << ")" << std::endl;
        GPU_DEBUG_COUT << " * host mem size     : " << get_mb_size(mem_size_padded_pool_host) << std::endl;
        GPU_DEBUG_COUT << " * device mem size   : " << get_mb_size((total_mem_size_padded_pool - mem_size_padded_pool_host)) << std::endl;
    }
    GPU_DEBUG_COUT << "Total memory size of no reusable memory  : " << get_mb_size(total_mem_size_no_reusable) << std::endl;
    if (total_mem_size_no_reusable > 0.f) {
        GPU_DEBUG_COUT << " * host mem size     : " << get_mb_size(mem_size_no_reusable_host) << std::endl;
        GPU_DEBUG_COUT << " * device mem size   : " << get_mb_size((total_mem_size_no_reusable - mem_size_no_reusable_host)) << std::endl;
    }
    GPU_DEBUG_COUT << "************************************************************************" << std::endl;
#endif
}
}  // namespace cldnn
