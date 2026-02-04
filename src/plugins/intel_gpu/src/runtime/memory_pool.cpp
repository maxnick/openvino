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
#include "ocl/ocl_engine.hpp"

#include <list>
#include <string>
#include <sstream>
#include <iomanip>
#include <utility>
#include <set>
#include <stdexcept>


#ifdef GPU_DEBUG_CONFIG
#define MEM_USER(uid, nid, pid, cnt) uid, nid, pid, cnt
#define MEMPOOL_SUBALLOC_LOG(msg) \
    GPU_DEBUG_IF(_config.get_dump_memory_pool()) { \
        GPU_DEBUG_TRACE_DETAIL << "MEMPOOL_SUBALLOC: " << msg << std::endl; \
    }
#else
#define MEM_USER(uid, nid, pid, cnt) uid, nid, pid
#define MEMPOOL_SUBALLOC_LOG(msg)
#endif
namespace cldnn {
memory_record::memory_record(memory_set users,
                             std::shared_ptr<memory>& memory,
                             uint32_t net_id,
                             allocation_type type)
    : _users(users), _memory(memory), _network_id(net_id), _type(type) {
    _total_bytes = _memory ? _memory->size() : 0;
    _total_users = _users.size();
}

memory::ptr memory_pool::alloc_memory(const layout& layout, allocation_type type, bool reset) {
    return _engine->allocate_memory(layout, type, reset);
}

memory_pool::~memory_pool() {}

bool memory_pool::is_suballocation_allowed(allocation_type type) const {
    if (_engine->runtime_type() != runtime_types::ocl)
        return false;
    if (memory_capabilities::is_usm_type(type))
        return _engine->use_unified_shared_memory();
    return type == allocation_type::cl_mem;
}

size_t memory_pool::get_ocl_sub_buffer_alignment() const {
    constexpr size_t default_alignment = 64;
    if (_engine->type() == engine_types::ocl) {
        if (auto* ocl_engine_ptr = dynamic_cast<const cldnn::ocl::ocl_engine*>(_engine)) {
            try {
                const auto align_bits = ocl_engine_ptr->get_cl_device().getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>();
                const auto align_bytes = std::max<size_t>(1, static_cast<size_t>(align_bits) / 8);
                return std::max<size_t>(default_alignment, align_bytes);
            } catch (...) {
                return default_alignment;
            }
        }
    }
    return default_alignment;
}

size_t memory_pool::get_sub_buffer_alignment(const layout& layout, allocation_type type) const {
    constexpr size_t default_alignment = 64;
    const auto element_size = data_type_traits::size_of(layout.data_type);
    auto alignment = std::max<size_t>(default_alignment, element_size);
    if (type == allocation_type::cl_mem)
        alignment = std::max(alignment, get_ocl_sub_buffer_alignment());
    return alignment;
}

size_t memory_pool::align_sub_buffer_size(const layout& layout, allocation_type type) const {
    const auto alignment = get_sub_buffer_alignment(layout, type);
    return align_to(layout.bytes_count(), alignment);
}

void memory_pool::insert_non_padded_segment(const non_padded_pool_iter& record_it,
                                            const segment_iter& segment_it) {
    _non_padded_segments.emplace(segment_it->size, non_padded_segment_ref{record_it, segment_it});
}

void memory_pool::erase_non_padded_segment(const non_padded_pool_iter& record_it,
                                           const segment_iter& segment_it,
                                           size_t size) {
    auto range = _non_padded_segments.equal_range(size);
    for (auto it = range.first; it != range.second; ++it) {
        if (it->second.record_it == record_it && it->second.segment_it == segment_it) {
            _non_padded_segments.erase(it);
            return;
        }
    }
}

void memory_pool::erase_non_padded_record(const non_padded_pool_iter& record_it) {
    for (auto seg_it = record_it->second._segments.begin(); seg_it != record_it->second._segments.end(); ++seg_it) {
        erase_non_padded_segment(record_it, seg_it, seg_it->size);
    }
    if (record_it->second._memory && record_it->second._memory->get_mem_tracker()) {
        _non_padded_tracker_map.erase(record_it->second._memory->get_mem_tracker().get());
    }
}

bool memory_pool::has_conflict(const memory_set& mem_cand,
                               const memory_restricter<uint32_t>& restrictions) {
    for (const auto& mem_usr : mem_cand) {
        if (restrictions.contains(static_cast<uint32_t>(mem_usr._unique_id)))
            return true;
    }
    return false;
}

void memory_pool::release_memory(memory* mem, const size_t& unique_id, primitive_id prim_id, uint32_t network_id) {
    // check non padded pool first
    auto _layout = mem->get_layout();
    if (_layout.is_dynamic()) {
        const auto max_shape = _layout.get_partial_shape().get_max_shape();
        _layout = _layout.clone_with_other_shape(max_shape);
    }
    auto type = mem->get_allocation_type();
    const auto _layout_bytes_count = _layout.bytes_count();

    if (is_suballocation_allowed(type)) {
        auto tracker = mem->get_mem_tracker();
        if (tracker) {
            auto map_it = _non_padded_tracker_map.find(tracker.get());
            if (map_it != _non_padded_tracker_map.end()) {
                auto record_it = map_it->second;
                auto& record = record_it->second;
                if (record._network_id == network_id && record._type == type) {
                    const memory_user user_key{MEM_USER(unique_id, network_id, prim_id, _layout_bytes_count)};
                    auto record_user_it = record._users.find(user_key);
                    bool removed = false;

                    for (auto seg_it = record._segments.begin(); seg_it != record._segments.end(); ++seg_it) {
                        auto user_it = seg_it->users.find(user_key);
                        if (user_it != seg_it->users.end()) {
                            seg_it->users.erase(user_it);
                            if (record_user_it != record._users.end())
                                record._users.erase(record_user_it);
                            if (record._total_users > 0)
                                record._total_users--;
                            removed = true;

                            if (record._total_users == 0) {
#ifdef GPU_DEBUG_CONFIG
                                GPU_DEBUG_IF(_config.get_dump_memory_pool()) {
                                    auto released_mem_size = record_it->first;
                                    total_mem_size_non_padded_pool -= released_mem_size;
                                    if (type == allocation_type::usm_host)
                                        mem_size_non_padded_pool_host -= released_mem_size;
                                }
#endif
                                erase_non_padded_record(record_it);
                                _non_padded_pool.erase(record_it);
                                return;
                            }

                            if (seg_it->users.empty()) {
                                // merge with previous free segment
                                if (seg_it != record._segments.begin()) {
                                    auto prev_it = std::prev(seg_it);
                                    if (prev_it->users.empty() && (prev_it->offset + prev_it->size == seg_it->offset)) {
                                        erase_non_padded_segment(record_it, prev_it, prev_it->size);
                                        erase_non_padded_segment(record_it, seg_it, seg_it->size);
                                        prev_it->size += seg_it->size;
                                        seg_it = record._segments.erase(seg_it);
                                        insert_non_padded_segment(record_it, prev_it);
                                        seg_it = prev_it;
                                    }
                                }

                                // merge with next free segment
                                auto next_it = std::next(seg_it);
                                if (next_it != record._segments.end() && next_it->users.empty() &&
                                    (seg_it->offset + seg_it->size == next_it->offset)) {
                                    erase_non_padded_segment(record_it, seg_it, seg_it->size);
                                    erase_non_padded_segment(record_it, next_it, next_it->size);
                                    seg_it->size += next_it->size;
                                    record._segments.erase(next_it);
                                    insert_non_padded_segment(record_it, seg_it);
                                }
                            }
                            break;
                        }
                    }

                    if (removed)
                        return;
                }
            }
        }
    }

    {
        auto it = _non_padded_pool.lower_bound(_layout_bytes_count);

        while (it != _non_padded_pool.end()) {
            if (it->second._network_id == network_id &&
                it->second._type == type &&
                it->second._memory->get_internal_params().mem == mem->get_internal_params().mem) {
                auto user_it = it->second._users.find({MEM_USER(unique_id, network_id, prim_id, _layout_bytes_count)});
                // normally there should be only one entry
                if (user_it != it->second._users.end()) {
                    user_it = it->second._users.erase(user_it);
                    if (it->second._total_users > 0)
                        it->second._total_users--;
                }
                if (it->second._users.empty()) {
#ifdef GPU_DEBUG_CONFIG
                    GPU_DEBUG_IF(_config.get_dump_memory_pool()) {
                        auto released_mem_size = it->first;
                        total_mem_size_non_padded_pool -= released_mem_size;
                        if (type == allocation_type::usm_host)
                            mem_size_non_padded_pool_host -= released_mem_size;
                    }
#endif
                    // if this was the only user of the memory, then free it up
                    it = _non_padded_pool.erase(it);
                }

                //entry found and processed - so return
                return;
            } else {
                ++it;
            }
        }
    }
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

                    // normally there should be only one entry
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
                        // if this was the only user of the memory, then free it up
                        list.erase(list_itr);
                    }

                    //entry found and processed - so return
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
    auto can_reuse = [&](const memory_record& record,
                         const memory_record::memory_segment& segment,
                         size_t request_size) {
        const auto candidate_size = segment.size;

        if (is_dynamic && !(request_size > candidate_size * _mem_pool_util_threshold))
            return false;
        if (record._network_id != network_id)
            return false;
        if (record._type != type)
            return false;
        if (record._memory->get_layout().format == format::fs_b_yx_fsv32 || layout.format == format::fs_b_yx_fsv32)
            return false;
        if ((layout.format == format::b_fs_yx_fsv32 || layout.format == format::b_fs_zyx_fsv32) &&
            (layout.feature() % 32 != 0))
            return false;
        if (has_conflict(segment.users, restrictions))
            return false;
        return true;
    };
    if (is_suballocation_allowed(type)) {
        const auto aligned_bytes_count = align_sub_buffer_size(layout, type);
        auto seg_it = _non_padded_segments.lower_bound(aligned_bytes_count);
        while (seg_it != _non_padded_segments.end()) {
            auto record_it = seg_it->second.record_it;
            auto& record = record_it->second;
            auto& segment = *seg_it->second.segment_it;

            if (can_reuse(record,
                          segment,
                          aligned_bytes_count)) {
                const memory_user user_key{MEM_USER(unique_id, network_id, prim_id, layout_bytes_count)};
                const auto segment_offset = segment.offset;
#ifdef GPU_DEBUG_CONFIG
                const auto mem_ptr_value = reinterpret_cast<uintptr_t>(record._memory->buffer_ptr());
                std::ostringstream mem_ptr_hex;
                mem_ptr_hex << std::hex << mem_ptr_value;
                const auto mem_ptr_hex_str = mem_ptr_hex.str();
#endif
                memory::ptr ret_mem;
                try {
                    ret_mem = _engine->create_subbuffer(*record._memory, layout, segment_offset);
                } catch (...) {
                    MEMPOOL_SUBALLOC_LOG("create_subbuffer failed, skip segment: req=" + std::to_string(aligned_bytes_count) +
                                         " seg=" + std::to_string(segment.size) + " offset=" + std::to_string(segment_offset));
                    ++seg_it;
                    continue;
                }

                if (segment.size > aligned_bytes_count) {
                    auto segment_it = seg_it->second.segment_it;
                    erase_non_padded_segment(record_it, segment_it, segment.size);
                    const auto original_size = segment.size;
                    const auto original_users = segment.users;
                    segment.size = aligned_bytes_count;
                    segment.users.insert(user_key);
                    record._users.insert(user_key);
                    record._total_users++;

                    memory_record::memory_segment remainder;
                    remainder.offset = segment.offset + aligned_bytes_count;
                    remainder.size = original_size - aligned_bytes_count;
                    remainder.users = original_users;
                    auto remainder_it = record._segments.insert(std::next(segment_it), std::move(remainder));

                    insert_non_padded_segment(record_it, segment_it);
                    insert_non_padded_segment(record_it, remainder_it);
                    MEMPOOL_SUBALLOC_LOG("split segment: req=" + std::to_string(aligned_bytes_count) +
                                         " used=" + std::to_string(segment.size) +
                                         " rem=" + std::to_string(remainder_it->size) +
                                         " offset=" + std::to_string(segment.offset) +
                                         " shared_users=" + std::to_string(original_users.size()) +
                                         " mem_ptr=0x" + mem_ptr_hex_str);
                } else {
                    segment.users.insert(user_key);
                    record._users.insert(user_key);
                    record._total_users++;
                    MEMPOOL_SUBALLOC_LOG("reuse segment: req=" + std::to_string(aligned_bytes_count) +
                                         " seg=" + std::to_string(segment.size) +
                                         " offset=" + std::to_string(segment.offset) +
                                         " mem_ptr=0x" + mem_ptr_hex_str);
                }
                ret_mem->from_memory_pool = true;
                return ret_mem;
            }
            ++seg_it;
        }
    } else {
        auto it = _non_padded_pool.lower_bound(layout_bytes_count);
        while (it != _non_padded_pool.end()) {
            memory_record::memory_segment record_segment;
            record_segment.size = it->second._memory->get_layout().bytes_count();
            record_segment.users = it->second._users;
            if (can_reuse(it->second,
                          record_segment,
                          layout_bytes_count)) {
                it->second._users.insert(memory_user(MEM_USER(unique_id, network_id, prim_id, layout_bytes_count)));
                it->second._total_users++;
                auto ret_mem = _engine->reinterpret_buffer(*it->second._memory, layout);
                ret_mem->from_memory_pool = true;
                return ret_mem;
            } else {
                ++it;
            }
        }
    }
    GPU_DEBUG_LOG << "[" << prim_id << "(" << unique_id << "): output]" << std::endl;
    // didn't find anything for you? create new resource
    auto mem = alloc_memory(layout, type, reset);
    {
        auto record_it = _non_padded_pool.emplace(layout_bytes_count,
                                                  memory_record({{MEM_USER(unique_id, network_id, prim_id, layout_bytes_count)}}, mem, network_id, type));
        if (is_suballocation_allowed(type) && mem->get_mem_tracker()) {
            auto& record = record_it->second;
            record._segments.clear();
            record._total_bytes = record_it->first;
            record._total_users = record._users.size();
            memory_record::memory_segment seg;
            seg.offset = 0;
            seg.size = align_sub_buffer_size(layout, type);
            seg.users = record._users;
            record._segments.push_back(seg);
            insert_non_padded_segment(record_it, record._segments.begin());
            _non_padded_tracker_map.emplace(mem->get_mem_tracker().get(), record_it);
        }
#ifdef GPU_DEBUG_CONFIG
        {
            GPU_DEBUG_IF(_config.get_dump_memory_pool()) {
                total_mem_size_non_padded_pool += layout_bytes_count;
                if (type == allocation_type::usm_host)
                    mem_size_non_padded_pool_host += layout_bytes_count;
            }
        }
#endif
    }
    return mem;
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
                // TODO: check if this condition always correct
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
    // free up _non_padded_pool for this network
    {
        auto itr = _non_padded_pool.begin();

        while (itr != _non_padded_pool.end()) {
            auto& record = itr->second;

            if (record._network_id == network_id) {
#ifdef GPU_DEBUG_CONFIG
                GPU_DEBUG_IF(_config.get_dump_memory_pool()) {
                    auto released_mem_size = itr->first;
                    total_mem_size_non_padded_pool -= released_mem_size;
                    if (record._type == allocation_type::usm_host)
                        mem_size_non_padded_pool_host -= released_mem_size;
                }
#endif
                if (is_suballocation_allowed(record._type)) {
                    erase_non_padded_record(itr);
                }
                itr = _non_padded_pool.erase(itr);
            } else {
                itr++;
            }
        }
    }

    // free up _padded_pool for this network
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
    // free up _no_reusable_mems for this network
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
    const std::string desc = "pool_type,layout,mem_ptr,mem_type,mem_pool_size,prim_id,unique_id,mem_size";
    const std::string dump_path = dump_dir_path + dump_file_name;
    std::ofstream of(dump_path);
    if (of.is_open()) {
        of << desc << std::endl;
        for (auto mem : _non_padded_pool) {
            for (auto user : mem.second._users) {
                of << "non_padded_pool,," << mem.second._memory->buffer_ptr() << "," << mem.second._type << ","
                    << mem.first << "," << user._prim_id << "," << user._unique_id << "," << user._mem_size << std::endl;
            }
        }

        for (auto mem : _padded_pool) {
            for (auto record : mem.second) {
                const size_t mem_pool_size = record._memory->size();
                for (auto user : record._users) {
                    of << "padded_pool," << mem.first.to_short_string() << "," << record._memory->buffer_ptr() << "," << record._type << ","
                        << mem_pool_size << "," << user._prim_id << "," << user._unique_id << "," << user._mem_size << std::endl;
                }
            }
        }
        for (auto mem : _no_reusable_mems) {
            for (auto user : mem._users) {
                of << "no_reusable_pool,," << mem._memory->buffer_ptr() << "," << mem._type << ","
                    << user._mem_size << "," << user._prim_id << "," << user._unique_id << "," << user._mem_size << std::endl;
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

    {
        GPU_DEBUG_COUT << "========== non-padded pool ( " << _non_padded_pool.size() << " records) ==========" << std::endl;
        for (auto mem : _non_padded_pool) {
            GPU_DEBUG_COUT << mem.second._memory->buffer_ptr() << " (size: " << get_mb_size(mem.first)
                << ", type: " << mem.second._type << ")'s users: " << std::endl;
            float min_utilization = 100.0f;
            float max_utilization = 0.f;
            for (auto user : mem.second._users) {
                float utilization = get_utilization(user._mem_size, mem.first);
                min_utilization = std::min(utilization, min_utilization);
                max_utilization = std::max(utilization, max_utilization);
                total_requested_mem_non_padded_pool += static_cast<float>(user._mem_size);
                GPU_DEBUG_COUT << "    --- " << user._prim_id << " (" << user._unique_id << "), "
                    << get_mb_size(user._mem_size) << ", " << utilization << "%" << std::endl;
            }
            GPU_DEBUG_COUT <<  "   - min utilization of the memory pool entry: " << min_utilization << " %" << std::endl;
            GPU_DEBUG_COUT <<  "   - max utilization of the memory pool entry: " << max_utilization << " %" << std::endl;
        }
    }

    {
        GPU_DEBUG_COUT << "========== padded pool (" << _padded_pool.size() << " records) ==========" << std::endl;
        for (auto mem : _padded_pool) {
            GPU_DEBUG_COUT << " layout: " << mem.first.to_short_string() << ", records(" << mem.second.size() << ")" << std::endl;
            for (auto record : mem.second) {
                size_t mem_size = record._memory->size();
                GPU_DEBUG_COUT << "  " << record._memory->buffer_ptr() << " (size:" << get_mb_size(mem_size)
                                << "MB, type: " << record._type << ")'s users : " << std::endl;
                float min_utilization = 100.0f;
                float max_utilization = 0.f;
                for (auto user : record._users) {
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
        for (auto mem : _no_reusable_mems) {
            GPU_DEBUG_COUT << mem._memory->buffer_ptr() << " (type: " << mem._type << ")'s user: " << std::endl;
            for (auto user : mem._users) {
                GPU_DEBUG_COUT << "    --- " << user._prim_id << " (" << user._unique_id << "), "
                    << get_mb_size(user._mem_size) << std::endl;
            }
        }
    }

    GPU_DEBUG_COUT << "************************************************************************" << std::endl;
    GPU_DEBUG_COUT << "Memory pool footprint of the network (net_id : " << net_id << ", iter : " << iter << ")" << std::endl;
    GPU_DEBUG_COUT << "Total memory size of non_padded_pool     : " << get_mb_size(total_mem_size_non_padded_pool) << std::endl;
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
