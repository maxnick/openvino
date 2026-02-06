// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/execution_config.hpp"
#include "layout.hpp"
#include "memory_caps.hpp"
#include "utils.hpp"

#include <vector>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <list>
#include <string>
#include <optional>

namespace cldnn {

struct memory;
struct shared_mem_params;
class engine;
class MemoryTracker;

using primitive_id = std::string;
using memory_ptr = std::shared_ptr<memory>;

template<typename Key, typename Hash = std::hash<Key>, typename KeyEqual = std::equal_to<Key>>
class memory_restricter {
    private:
        const std::unordered_set<Key, Hash, KeyEqual>* set1;  // Const reference to immutable set
        std::unordered_set<Key, Hash, KeyEqual> set2;         // Internal mutable set

    public:
        memory_restricter() : set1(nullptr) {};

        // Constructor to initialize with a const reference for set1
        explicit memory_restricter(const std::unordered_set<Key, Hash, KeyEqual>* externalSet)
            : set1(externalSet) {}

        // Insert into set2 (set1 is read-only)
        void insert(const Key& key) {
            if (set1->find(key) == set1->end())
                set2.insert(key);
        }

        // Check existence in either set
        bool contains(const Key& key) const {
            return set1->find(key) != set1->end() || set2.find(key) != set2.end();
        }

        // Total size of both sets
        size_t size() const {
            return set1->size() + set2.size();
        }

        // Check if both sets are empty
        bool empty() const {
            return set1->empty() && set2.empty();
        }

        // Iterate over both sets
        void for_each(void(*func)(const Key&)) const {
            for (const auto& key : set1) func(key);
            for (const auto& key : set2) func(key);
        }
}; // end of memory_restricter

struct memory_user {
    size_t _unique_id;
    uint32_t _network_id;
    primitive_id _prim_id;
#ifdef GPU_DEBUG_CONFIG
    size_t _mem_size;

    memory_user(size_t unique_id, uint32_t network_id, primitive_id prim_id, size_t mem_size)
        : _unique_id(unique_id), _network_id(network_id), _prim_id(prim_id), _mem_size(mem_size) {}
#endif

    memory_user(size_t unique_id, uint32_t network_id, primitive_id prim_id)
        : _unique_id(unique_id), _network_id(network_id), _prim_id(prim_id) {}

    bool operator==(const struct memory_user& rhs) const {
        return _unique_id == rhs._unique_id && _network_id == rhs._network_id;
    }

    friend std::ostream& operator<<(std::ostream& os, const memory_user& memory_user) {
        os << memory_user._prim_id << " (unique_id:" << memory_user._unique_id;
        os << ", net_id:" << memory_user._network_id << ")";
#ifdef GPU_DEBUG_CONFIG
        os << ", mem_size: " << memory_user._mem_size;
#endif
        return os;
    }
};
struct memory_set_hasher {
    size_t operator()(const memory_user& mem_user) const {
        return hash_combine(0, mem_user._unique_id);
    }
};

using memory_set = std::unordered_set<memory_user, memory_set_hasher>;

struct memory_user_comparer {
    bool operator()(const memory_user& l_mu, const memory_user& r_mu) const {
        if (l_mu._network_id != r_mu._network_id)
            return l_mu._network_id < r_mu._network_id;
        return l_mu._unique_id < r_mu._unique_id;
    }
};

// memory_record represents a single memory block allocation within a region
// Multiple records can share the same underlying GPU memory at different offsets
struct memory_record {
    memory_set _users;          // list of primitives that use this memory block
    memory_ptr _memory;         // the actual memory buffer (subbuffer or root)
    uint32_t _network_id;
    allocation_type _type;
    size_t _offset;             // byte offset within the region (0 for root/full-region blocks)
    size_t _size;               // size of this block in bytes

    // Constructor for memory records
    memory_record(memory_set users, memory_ptr& memory, uint32_t net_id, allocation_type type,
                  size_t offset = 0, size_t size = 0);

    // Check if this block is free (no users)
    bool is_free() const { return _users.empty(); }
};

struct padded_pool_comparer {
    bool operator()(const layout& ll, const layout& rl) const {
        if (ll.format != rl.format)
            return ll.format < rl.format;
        if (ll.data_type != rl.data_type)
            return ll.data_type < rl.data_type;
        if (ll.spatial(0) != rl.spatial(0))
            return ll.spatial(0) < rl.spatial(0);
        if (ll.spatial(1) != rl.spatial(1))
            return ll.spatial(1) < rl.spatial(1);
        return ll.data_padding < rl.data_padding;
    }
};

// memory_pool class implements memory manager that handles 4 memory pools
// - non padded buffers -
//     1 user requests for buffer with no padding.
//     2 Check if buffer with requested size exist
//     3   * yes: check if any of current users exist on request conflict list if no - return this memory, otherwise
//     goto 4
//         * no: goto 4
//     4 take next (allocations are sorted in increasing order) allocation. if there is no more allocations, create new
//     allocation otherwise go t
// - padded buffers - not implemented yet
// - images 2d - not implemented yet
// - images 2d arrays - not implemented yet
// - immutable - if user request for non reusable resource don't use pool, return

// Subblock allocation design:
// - Memory regions are contiguous GPU allocations stored in _non_padded_pool (keyed by size)
// - Each region contains multiple memory_records at different offsets (stored in _records list)
// - Offset search uses single-pass O(k) algorithm: start at offset 0, on conflict jump past conflicting block
// - MemoryTracker* is used as region identifier for O(1) reverse lookup during release
// - Regions are released when all their blocks are freed

// TODO list:
// - Move from runtime to graph part
// - Improve memory consumption

class memory_pool {
public:
    // Type aliases scoped to memory_pool
    using record_list = std::list<memory_record>;
    using record_iterator = record_list::iterator;

    // memory_region represents a contiguous GPU memory allocation
    // Multiple memory_records can be carved from a single region at different offsets
    struct memory_region {
        memory_ptr _memory;                             // actual GPU allocation (root)
        std::map<size_t, record_iterator> _blocks;      // offset -> record iterator

        explicit memory_region(memory_ptr memory) : _memory(std::move(memory)) {}
    };

    using region_map = std::multimap<uint64_t, memory_region>;
    using region_iterator = region_map::iterator;

private:
    memory_ptr alloc_memory(const layout& layout, allocation_type type, bool reset = true);

    // Primary storage: owns all memory_record instances (stable iterators)
    record_list _records;

    // Region pool: size -> memory_region (regions own GPU memory)
    region_map _non_padded_pool;

    // Reverse lookup: MemoryTracker* -> region iterator (for O(1) region lookup during release)
    std::unordered_map<MemoryTracker*, region_iterator> _tracker_to_region;

    std::map<layout, std::list<memory_record>, padded_pool_comparer> _padded_pool;
    engine* _engine;
    const ExecutionConfig& _config;
    float _mem_pool_util_threshold = 0.5f;

public:
    explicit memory_pool(engine& engine, const ExecutionConfig& config);
    ~memory_pool();
    memory_ptr get_memory(const layout& layout,
                          const primitive_id& id,
                          size_t unique_id,
                          uint32_t network_id,
                          const memory_restricter<uint32_t>& restrictions,
                          allocation_type type,
                          bool reusable = true,
                          bool reset = true,
                          bool is_dynamic = false);  // get from pool or create memory allocation
    memory_ptr get_memory(const layout& layout, allocation_type type, bool reset = true);
    memory_ptr get_from_non_padded_pool(const layout& layout,
                                        const primitive_id& prim_id,
                                        size_t unique_id,
                                        uint32_t network_id,
                                        const memory_restricter<uint32_t>&,
                                        allocation_type type,
                                        bool reset = true,
                                        bool is_dynamic = false);
    memory_ptr get_from_padded_pool(const layout& layout,
                                    const primitive_id& prim_id,
                                    size_t unique_id,
                                    uint32_t network_id,
                                    const memory_restricter<uint32_t>& restrictions,
                                    allocation_type type);
    void clear_pool_for_network(uint32_t network_id);
    void release_memory(memory* memory, const size_t& unique_id, primitive_id prim_id, uint32_t network_id);

    size_t get_non_padded_pool_size() {
        return _non_padded_pool.size();
    }

    void dump(uint32_t id, uint32_t iter, std::string dump_dir_path = "");
    size_t get_total_mem_pool_size(allocation_type type);

private:
    void dump_to_screen(uint32_t id, uint32_t iter);
    void dump_to_file(uint32_t id, uint32_t iter, std::string dump_dir_path);

    // Find a valid offset within a region for the requested size
    // Returns nullopt if no valid offset found
    // Uses single-pass O(k) algorithm: iterate blocks, on conflict jump past it
    std::optional<size_t> find_offset_in_region(const memory_region& region,
                                                size_t required_size,
                                                const memory_restricter<uint32_t>& restrictions);

    // Create a new memory region and add first block to it
    memory_ptr create_new_region(const layout& layout,
                                 const primitive_id& prim_id,
                                 size_t unique_id,
                                 uint32_t network_id,
                                 allocation_type type,
                                 bool reset);

    // Add a block to an existing region at the specified offset
    memory_ptr add_block_to_region(region_iterator region_it,
                                   const layout& layout,
                                   const primitive_id& prim_id,
                                   size_t unique_id,
                                   uint32_t network_id,
                                   size_t offset);

    // Remove a block from its region, potentially removing the region if empty
    void remove_block_from_region(region_iterator region_it, size_t offset);

#ifdef GPU_DEBUG_CONFIG
    std::vector<memory_record> _no_reusable_mems;

    float total_mem_size_non_padded_pool        = 0.f;
    float total_mem_size_padded_pool            = 0.f;
    float total_mem_size_no_reusable            = 0.f;
    float mem_size_non_padded_pool_host         = 0.f;
    float mem_size_padded_pool_host             = 0.f;
    float mem_size_no_reusable_host             = 0.f;
#endif
};

}  // namespace cldnn
