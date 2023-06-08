// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"
#include "openvino/runtime/allocator.hpp"
#include "memory_desc/cpu_memory_desc.h"

namespace ov {
namespace intel_cpu {

class OutputAllocator : public Allocator {
public:
    OutputAllocator(InferenceEngine::Blob::Ptr blob) : m_blob{blob} {}

    void* allocate(const size_t bytes, const size_t alignment = alignof(max_align_t));
    void setMemDesc(MemoryDescPtr desc);
private:
    InferenceEngine::Blob::Ptr m_blob;
    MemoryDescPtr m_memDesc;
};
using OutputAllocatorPtr = std::shared_ptr<OutputAllocator>;
using OutputAllocatorCPtr = std::shared_ptr<const OutputAllocator>;


class OutputMemoryMngr : public IMemoryMngr {
public:
    explicit OutputMemoryMngr(std::unique_ptr<IMemoryMngr> mngr) : _pMemMngr(std::move(mngr)) {}
    // OutputMemoryMngr(OutputAllocatorPtr allocator) : m_allocator{allocator} {}

    ~OutputMemoryMngr() {
        // m_allocator.deallocate(m_ptr, get_byte_size());
    }

    void* getRawPtr() const noexcept override;
    void setExtBuff(void* ptr, size_t size) override;
    bool resize(size_t size) override;
    bool hasExtBuffer() const noexcept override;

    void setAllocator(OutputAllocatorPtr allocator) { m_allocator = allocator; }
    void setMemDesc(MemoryDescPtr desc);

private:
    OutputAllocatorPtr m_allocator;
    MemoryDescPtr m_memDesc;
    size_t _memUpperBound = 0ul;
    void*  _data;
    // We need the default MemMngr as may fallback to copy output... and
    // we have no idea of this in early stages of graph memory allocation.
    std::unique_ptr<IMemoryMngr> _pMemMngr;
};
using OutputMemoryMngrPtr = std::shared_ptr<OutputMemoryMngr>;
using OutputMemoryMngrCPtr = std::shared_ptr<const OutputMemoryMngr>;

}   // namespace intel_cpu
}   // namespace ov