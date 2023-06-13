// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"
#include "openvino/runtime/allocator.hpp"
#include "memory_desc/cpu_memory_desc.h"
#include "ie_allocator.hpp"  // IE public header
// #include "blob_allocator.hpp"

namespace ov {
namespace intel_cpu {

class OutputMemoryMngr : public IMemoryMngrObserver {
public:
    explicit OutputMemoryMngr(MemoryMngrPtr pMngr) : m_pMngr(pMngr) {
        IE_ASSERT(m_pMngr) << "Memory manager is uninitialized";
    }

    void* getRawPtr() const noexcept override;
    void setExtBuff(void* ptr, size_t size) override;
    bool resize(size_t size) override;
    bool hasExtBuffer() const noexcept override;

    void registerMemory(Memory* memPtr) override;
    void unregisterMemory(Memory* memPtr) override;

    void setAllocator(std::shared_ptr<InferenceEngine::IAllocator> allocator) { m_allocator = allocator; }

private:
    std::shared_ptr<InferenceEngine::IAllocator> m_allocator;

    size_t _memUpperBound = 0ul;
    void*  _data;
    // We need the default MemMngr as may fallback to copy output... and
    // we have no idea of this in early stages of graph memory allocation.
    MemoryMngrPtr m_pMngr;
};
using OutputMemoryMngrPtr = std::shared_ptr<OutputMemoryMngr>;
using OutputMemoryMngrCPtr = std::shared_ptr<const OutputMemoryMngr>;

}   // namespace intel_cpu
}   // namespace ov