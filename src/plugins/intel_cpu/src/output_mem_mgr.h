// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"
#include "openvino/runtime/allocator.hpp"
#include "memory_desc/cpu_memory_desc.h"
#include "ie_allocator.hpp"  // IE public header

IE_SUPPRESS_DEPRECATED_START
namespace InferenceEngine {
class OutputAllocator : public InferenceEngine::IAllocator {
public:
    OutputAllocator() : _data(nullptr, release) {}

    void* lock(void* handle, InferenceEngine::LockOp = InferenceEngine::LOCK_FOR_WRITE) noexcept override {
        return handle;
    }

    void unlock(void* handle) noexcept override {}

    void* alloc(size_t size) noexcept override;

    bool free(void* handle) noexcept override {
        // do nothing, as _data will be deleted along with the instance.
        return true;
    }

    void* getRawPtr() {
        return _data.get();
    }

private:
    std::unique_ptr<void, void (*)(void *)> _data;

    static void release(void *ptr);
    static void destroy(void *ptr);
};

std::shared_ptr<IAllocator> CreateOutputAllocator() noexcept;
}  // namespace InferenceEngine
IE_SUPPRESS_DEPRECATED_END

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

    void setAllocator(std::shared_ptr<InferenceEngine::IAllocator> allocator);

private:
    std::shared_ptr<InferenceEngine::OutputAllocator> m_allocator;
    size_t _memUpperBound = 0ul;

    // We need the default MemMngr as may fallback to copy output... and
    // we have no idea of this in early stages of graph memory allocation.
    MemoryMngrPtr m_pMngr;
};
using OutputMemoryMngrPtr = std::shared_ptr<OutputMemoryMngr>;
using OutputMemoryMngrCPtr = std::shared_ptr<const OutputMemoryMngr>;

}   // namespace intel_cpu
}   // namespace ov