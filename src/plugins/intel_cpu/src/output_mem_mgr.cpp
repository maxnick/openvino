// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "output_mem_mgr.h"
#include "utils/debug_capabilities.h"

using namespace ov::intel_cpu;

namespace InferenceEngine {
IE_SUPPRESS_DEPRECATED_START
std::shared_ptr<IAllocator> CreateOutputAllocator() noexcept {
    try {
        auto data = std::make_shared<OutputAllocator>();
        DEBUG_LOG(data);
        return data;
    } catch (...) {
        return nullptr;
    }
}

void OutputAllocator::release(void *ptr) { DEBUG_LOG(ptr); }
void OutputAllocator::destroy(void *ptr) {
    DEBUG_LOG(ptr);
    if (!ptr) return;

    // dnnl::impl::free(ptr);
    delete[] reinterpret_cast<char*>(ptr);
    ptr = nullptr;
}

void* OutputAllocator::alloc(size_t size) noexcept {
    try {
        // constexpr int cacheLineSize = 64;
        // void *ptr = dnnl::impl::malloc(size, cacheLineSize);
        // if (!ptr) {
        //     IE_THROW() << "Failed to allocate " << size << " bytes of memory";
        // }
        // if (size) {
            auto ptr = reinterpret_cast<void*>(new char[size]);
            _data = decltype(_data)(ptr, destroy);
        // } else {
        //     _data = decltype(_data)(nullptr, release);
        // }

        DEBUG_LOG(_data.get(), "_", size);

        return _data.get();
    } catch (...) {
        return nullptr;
    }
}
}  // namespace InferenceEngine

void OutputMemoryMngr::setAllocator(std::shared_ptr<InferenceEngine::IAllocator> allocator) {
    DEBUG_LOG(allocator, " ", allocator ? std::dynamic_pointer_cast<InferenceEngine::OutputAllocator>(allocator)->getRawPtr() : "null", " this = ", this);

    if (allocator) {
        auto _allocator = std::dynamic_pointer_cast<InferenceEngine::OutputAllocator>(allocator);
        IE_ASSERT(_allocator);
        m_allocator = _allocator;
    } else {
        m_allocator = nullptr;
    }
}

void* OutputMemoryMngr::getRawPtr() const noexcept {
    void *ptr;
    if (m_allocator) {
        ptr = m_allocator->getRawPtr();
    } else {
        ptr = m_pMngr->getRawPtr();
    }

    DEBUG_LOG(m_allocator, " ", ptr, " this = ", this);
    return ptr;
}

void OutputMemoryMngr::setExtBuff(void* ptr, size_t size) {
    DEBUG_LOG(ptr, "_", size, " this = ", this);
    if (m_allocator) {
        IE_THROW() << "Should not call setExtBuff when it is an OutputMemoryMngr with the allocator!";
        return;
    }
    return m_pMngr->setExtBuff(ptr, size);
}

bool OutputMemoryMngr::resize(size_t size) {
    DEBUG_LOG(m_allocator, " ", size, " ", _memUpperBound, " this = ", this);
    if (m_allocator) {
        bool sizeChanged = false;
        if (size > _memUpperBound) {
            auto ptr = m_allocator->alloc(size);
            DEBUG_LOG(ptr, "_", size);
            _memUpperBound = size;
            sizeChanged = true;
        }
        return sizeChanged;
    } else {
        return m_pMngr->resize(size);
    }
}

bool OutputMemoryMngr::hasExtBuffer() const noexcept {
    DEBUG_LOG("", " this = ", this);
    return true;
}

void OutputMemoryMngr::registerMemory(Memory* memPtr) {
    DEBUG_LOG(memPtr->GetData(), " this = ", this);
    m_pMngr->registerMemory(memPtr);
}

void OutputMemoryMngr::unregisterMemory(Memory* memPtr) {
    DEBUG_LOG(memPtr->GetData(), " this = ", this);
    m_pMngr->unregisterMemory(memPtr);
}