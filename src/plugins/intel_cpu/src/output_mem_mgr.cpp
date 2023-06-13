// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "output_mem_mgr.h"

using namespace ov::intel_cpu;

void* OutputMemoryMngr::getRawPtr() const noexcept {
    if (m_allocator) {
        return _data;
    } else {
        return m_pMngr->getRawPtr();
    }
}

void OutputMemoryMngr::setExtBuff(void* ptr, size_t size) {
    IE_ASSERT(!m_allocator);   // FIXME: shouldn't set extbuff when there is an allocator?
    return m_pMngr->setExtBuff(ptr, size);
}

bool OutputMemoryMngr::resize(size_t size) {
    if (m_allocator) {
        // constexpr int cacheLineSize = 64;
        bool sizeChanged = false;
        if (size > _memUpperBound) {
            _data = m_allocator->alloc(size);
            _memUpperBound = size;
            sizeChanged = true;
        }
        return sizeChanged;
    } else {
        return m_pMngr->resize(size);
    }
}

bool OutputMemoryMngr::hasExtBuffer() const noexcept {
    return true;
}

void OutputMemoryMngr::registerMemory(Memory* memPtr) {
    m_pMngr->registerMemory(memPtr);
}

void OutputMemoryMngr::unregisterMemory(Memory* memPtr) {
    m_pMngr->unregisterMemory(memPtr);
}