// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "output_mem_mgr.h"

using namespace ov::intel_cpu;

void* OutputMemoryMngr::getRawPtr() const noexcept {
    if (m_allocator) {
        return _data;
    } else {
        return _pMemMngr->getRawPtr();
    }
}

void OutputMemoryMngr::setExtBuff(void* ptr, size_t size) {
    if (m_allocator) {
        return;
    } else {
        return _pMemMngr->setExtBuff(ptr, size);
    }
}

bool OutputMemoryMngr::resize(size_t size) {
    if (m_allocator) {
        constexpr int cacheLineSize = 64;
        bool sizeChanged = false;
        if (size > _memUpperBound) {
            m_allocator->setMemDesc(m_memDesc);
            _data = m_allocator->allocate(size, cacheLineSize);
            _memUpperBound = size;
            sizeChanged = true;
        }
        return sizeChanged;
    } else {
        return _pMemMngr->resize(size);
    }
}

bool OutputMemoryMngr::hasExtBuffer() const noexcept {
    return true;
}

void OutputMemoryMngr::setMemDesc(MemoryDescPtr desc) {
    m_memDesc = desc;
    return;
}

void* OutputAllocator::allocate(const size_t bytes, const size_t alignment) {
    (void)alignment;
    const auto actualDesc = MemoryDescUtils::convertToTensorDesc(*m_memDesc.get());
    IE_ASSERT(m_memDesc->getCurrentMemSize()==bytes);

    auto &currentDesc = m_blob->getTensorDesc();
    const auto outDims = actualDesc.getDims();
    if (currentDesc.getDims() != outDims) {
        // WA: because input/output info initially contains non empty dims, order etc.
        // and setDims (called inside setShape) can't correct modify blocked desc for desc with blocked layout
        if (currentDesc.getLayout() == InferenceEngine::Layout::BLOCKED) {
            currentDesc = InferenceEngine::TensorDesc(currentDesc.getPrecision(), currentDesc.getLayout());
        }
        m_blob->setShape(outDims);
    }
    return m_blob->buffer();
}

void OutputAllocator::setMemDesc(MemoryDescPtr desc) {
    m_memDesc = desc;
    return;
}