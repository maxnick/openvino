// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_blocked_memory_desc.h"
#include "mkldnn_memory.h"

using namespace MKLDNNPlugin;

CpuBlockedMemoryDesc::CpuBlockedMemoryDesc(InferenceEngine::Precision prc, const Shape& shape) : MemoryDesc(shape, CpuBlocked), precision(prc) {
    auto& dims = shape.getDims();
    order.resize(dims.size());
    std::iota(order.begin(), order.end(), 0);
    blockedDims = dims;
    offsetPadding = 0;
    offsetPaddingToData.resize(dims.size(), 0);
    strides.resize(order.size());
    strides[strides.size() - 1] = 1;
    for (size_t i = 2; i <= order.size(); i++) {
        strides[strides.size() - i] = strides[strides.size() - (i - 1)] * blockedDims[blockedDims.size() - (i - 1)];
    }
}

CpuBlockedMemoryDesc::CpuBlockedMemoryDesc(InferenceEngine::Precision prc, const Shape& shape, const std::vector<size_t>& blockedDims,
                  const std::vector<size_t>& order, size_t offsetPadding, const std::vector<size_t>& offsetPaddingToData,
                  const std::vector<size_t>& strides) : MemoryDesc(shape, CpuBlocked), precision(prc) {
    if (std::any_of(order.begin(), order.end(), [](size_t val) { return val == Shape::UNDEFINED_DIM; })) {
        IE_THROW() << "CpuBlockedMemoryDesc do not support undefined order.";
    }

    if (std::any_of(blockedDims.begin() + shape.getRank(), blockedDims.end(), [](size_t val) { return val == Shape::UNDEFINED_DIM; })) {
        IE_THROW() << "CpuBlockedMemoryDesc doesn't support undefined blockedDims.";
    }

    this->order = order;
    this->blockedDims = blockedDims;
    this->offsetPadding = offsetPadding;

    if (offsetPaddingToData.empty() && !order.empty()) {
        this->offsetPaddingToData.resize(order.size());
        this->offsetPaddingToData[order.size() - 1] = 0;
        for (size_t i = 2; i <= order.size(); i++) {
            this->offsetPaddingToData[order.size() - i] = 0;
        }
    } else {
        this->offsetPaddingToData = offsetPaddingToData;
    }

    if (strides.empty() && !order.empty()) {
        if (std::any_of(this->blockedDims.begin(), this->blockedDims.end(), [](size_t val) { return val == Shape::UNDEFINED_DIM; })) {
            this->strides.resize(order.size(), Shape::UNDEFINED_DIM);
        } else {
            this->strides.resize(order.size());
            this->strides[order.size() - 1] = 1;
            for (size_t i = 2; i <= order.size(); i++) {
                this->strides[order.size() - i] = this->strides[order.size() - (i - 1)] * this->blockedDims[blockedDims.size() - (i - 1)];
            }
        }
    } else {
        this->strides = strides;
    }

    if (!everyone_is(this->order.size(), this->blockedDims.size(), this->offsetPaddingToData.size(), this->strides.size())) {
        IE_THROW() << "Order, blocked dims, offset padding to data and strides must have equals size";
    }
}

bool CpuBlockedMemoryDesc::isDefinedImp() const {
    bool defined = true;
    defined = defined && std::none_of(blockedDims.cbegin(), blockedDims.cend(), [](size_t val) { return val == Shape::UNDEFINED_DIM; });
    defined = defined && std::none_of(strides.cbegin(), strides.cend(), [](size_t val) { return val == Shape::UNDEFINED_DIM; });
    defined = defined && std::none_of(order.cbegin(), order.cend(), [](size_t val) { return val == Shape::UNDEFINED_DIM; });
    defined = defined && std::none_of(offsetPaddingToData.cbegin(), offsetPaddingToData.cend(), [](size_t val) { return val == Shape::UNDEFINED_DIM; });
    defined = defined && offsetPadding != Shape::UNDEFINED_DIM;

    return defined;
}

bool CpuBlockedMemoryDesc::isCompatible(const MemoryDesc& rhs) const {
    const MemoryDesc* pRhs = &rhs;
    if (auto blockingDesc = dynamic_cast<const BlockedMemoryDesc*>(pRhs)) {
        return BlockedMemoryDesc::isCompatible(blockingDesc);
    } else {
        return false;
    }
}

size_t CpuBlockedMemoryDesc::getMemSizeImp() const {
    int64_t e_size = getOffsetPadding() + 1;  // size in bytes (from begin of data to last element)
    for (int j = 0; j < getBlockDims().size(); j++)
        e_size += (getBlockDims()[j] - 1) * getStrides()[j];


    e_size *= getPrecision() == InferenceEngine::Precision::BIN ? 1 : getPrecision().size();

    return e_size;
}

size_t CpuBlockedMemoryDesc::getMaxMemSize() const {
    if (shape.isStatic()) {
        return getCurrentSize();
    }

    auto& maxDims = shape.getMaxDims();
    if (std::any_of(maxDims.begin(), maxDims.end(), [](size_t x){ return Shape::UNDEFINED_DIM == x; })) {
        return UNDEFINED_SIZE;
    }

    auto maxDimsDesc = cloneWithNewDims(maxDims);
    return maxDimsDesc->getCurrentSize();
}

size_t CpuBlockedMemoryDesc::getOffset(const InferenceEngine::SizeVector& v) const {
    InferenceEngine::SizeVector off_v = v;

    size_t n_blocked_dims = order.size();
    if (blockedDims.size() != n_blocked_dims || strides.size() != n_blocked_dims) {
        IE_THROW() << "Cannot calculate offset. Incorrect primitive descriptor!";
    }
    InferenceEngine::SizeVector blockedShift(n_blocked_dims);
    for (size_t i = 1; i <= n_blocked_dims; i++) {
        blockedShift[n_blocked_dims - i] = off_v[order[n_blocked_dims - i]] % blockedDims[n_blocked_dims - i];
        off_v[order[n_blocked_dims - i]] /= blockedDims[n_blocked_dims - i];
    }
    size_t offset = getOffsetPadding();
    for (size_t d = 0; d < n_blocked_dims; ++d) {
        const size_t p = blockedShift[d] + getOffsetPaddingToData()[d];
        offset += p * strides[d];
    }
    return offset;
}

size_t CpuBlockedMemoryDesc::getElementOffset(size_t elemNumber) const {
    // TODO [DS]: rewrite to support dynamic shapes
    auto& dims = shape.getStaticDims();
    size_t n_dims = dims.size();
    InferenceEngine::SizeVector pos(n_dims);
    for (size_t rd = 1; rd <= n_dims; ++rd) {
        const size_t d = n_dims - rd;
        const size_t cur_dim = dims[d];
        pos[d] = elemNumber % cur_dim;
        elemNumber /= cur_dim;
    }
    return getOffset(pos);
}

bool CpuBlockedMemoryDesc::hasLayoutType(LayoutType layoutType) const {
    switch (layoutType) {
        case LayoutType::ncsp:
            return isPlainFormat();
        case LayoutType::nspc:
            return isTailCFormat();
        case LayoutType::nCsp8c:
            return isBlockedCFormat(8);
        case LayoutType::nCsp16c:
            return isBlockedCFormat(16);
        default:
            return false;
    }
}

bool CpuBlockedMemoryDesc::isPlainFormat() const {
    if (shape.getRank() != order.size()) {
        return false;
    }
    for (size_t i = 0; i < order.size(); ++i) {
        if (order[i] != i) {
            return false;
        }
    }
    return true;
}

bool CpuBlockedMemoryDesc::isBlockedCFormat(size_t blk_size) const {
    if ((order.size() - shape.getRank()) != 1) {
        return false;
    }
    for (size_t i = 0; i < order.size() - 1; ++i) {
        if (order[i] != i) {
            return false;
        }
    }
    if (order.back() != 1) {
        return false;
    }
    if (blockedDims.back() != blk_size) {
        return false;
    }
    return true;
}

bool CpuBlockedMemoryDesc::isTailCFormat() const {
    if (shape.getRank() < 3) {
        return false;
    }
    if (shape.getRank() != order.size()) {
        return false;
    }
    if (!std::is_sorted(order.begin(), --order.end())) {
        return false;
    }
    if (order.back() != 1) {
        return false;
    }
    return true;
}

std::string CpuBlockedMemoryDesc::serializeFormat() const {
    std::stringstream result;
    char startLetter = 'a';
    std::unordered_map<size_t, size_t> mapAxisBlockSize;
    for (size_t i = shape.getRank(); i < order.size(); ++i) {
        mapAxisBlockSize.insert({order[i], blockedDims[i]});
    }

    for (size_t i = 0; i < shape.getRank(); ++i) {
        char nextLetter = startLetter + order[i];
        if (mapAxisBlockSize.count(i)) {
            nextLetter = toupper(nextLetter);
        }
        result << nextLetter;
    }

    for (auto& item : mapAxisBlockSize) {
        result << item.second << char(startLetter + item.first);
    }

    return result.str();
}

std::unique_ptr<MemoryDesc> CpuBlockedMemoryDesc::cloneWithNewDimsImp(const std::vector<size_t> &dims) const {
    std::vector<size_t> newBlockedDims(order.size());

    for (size_t i = 0; i < dims.size(); ++i) {
        newBlockedDims[order[i]] = dims[i];
    }

    for (size_t i = dims.size(); i < order.size(); ++i) {
        if (newBlockedDims[order[i]] != Shape::UNDEFINED_DIM) {
            newBlockedDims[order[i]] = div_up(newBlockedDims[order[i]], blockedDims[i]);
            newBlockedDims[i] = blockedDims[i];
        }
    }

    std::vector<size_t> newOffsetPaddingToData;
    if (std::none_of(offsetPaddingToData.begin(), offsetPaddingToData.end(), [](size_t x){ return x == Shape::UNDEFINED_DIM;})) {
        newOffsetPaddingToData = offsetPaddingToData;
    }

    return MKLDNNPlugin::make_unique<CpuBlockedMemoryDesc>(precision, Shape(dims), newBlockedDims, order, offsetPadding, newOffsetPaddingToData);
}
