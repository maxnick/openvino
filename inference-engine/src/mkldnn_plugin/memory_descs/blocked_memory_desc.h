// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory_desc.h"

namespace MKLDNNPlugin {

class BlockedMemoryDesc : public virtual MemoryDesc {
public:
    BlockedMemoryDesc() {}

    bool isCompatible(const BlockedMemoryDesc &rhs) const;

    /**
     * @brief Returns the blocked dimensions
     *
     * @return blocked dimensions
     */
    virtual const std::vector<size_t> getBlockDims() const = 0;

    /**
     * @brief Returns the vector of order
     *
     * @return order
     */
    virtual const std::vector<size_t> getOrder() const = 0;

    /**
     * @brief Returns the per-dimension offset vector
     *
     * @return offsets
     */
    virtual const std::vector<size_t> getOffsetPaddingToData() const = 0;

    /**
     * @brief Returns the offset to the current memory block
     *
     * @return offset
     */
    virtual size_t getOffsetPadding() const = 0;

    /**
     * @brief Returns strides for each dimension
     *
     * @return strides
     */
    virtual const std::vector<size_t> getStrides() const = 0;
};

using BlockedMemoryDescPtr = std::unique_ptr<BlockedMemoryDesc>;

} // namespace MKLDNNPlugin
