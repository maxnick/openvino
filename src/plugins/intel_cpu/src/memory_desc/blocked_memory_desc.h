// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory_desc.h"

namespace MKLDNNPlugin {

class BlockedMemoryDesc : public virtual MemoryDesc {
public:
    BlockedMemoryDesc() {}

    /**
     * @brief Returns the blocked dimensions
     *
     * @return blocked dimensions
     */
    virtual const VectorDims& getBlockDims() const = 0;

    /**
     * @brief Returns the vector of order
     *
     * @return order
     */
    virtual const VectorDims& getOrder() const = 0;

    /**
     * @brief Returns the per-dimension offset vector
     *
     * @return offsets
     */
    virtual const VectorDims& getOffsetPaddingToData() const = 0;

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
    virtual const VectorDims& getStrides() const = 0;

    /**
     * @brief Check that desc has padded dims
     *
     * @return true if exist padded dims, otherwise false
     */
    virtual bool blocksExtended() const = 0;

    /**
     * @brief Compute number of elements taking into account padded dims
     *
     * @return number of elements taking into account padded dims
     */
    virtual size_t getPaddedElementsCount() const = 0;

    /**
     * @brief Performs masked compatibility check, where the mask defines which strides to check,
     * the most significant bit defines whether to check offset compatibility.
     * @param rhs - desc to compare to
     * @param cmpMask - a bit mask that defines compatibility check rules
     *
     * @return the result of the compatibility check
     */
    virtual bool isCompatible(const BlockedMemoryDesc &rhs, uint32_t cmpMask) const = 0;

    virtual ~BlockedMemoryDesc() = default;

    std::string serializeFormat() const override;

protected:
    /**
     * @brief Check descs on compatibility
     * WARNING: Check only BlockedMemoryDesc specific attributes like: strides, order etc.
     * Doesn't perform type check for descs
     * Doesn't perform descs specific attributes check
     * @return true if compatible, otherwise false
     */
    bool isCompatibleInternal(const BlockedMemoryDesc &rhs, uint32_t compMask = 0xffffffff) const;

    mutable VectorDims blockedDims;
    mutable VectorDims strides;
    mutable VectorDims order;
    mutable VectorDims offsetPaddingToData;
};

using BlockedMemoryDescPtr = std::shared_ptr<BlockedMemoryDesc>;
using BlockedMemoryDescCPtr = std::shared_ptr<const BlockedMemoryDesc>;

} // namespace MKLDNNPlugin
