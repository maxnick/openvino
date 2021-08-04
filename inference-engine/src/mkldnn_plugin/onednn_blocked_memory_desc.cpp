// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "onednn_blocked_memory_desc.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

OnednnBlockedMemoryDesc::OnednnBlockedMemoryDesc(InferenceEngine::Precision prc, const Shape& shape, const std::vector<size_t>& blockedDims,
                                                 const std::vector<size_t>& order, size_t offsetPadding, const std::vector<size_t>& offsetPaddingToData,
                                                 const std::vector<size_t>& strides)
                : MKLDNNMemoryDesc(prc, shape, blockedDims, order, offsetPadding, offsetPaddingToData, strides), MemoryDesc(shape, OneDnnBlocked) {}

const std::vector<size_t> OnednnBlockedMemoryDesc::getBlockDims() const {
    const auto dims = desc.dims();

    const auto &blk_desc = desc.data.format_desc.blocking;

    const size_t outer_ndims = dims.size();
    const size_t inner_ndims = blk_desc.inner_nblks;
    const size_t total_ndims = outer_ndims + inner_ndims;

    // total inner block size. in case of 4i16o4i will be {16, 16, 1, 1}
    std::vector<size_t> total_block_per_dim(outer_ndims, 1);
    for (int i = 0; i < inner_ndims; i++) {
        total_block_per_dim[blk_desc.inner_idxs[i]] *= blk_desc.inner_blks[i];
    }
    // blocked dims
    // [dims via new_outer_order with auto pad] U [inner_blk_dims]
    std::vector<size_t> outer_block_dims = MKLDNNExtensionUtils::convertToSizeVector(dims);
    for (size_t i = 0; i < outer_block_dims.size(); i++) {
        if (outer_block_dims[i] != Shape::UNDEFINED_DIM) {
            outer_block_dims[i] = div_up(outer_block_dims[i], total_block_per_dim[i]);
        }
    }

    // order of outer dims. In case of IOhw_ will be {1, 0, 2, 3}
    std::vector<size_t> outer_order(outer_ndims);
    std::copy(order.begin(), order.begin() + outer_ndims, outer_order.begin());

    SizeVector blk_dims(total_ndims, 0);
    std::copy(blk_desc.inner_blks, blk_desc.inner_blks + blk_desc.inner_nblks,
              blk_dims.end() - blk_desc.inner_nblks);
    std::transform(outer_order.begin(), outer_order.end(), blk_dims.begin(),
                   [&] (size_t i) { return outer_block_dims[i]; });
    return blk_dims;
}

const std::vector<size_t> OnednnBlockedMemoryDesc::getStrides() const {
    const auto dims = desc.dims();

    const auto &blk_desc = desc.data.format_desc.blocking;

    const size_t outer_ndims = dims.size();
    const size_t inner_ndims = blk_desc.inner_nblks;
    const size_t total_ndims = outer_ndims + inner_ndims;

    // strides of inner dims. In case of 4i16o4i will be {64, 4, 1}
    std::vector<size_t> inner_strides(inner_ndims, 1);
    for (size_t i = 1; i < blk_desc.inner_nblks; i++) {
        inner_strides[blk_desc.inner_nblks - 1 - i] = inner_strides[blk_desc.inner_nblks - i] * blk_desc.inner_blks[blk_desc.inner_nblks - i];
    }

    // order of outer dims. In case of IOhw_ will be {1, 0, 2, 3}
    std::vector<size_t> outer_order(outer_ndims);
    std::copy(order.begin(), order.begin() + outer_ndims, outer_order.begin());

    // blocked strides
    // [outer_strides via new_outer_order] U [inner_strides]
    SizeVector blk_strides(total_ndims, 0);
    std::copy(inner_strides.rbegin(), inner_strides.rend(), blk_strides.rbegin());
    std::transform(outer_order.begin(), outer_order.end(), blk_strides.begin(),
                   [&](size_t i) { return blk_desc.strides[i] == DNNL_RUNTIME_DIM_VAL ? Shape::UNDEFINED_DIM : blk_desc.strides[i]; });
    return blk_strides;
}

const std::vector<size_t> OnednnBlockedMemoryDesc::getOrder() const {
    return order;
}

const std::vector<size_t> OnednnBlockedMemoryDesc::getOffsetPaddingToData() const {
    return std::vector<size_t>(std::begin(desc.data.padded_offsets), std::begin(desc.data.padded_offsets) + getOrder().size());
}

size_t OnednnBlockedMemoryDesc::getOffsetPadding() const {
    return desc.data.offset0;
}

bool OnednnBlockedMemoryDesc::isCompatible(const MemoryDesc& rhs) const {
    const MemoryDesc* pRhs = &rhs;
    if (auto blockingDesc = dynamic_cast<const BlockedMemoryDesc*>(pRhs)) {
        return BlockedMemoryDesc::isCompatible(blockingDesc);
    } else {
        return false;
    }
}
