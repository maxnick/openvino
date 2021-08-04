// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_memory_desc.h"
#include "cpu_memory_desc_utils.h"
#include "mkldnn_memory.h"
#include "onednn_blocked_memory_desc.h"
#include "utils/general_utils.h"
#include "utils/cpu_utils.hpp"
#include <limits>
#include <vector>
#include <numeric>
#include <blob_factory.hpp>
#include <dnnl_types.h>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

namespace MKLDNNPlugin {

MemoryDescPtr MemoryDescUtils::makeDescriptor(const Shape& shape, mkldnn::memory::data_type dataType, mkldnn::memory::format_tag format) {
    // TODO: mandrono
    MKLDNNMemoryDesc mdesc(shape, dataType, format);
    return MemoryDescUtils::makeDescriptor(mdesc.getMklDesc());
}

MemoryDescPtr MemoryDescUtils::makeDescriptor(const mkldnn::memory::desc &desc) {
    if (desc.data.format_kind == dnnl_blocked && desc.data.extra.flags == dnnl_memory_extra_flag_none) {
        return std::unique_ptr<OnednnBlockedMemoryDesc>(new OnednnBlockedMemoryDesc(desc));
    } else {
        return MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(desc);
    }
}

MKLDNNMemoryDesc MemoryDescUtils::convertToMKLDNNMemoryDesc(const MemoryDesc& desc) {
    if (MemoryDescType::CpuBlocked == desc.getType() || MemoryDescType::OneDnnBlocked == desc.getType()) {
        return convertToMKLDNNMemoryDesc(*(desc.as<BlockedMemoryDesc>()));
    } else if (MemoryDescType::Mkldnn == desc.getType()) {
        return *(desc.as<MKLDNNMemoryDesc>());
    } else {
        IE_THROW() << "Cannot convert MemoryDesc to MKLDNNMemoryDesc";
    }
}

MKLDNNMemoryDesc MemoryDescUtils::convertToMKLDNNMemoryDesc(const BlockedMemoryDesc& desc) {
    return MKLDNNMemoryDesc(desc.getPrecision(), desc.getShape(), desc.getBlockDims(),
                            desc.getOrder(), desc.getOffsetPadding(), desc.getOffsetPaddingToData(), desc.getStrides());
}

/**
 * Construct from IE::TensorDesc
 * @param tDesc
 *
 * IE  IOhw_4i16o4i   dims(N) = {32, 64, 128, 128}
 *   blockedDims  {4, 2, 128, 128, 4, 16, 4}                      // total dims(inner, outermost, auto blocked/padded). Generally sorted by strides.
 *   strides      {8388608, 4194304,  32768, 256, 64,  4, 1}      // strides for blockedDims, growing sequence
 *   order        {1, 0,   2,   3, 1,  0, 1}                      // matching to original dims
 *
 *   All vectors blockedDims/strides/order have same size equals total num of internal blocked dims(inner_dims + outer_dims)
 *
 *   Tensor descriptor filing is not deterministic. It allows any permutation of index which keeps order of
 *   real dims spliting.
 *      for {1, 0, 2, 3, 1, 0, 1} we can swap elements [1] <=> [4]
 *      but not [0]<=>[4] because it breacke spliting original dims into internal blocked dims
 *   Normalization of representation: Make strides growing but keep layout same as original. Not all
 *   layout allow us to meet normalize form of tensor desc.
 *
 *   Limitation of conversion first N elements of order should be permutation of [0,1,2 ... N]
 */
// TODO [DS mandrono]: remove ???
MKLDNNMemoryDesc MemoryDescUtils::convertToMKLDNNMemoryDesc(const InferenceEngine::TensorDesc& tDesc) {
    mkldnn::memory::desc mkldnnDesc({}, mkldnn::memory::data_type::undef, mkldnn::memory::format_tag::undef);
    auto dims = tDesc.getDims();

    // TODO: implicit conversion of dims is no good...
    if (tDesc.getLayout() == Layout::SCALAR) {
        mkldnnDesc.data.format_kind = dnnl_blocked;
        mkldnnDesc.data.data_type = memory::convert_to_c(MKLDNNMemory::convertToDataType(tDesc.getPrecision()));
        mkldnnDesc.data.ndims = 1;
        mkldnnDesc.data.dims[0] = 1;
        mkldnnDesc.data.padded_dims[0] = 1;
        mkldnnDesc.data.format_desc.blocking.strides[0] = 1;
        mkldnnDesc.data.padded_offsets[0] = 0;
        mkldnnDesc.data.offset0 = tDesc.getBlockingDesc().getOffsetPadding();
        return MKLDNNMemoryDesc(mkldnnDesc);
    }

    if (tDesc.getLayout() == Layout::ANY) {
        mkldnnDesc.data.format_kind = dnnl_format_kind_any;
        mkldnnDesc.data.data_type = memory::convert_to_c(MKLDNNMemory::convertToDataType(tDesc.getPrecision()));
        mkldnnDesc.data.ndims = dims.size();
        std::copy(dims.begin(), dims.end(), mkldnnDesc.data.dims);
        std::copy(dims.begin(), dims.end(), mkldnnDesc.data.padded_dims);
        mkldnnDesc.data.offset0 = tDesc.getBlockingDesc().getOffsetPadding();
        std::fill(mkldnnDesc.data.padded_offsets, mkldnnDesc.data.padded_offsets + dims.size(), 0);
        return MKLDNNMemoryDesc(mkldnnDesc);
    }

    auto ie_blkdDims = tDesc.getBlockingDesc().getBlockDims();
    auto ie_order = tDesc.getBlockingDesc().getOrder();
    auto ie_offsetsToData = tDesc.getBlockingDesc().getOffsetPaddingToData();
    auto ie_strides = tDesc.getBlockingDesc().getStrides();

    size_t outer_ndims = dims.size();
    size_t inner_ndims = ie_order.size() - dims.size();

    bool is_descending_strides = true;
    for (int i = 1; i < ie_strides.size(); i++) {
        is_descending_strides &= (ie_strides[i-1] >= ie_strides[i]);
    }

    // TODO: That's strong constrains and can be mitigated. IE::TensorDesc allow to transpose blocked dims
    //       and may be we can achieve correct "descending strides" form which allow conversion.
    if (!is_descending_strides)
        IE_THROW() << "Unsupported case for conversion";

    std::vector<size_t> outer_order(outer_ndims, outer_ndims + 1); // outer_order[i] is index of stride for i-th dimension
    for (size_t i = 0; i < outer_ndims; i++) {
        outer_order[ie_order[i]] = i;
    }
    bool outer_is_correct_permutation_of_n =
            std::find(outer_order.begin(), outer_order.end(), outer_ndims + 1) == outer_order.end();

    if (!outer_is_correct_permutation_of_n)
        IE_THROW() << "Unsupported case for conversion";

    bool inner_block_are_dense = one_of(ie_strides.back(), 0, 1);  // stride 1 - is dense case, 0 - broad casted
    for (int i = outer_ndims; i < ie_strides.size() - 1; i++) {
        inner_block_are_dense &= (ie_strides[i] == ie_strides[i+1] * ie_blkdDims[i+1]);
    }

    if (!inner_block_are_dense)
        IE_THROW() << "Unsupported case for conversion";

    bool inner_pad_offsets_is_zero = std::all_of(ie_offsetsToData.begin() + outer_ndims, ie_offsetsToData.end(),
                                                 [](size_t pad) { return  pad == 0; });

    if (!inner_pad_offsets_is_zero)
        IE_THROW() << "Unsupported case for conversion";

    // Fill general memory desc fields
    mkldnnDesc.data.format_kind = dnnl_blocked;
    mkldnnDesc.data.data_type = memory::convert_to_c(MKLDNNMemory::convertToDataType(tDesc.getPrecision()));
    mkldnnDesc.data.ndims = dims.size();
    mkldnnDesc.data.offset0 = tDesc.getBlockingDesc().getOffsetPadding();
    std::copy(dims.begin(), dims.end(), mkldnnDesc.data.dims);
    std::copy(ie_offsetsToData.begin(), ie_offsetsToData.begin() + outer_ndims, mkldnnDesc.data.padded_offsets);
    std::fill(mkldnnDesc.data.padded_dims, mkldnnDesc.data.padded_dims + outer_ndims, 1);
    for (size_t i = 0; i < ie_order.size(); i++) {
        auto idx = ie_order[i];
        mkldnnDesc.data.padded_dims[idx] *= ie_blkdDims[i];
    }

    // Fill blocking desc
    auto &dnn_blk_desc = mkldnnDesc.data.format_desc.blocking;
    dnn_blk_desc.inner_nblks = inner_ndims;
    std::copy(ie_blkdDims.end() - inner_ndims, ie_blkdDims.end(), dnn_blk_desc.inner_blks);
    std::copy(ie_order.end() - inner_ndims, ie_order.end(), dnn_blk_desc.inner_idxs);
    for (size_t i = 0; i < outer_ndims; i++) {
        dnn_blk_desc.strides[i] = ie_strides[outer_order[i]];
    }

    return MKLDNNMemoryDesc(mkldnnDesc);
}

CpuBlockedMemoryDesc MemoryDescUtils::convertToCpuBlockedDescriptor(const MemoryDesc &desc) {
    if (desc.getType() == MemoryDescType::CpuBlocked) {
        return *(desc.as<CpuBlockedMemoryDesc>());
    } else if (desc.getType() == MemoryDescType::OneDnnBlocked) {
        return MemoryDescUtils::convertToCpuBlockedDescriptor(*(desc.as<OnednnBlockedMemoryDesc>()));
    } else {
        IE_THROW() << "Cannot convert to cpu blocked memory descriptor. Unsupported memory desc type";
    }
}

CpuBlockedMemoryDesc MemoryDescUtils::convertToCpuBlockedDescriptor(const OnednnBlockedMemoryDesc& inpDesc) {
    return CpuBlockedMemoryDesc(inpDesc.getPrecision(), inpDesc.getShape(), inpDesc.getBlockDims(),
                                inpDesc.getOrder(), inpDesc.getOffsetPadding(), inpDesc.getOffsetPaddingToData(), inpDesc.getStrides());
}

MemoryDescPtr MemoryDescUtils::applyUndefinedOffset(const MemoryDesc* desc) {
    if (desc->getType() == MemoryDescType::Mkldnn) {
        return MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(*dynamic_cast<const MKLDNNMemoryDesc*>(desc));
    }

    const auto blkMemDesc = desc->as<BlockedMemoryDesc>();

    std::vector<size_t> strides;
    std::vector<size_t> offsetPaddingToData;
    strides.resize(blkMemDesc->getBlockDims().size(), Shape::UNDEFINED_DIM);
    offsetPaddingToData.resize(blkMemDesc->getBlockDims().size(), 0);
    size_t offsetPadding = Shape::UNDEFINED_DIM;

    if (blkMemDesc->getType() == MemoryDescType::CpuBlocked) {
        return MKLDNNPlugin::make_unique<CpuBlockedMemoryDesc>(blkMemDesc->getPrecision(), blkMemDesc->getShape(), blkMemDesc->getBlockDims(),
                                                               blkMemDesc->getOrder(), offsetPadding, offsetPaddingToData, strides);
    } else if (blkMemDesc->getType() == MemoryDescType::OneDnnBlocked) {
        return MKLDNNPlugin::make_unique<OnednnBlockedMemoryDesc>(blkMemDesc->getPrecision(), blkMemDesc->getShape(), blkMemDesc->getBlockDims(),
                                                                  blkMemDesc->getOrder(), offsetPadding, offsetPaddingToData, strides);
    } else {
        IE_THROW() << "Cannot apply undefined offset. Unsupported memory desc type";
    }
}

MemoryDescPtr MemoryDescUtils::resetOffset(const MemoryDesc* desc) {
    const auto blkMemDesc = desc->as<BlockedMemoryDesc>();

    if (MemoryDescType::CpuBlocked == desc->getType()) {
        return MKLDNNPlugin::make_unique<CpuBlockedMemoryDesc>(blkMemDesc->getPrecision(), blkMemDesc->getShape(),
                                                               blkMemDesc->getBlockDims(), blkMemDesc->getOrder());
    } else if (MemoryDescType::OneDnnBlocked == desc->getType()) {
        return MKLDNNPlugin::make_unique<OnednnBlockedMemoryDesc>(blkMemDesc->getPrecision(), blkMemDesc->getShape(),
                                                                  blkMemDesc->getBlockDims(), blkMemDesc->getOrder());
    } else {
        IE_THROW() << "resetOffset supports Blocked descriptors only";
    }
}

InferenceEngine::Blob::Ptr MemoryDescUtils::createBlob(const MemoryDesc &memDesc) {
    // TODO [DS]: Rewrite when IE is moved to the new TensorDescriptor
    InferenceEngine::TensorDesc desc = convertToTensorDesc(memDesc);

    desc = InferenceEngine::TensorDesc(desc.getPrecision(), memDesc.getShape().getStaticDims(), desc.getBlockingDesc());
    InferenceEngine::Blob::Ptr blob = make_blob_with_precision(desc);
    blob->allocate();
    return blob;
}

InferenceEngine::Blob::Ptr MemoryDescUtils::interpretAsBlob(const MKLDNNMemory &mem) {
    // TODO [DS]: Rewrite when IE is moved to the new TensorDescriptor
    auto& memDesc = mem.GetDesc();
    InferenceEngine::TensorDesc desc = convertToTensorDesc(memDesc);

    desc = InferenceEngine::TensorDesc(desc.getPrecision(), memDesc.getShape().getStaticDims(), desc.getBlockingDesc());
    return make_blob_with_precision(desc, mem.GetData());
}

InferenceEngine::TensorDesc MemoryDescUtils::convertToTensorDesc(const MemoryDesc& desc) {
    if (auto blockingDesc = dynamic_cast<const BlockedMemoryDesc*>(&desc)) {
        return InferenceEngine::TensorDesc(blockingDesc->getPrecision(), blockingDesc->getShape().getStaticDims(),
                                           {blockingDesc->getBlockDims(), blockingDesc->getOrder(), blockingDesc->getOffsetPadding(),
                                            blockingDesc->getOffsetPaddingToData(), blockingDesc->getStrides()});
    } else {
        IE_THROW() << "Cannot convert MemoryDesc to InferenceEngine::TensorDesc";
    }
}

} // namespace MKLDNNPlugin
