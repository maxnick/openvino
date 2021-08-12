// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_memory_desc.h"
#include "memory_descs/cpu_memory_desc_utils.h"
#include "mkldnn_memory.h"
#include "memory_descs/onednn_blocked_memory_desc.h"
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

MemoryDescPtr MemoryDescUtils::makeDescriptor(const mkldnn::memory::desc &desc) {
    if (desc.data.format_kind == dnnl_blocked) {
        return std::unique_ptr<OnednnBlockedMemoryDesc>(new OnednnBlockedMemoryDesc(desc));
    } else {
        return MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(desc);
    }
}

std::unique_ptr<MKLDNNMemoryDesc> MemoryDescUtils::convertToMKLDNNMemoryDesc(const MemoryDesc& desc) {
    if (MemoryDescType::Blocked == desc.getType()) {
        return convertToMKLDNNMemoryDesc(*(desc.as<CpuBlockedMemoryDesc>()));
    } else if (MemoryDescType::Mkldnn & desc.getType()) {
        return std::unique_ptr<MKLDNNMemoryDesc>(dynamic_cast<MKLDNNMemoryDesc *>(desc.clone().release()));
    } else {
        IE_THROW() << "Cannot convert MemoryDesc to MKLDNNMemoryDesc";
    }
}

std::unique_ptr<MKLDNNMemoryDesc> MemoryDescUtils::convertToMKLDNNMemoryDesc(const CpuBlockedMemoryDesc& desc) {
    return std::unique_ptr<OnednnBlockedMemoryDesc>(new OnednnBlockedMemoryDesc(desc.getPrecision(), desc.getShape(), desc.getBlockDims(),
                                                              desc.getOrder(), desc.getOffsetPadding(), desc.getOffsetPaddingToData(), desc.getStrides()));
}

std::unique_ptr<OnednnBlockedMemoryDesc> MemoryDescUtils::convertToOnednnBlockedMemoryDesc(const InferenceEngine::TensorDesc& desc) {
    const auto &blkDesc = desc.getBlockingDesc();
    return std::unique_ptr<OnednnBlockedMemoryDesc>(new OnednnBlockedMemoryDesc(desc.getPrecision(), Shape(desc.getDims()), blkDesc.getBlockDims(),
                                                                                blkDesc.getOrder(), blkDesc.getOffsetPadding(),
                                                                                blkDesc.getOffsetPaddingToData(), blkDesc.getStrides()));
}

MemoryDescPtr MemoryDescUtils::applyUndefinedOffset(const MemoryDesc& desc) {
    if (desc.getType() == MemoryDescType::Mkldnn) {
        return desc.clone();
    }

    const auto blkMemDesc = desc.as<BlockedMemoryDesc>();

    std::vector<size_t> strides;
    std::vector<size_t> offsetPaddingToData;
    strides.resize(blkMemDesc->getBlockDims().size(), Shape::UNDEFINED_DIM);
    offsetPaddingToData.resize(blkMemDesc->getBlockDims().size(), 0);
    size_t offsetPadding = Shape::UNDEFINED_DIM;

    if (blkMemDesc->getType() == MemoryDescType::Blocked) {
        return MKLDNNPlugin::make_unique<CpuBlockedMemoryDesc>(blkMemDesc->getPrecision(), blkMemDesc->getShape(), blkMemDesc->getBlockDims(),
                                                               blkMemDesc->getOrder(), offsetPadding, offsetPaddingToData, strides);
    } else if (blkMemDesc->getType() == MemoryDescType::OneDnnBlocked) {
        return std::unique_ptr<OnednnBlockedMemoryDesc>(new OnednnBlockedMemoryDesc(blkMemDesc->getPrecision(), blkMemDesc->getShape(),
                                                                                    blkMemDesc->getBlockDims(), blkMemDesc->getOrder(),
                                                                                    offsetPadding, offsetPaddingToData, strides));
    } else {
        IE_THROW() << "Cannot apply undefined offset. Unsupported memory desc type";
    }
}

MemoryDescPtr MemoryDescUtils::resetOffset(const MemoryDesc* desc) {
    const auto blkMemDesc = desc->as<BlockedMemoryDesc>();

    if (MemoryDescType::Blocked == desc->getType()) {
        return MKLDNNPlugin::make_unique<CpuBlockedMemoryDesc>(blkMemDesc->getPrecision(), blkMemDesc->getShape(),
                                                               blkMemDesc->getBlockDims(), blkMemDesc->getOrder());
    } else if (MemoryDescType::OneDnnBlocked == desc->getType()) {
        return std::unique_ptr<OnednnBlockedMemoryDesc>(new OnednnBlockedMemoryDesc(blkMemDesc->getPrecision(), blkMemDesc->getShape(),
                                                                  blkMemDesc->getBlockDims(), blkMemDesc->getOrder()));
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
