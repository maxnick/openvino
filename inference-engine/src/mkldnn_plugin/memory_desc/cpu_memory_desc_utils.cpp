// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_memory_desc.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "mkldnn_memory.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
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

std::unique_ptr<DnnlMemoryDesc> MemoryDescUtils::convertToDnnlMemoryDesc(const MemoryDesc& desc) {
    if (MemoryDescType::Blocked == desc.getType()) {
        const auto cpuDesc = desc.as<CpuBlockedMemoryDesc>();
        return std::unique_ptr<DnnlBlockedMemoryDesc>(new DnnlBlockedMemoryDesc(cpuDesc->getPrecision(), cpuDesc->getShape(), cpuDesc->getBlockDims(),
                                                                                cpuDesc->getOrder(), cpuDesc->getOffsetPadding(),
                                                                                cpuDesc->getOffsetPaddingToData(), cpuDesc->getStrides()));
    } else if (MemoryDescType::Mkldnn & desc.getType()) {
        return std::unique_ptr<DnnlMemoryDesc>(dynamic_cast<DnnlMemoryDesc *>(desc.clone().release()));
    } else {
        IE_THROW() << "Cannot convert MemoryDesc to DnnlMemoryDesc";
    }
}

std::unique_ptr<DnnlBlockedMemoryDesc> MemoryDescUtils::convertToDnnlBlockedMemoryDesc(const MemoryDesc& desc) {
    if (MemoryDescType::DnnlBlocked == desc.getType()) {
        return std::unique_ptr<DnnlBlockedMemoryDesc>(dynamic_cast<DnnlBlockedMemoryDesc *>(desc.clone().release()));
    } else if (MemoryDescType::Blocked == desc.getType()) {
        const auto cpuDesc = desc.as<CpuBlockedMemoryDesc>();
        return std::unique_ptr<DnnlBlockedMemoryDesc>(new DnnlBlockedMemoryDesc(cpuDesc->getPrecision(), cpuDesc->getShape(), cpuDesc->getBlockDims(),
                                                                                cpuDesc->getOrder(), cpuDesc->getOffsetPadding(),
                                                                                cpuDesc->getOffsetPaddingToData(), cpuDesc->getStrides()));
    } else {
        IE_THROW() << "Cannot convert MemoryDesc to DnnlMemoryDesc";
    }
}

std::unique_ptr<DnnlBlockedMemoryDesc> MemoryDescUtils::convertToDnnlBlockedMemoryDesc(const InferenceEngine::TensorDesc& desc) {
    const auto &blkDesc = desc.getBlockingDesc();
    return std::unique_ptr<DnnlBlockedMemoryDesc>(new DnnlBlockedMemoryDesc(desc.getPrecision(), Shape(desc.getDims()), blkDesc.getBlockDims(),
                                                                                blkDesc.getOrder(), blkDesc.getOffsetPadding(),
                                                                                blkDesc.getOffsetPaddingToData(), blkDesc.getStrides()));
}

std::unique_ptr<BlockedMemoryDesc> MemoryDescUtils::convertToBlockedMemoryDesc(const MemoryDesc& desc) {
    if (desc.getType() & MemoryDescType::Blocked) {
        return std::unique_ptr<BlockedMemoryDesc>(dynamic_cast<BlockedMemoryDesc *>(desc.clone().release()));
    } else {
        IE_THROW() << "Can not convert unsupported memory descriptor";
    }
}

MemoryDescPtr MemoryDescUtils::cloneWithUndefStridesAndOffset(const MemoryDesc& desc) {
    if (desc.getType() == MemoryDescType::Mkldnn) {
        IE_THROW() << "Can't apply undefined offset for mkldnn memory desc";
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
    } else if (blkMemDesc->getType() == MemoryDescType::DnnlBlocked) {
        return std::unique_ptr<DnnlBlockedMemoryDesc>(new DnnlBlockedMemoryDesc(blkMemDesc->getPrecision(), blkMemDesc->getShape(),
                                                                                    blkMemDesc->getBlockDims(), blkMemDesc->getOrder(),
                                                                                    offsetPadding, offsetPaddingToData, strides));
    } else {
        IE_THROW() << "Cannot apply undefined offset. Unsupported memory desc type";
    }
}

MemoryDescPtr MemoryDescUtils::cloneWithDefaultStridesAndOffset(const MemoryDesc* desc) {
    const auto blkMemDesc = desc->as<BlockedMemoryDesc>();

    if (MemoryDescType::Blocked == desc->getType()) {
        return MKLDNNPlugin::make_unique<CpuBlockedMemoryDesc>(blkMemDesc->getPrecision(), blkMemDesc->getShape(),
                                                               blkMemDesc->getBlockDims(), blkMemDesc->getOrder());
    } else if (MemoryDescType::DnnlBlocked == desc->getType()) {
        return std::unique_ptr<DnnlBlockedMemoryDesc>(new DnnlBlockedMemoryDesc(blkMemDesc->getPrecision(), blkMemDesc->getShape(),
                                                                  blkMemDesc->getBlockDims(), blkMemDesc->getOrder()));
    } else {
        IE_THROW() << "cloneWithDefaultStridesAndOffset supports Blocked descriptors only";
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
    auto& memDesc = mem.getDesc();
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

std::string MemoryDescUtils::dim2str(Dim dim) {
    return dim == Shape::UNDEFINED_DIM ? "?" : std::to_string(dim);
}

std::string MemoryDescUtils::dims2str(const VectorDims& dims) {
    std::stringstream output;
    output << "{";

    auto itr = dims.begin();
    do {
        output << dim2str(*itr);
    } while (++itr != dims.end() && output << ", ");

    output << "}";
    return output.str();
}

} // namespace MKLDNNPlugin
