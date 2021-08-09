// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_layouts.h>
#include <ie_blob.h>
#include "mkldnn/ie_mkldnn.h"

namespace MKLDNNPlugin {

class MKLDNNMemoryDesc;
class BlockedMemoryDesc;
class OnednnBlockedMemoryDesc;
class CpuBlockedMemoryDesc;
class MKLDNNMemory;

class MemoryDescUtils {
public:
    /**
     * @brief Creates OnednnBlockedMemoryDesc if desc is blocked, otherwise MKLDNNMemoryDesc
     * @param desc mkldnn::memory::desc from which one of the descriptors will be created
     * @return pointer to OnednnBlockedMemoryDesc or MKLDNNMemoryDesc
     */
    static MemoryDescPtr makeDescriptor(const mkldnn::memory::desc &desc);

    /**
     * @brief Converts MemoryDesc to MKLDNNMemoryDesc
     * @param desc MemoryDesc to be converted
     * @return converted MKLDNNMemoryDesc
     */
    static std::unique_ptr<MKLDNNMemoryDesc> convertToMKLDNNMemoryDesc(const MemoryDesc& desc);

    /**
     * @brief Converts BlockedMemoryDesc to MKLDNNMemoryDesc
     * @param desc BlockedMemoryDesc to be converted
     * @return converted MKLDNNMemoryDesc
     */
    static std::unique_ptr<MKLDNNMemoryDesc> convertToMKLDNNMemoryDesc(const CpuBlockedMemoryDesc& desc);

    /**
     * @brief Converts InferenceEngine::TensorDesc to OnednnBlockedMemoryDesc
     * @param desc InferenceEngine::TensorDesc to be converted
     * @return converted OnednnBlockedMemoryDesc
     */
    static std::unique_ptr<OnednnBlockedMemoryDesc> convertToOnednnBlockedMemoryDesc(const InferenceEngine::TensorDesc& desc);

    /**
     * @brief Creates BlockedMemoryDesc with offsetPadding of UNDEFINED_DIM size
     * @param desc modifiable BlockedMemoryDesc
     * @return pointer to MemoryDesc
     */
    static MemoryDescPtr applyUndefinedOffset(const MemoryDesc& desc);

    /**
     * @brief Creates MemoryDesc with offsetPadding of 0 size
     * @param desc modifiable MemoryDesc
     * @return pointer to MemoryDesc
     */
    static MemoryDescPtr resetOffset(const MemoryDesc* desc);

    /**
     * @brief Creates InferenceEngine::Blob from MemoryDesc
     * @param desc MemoryDesc from which will be created InferenceEngine::Blob
     * @return pointer to InferenceEngine::Blob
     */
    static InferenceEngine::Blob::Ptr createBlob(const MemoryDesc& memDesc);

    /**
     * @brief Creates InferenceEngine::Blob from MKLDNNMemory with the memory reuse
     * @param desc MKLDNNMemory from which will be created InferenceEngine::Blob
     * @return pointer to InferenceEngine::Blob
     */
    static InferenceEngine::Blob::Ptr interpretAsBlob(const MKLDNNMemory& mem);

    /**
     * @brief Converts MemoryDesc to InferenceEngine::TensorDesc
     * @param desc MemoryDesc to be converted
     * @return converted InferenceEngine::TensorDesc
     */
    static InferenceEngine::TensorDesc convertToTensorDesc(const MemoryDesc& desc);
};

}  // namespace MKLDNNPlugin
