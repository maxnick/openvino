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

    // TODO [mandrono]
    static MemoryDescPtr makeDescriptor(const mkldnn::memory::desc &desc);

    static MemoryDescPtr makeDescriptor(const Shape& shape, mkldnn::memory::data_type dataType, mkldnn::memory::format_tag format);

    /**
     * @brief Converts MemoryDesc to MKLDNNMemoryDesc
     * @param desc MemoryDesc to be converted
     * @return converted MKLDNNMemoryDesc
     */
    static MKLDNNMemoryDesc convertToMKLDNNMemoryDesc(const MemoryDesc& desc);

    /**
     * @brief Converts BlockedMemoryDesc to MKLDNNMemoryDesc
     * @param desc BlockedMemoryDesc to be converted
     * @return converted MKLDNNMemoryDesc
     */
    static MKLDNNMemoryDesc convertToMKLDNNMemoryDesc(const BlockedMemoryDesc& desc);

    /**
     * @brief Converts InferenceEngine::TensorDesc to MKLDNNMemoryDesc
     * @param desc InferenceEngine::TensorDesc to be converted
     * @return converted MKLDNNMemoryDesc
     */
    static MKLDNNMemoryDesc convertToMKLDNNMemoryDesc(const InferenceEngine::TensorDesc& desc);

    /**
     * @brief Converts MemoryDesc to CpuBlockedMemoryDesc
     * @param desc MemoryDesc to be converted
     * @return converted CpuBlockedMemoryDesc
     */
    static CpuBlockedMemoryDesc convertToCpuBlockedDescriptor(const MemoryDesc& desc);

    /**
     * @brief Converts MKLDNNMemoryDesc to CpuBlockedMemoryDesc
     * @param desc MKLDNNMemoryDesc to be converted
     * @return converted CpuBlockedMemoryDesc
     */
    static CpuBlockedMemoryDesc convertToCpuBlockedDescriptor(const OnednnBlockedMemoryDesc& inpDesc);

    /**
     * @brief Creates BlockedMemoryDesc with offsetPadding of UNDEFINED_DIM size
     * @param desc modifiable BlockedMemoryDesc
     * @return pointer to MemoryDesc
     */
    static MemoryDescPtr applyUndefinedOffset(const MemoryDesc* desc);

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
