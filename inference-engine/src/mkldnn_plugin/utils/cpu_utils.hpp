// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/common/cpu_memcpy.h"
#include "nodes/common/cpu_convert.h"
#include "mkldnn_memory.h"

namespace MKLDNNPlugin {

/**
* @brief Returns normalized by size dims where missing dimensions are filled with units from the beginning
* Example: dims = {2, 3, 5}; ndims = 5; result = {1, 1, 2, 3, 5}
* @param dims
* shape to normalize
* @param ndims
* rank of resulting shape
* @return normalized vector
*/
inline std::vector<size_t> getNormalizedDimsBySize(const InferenceEngine::SizeVector &dims, size_t ndims) {
    if (dims.size() >= ndims)
        return dims;

    std::vector<size_t> normalizedDims = dims;
    for (size_t i = 0; i < (ndims - dims.size()); i++) {
        normalizedDims.insert(normalizedDims.begin(), 1);
    }
    return normalizedDims;
}

/**
* @brief Checked that secondInputDims unidirectional broadcastable per tensor or per channel to firstInputDims
* @param firstInputDims
* shape on which should be broadcastable
* @param secondInputDims
* shape which should be broadcastable
* @return true if broadcastable, false otherwise.
*/
inline bool isPerTensorOrPerChannelBroadcastable(const InferenceEngine::SizeVector &firstInputDims, const InferenceEngine::SizeVector& secondInputDims) {
    if (secondInputDims.size() > firstInputDims.size())
        return false;
    if (std::accumulate(secondInputDims.begin(), secondInputDims.end(), 1, std::multiplies<size_t>()) == 1)
        return true;

    std::vector<size_t> normalizedSecondInputDims = getNormalizedDimsBySize(secondInputDims, firstInputDims.size());
    for (size_t i = 0; i < normalizedSecondInputDims.size(); i++) {
        if ((i == 1 && normalizedSecondInputDims[i] != firstInputDims[1]) || (i != 1 && normalizedSecondInputDims[i] != 1))
            return false;
    }
    return true;
}

inline bool isEmptyTensorDesc(const InferenceEngine::TensorDesc &td) {
    const auto dims = td.getDims();
    return std::any_of(dims.begin(), dims.end(), [](size_t dim) { return dim == 0; } );
}

/**
* @brief Return precision to which given precision must be converted to be supported in plug-in
* @param precision
* precision for convert
* @return plug-in supported precision or UNSPECIFIED if precision unsupported
*/
inline InferenceEngine::Precision normalizeToSupportedPrecision(InferenceEngine::Precision precision) {
    switch (precision) {
        case InferenceEngine::Precision::U8:
        case InferenceEngine::Precision::I8:
        case InferenceEngine::Precision::I32:
        case InferenceEngine::Precision::BF16:
        case InferenceEngine::Precision::FP32: {
            break;
        }
        case InferenceEngine::Precision::BOOL: {
            precision = InferenceEngine::Precision::U8;
            break;
        }
        case InferenceEngine::Precision::U16:
        case InferenceEngine::Precision::I16:
        case InferenceEngine::Precision::I64:
        case InferenceEngine::Precision::U64: {
            precision = InferenceEngine::Precision::I32;
            break;
        }
        case InferenceEngine::Precision::FP16: {
            precision = InferenceEngine::Precision::FP32;
            break;
        }
        default: {
            precision = InferenceEngine::Precision::UNSPECIFIED;
        }
    }
    return precision;
}

inline void reorderData(const MKLDNNMemory &input, const MKLDNNMemory &output, size_t size = 0) {
    if (size != 0)
        IE_ASSERT(size <= output.GetSize());
    if (input.GetDesc().isCompatible(output.GetDesc())) {
        auto srcPtr = static_cast<uint8_t*>(input.GetPtr());
        auto dstPtr = static_cast<uint8_t*>(output.GetPtr());

        auto copySize = size == 0 ? output.GetSize() : size;
        cpu_memcpy(dstPtr, srcPtr, copySize);
    } else {
        std::unique_ptr<mkldnn::reorder> pReorder;
        std::shared_ptr<mkldnn::memory> srcMemoryPtr;
        std::vector<uint8_t> tmpBuff;

        try {
            pReorder = std::unique_ptr<mkldnn::reorder>(new mkldnn::reorder(input.GetPrimitive(), output.GetPrimitive()));
            srcMemoryPtr = input.prim;
        }
        catch (const mkldnn::error& err) {
            if (mkldnn_unimplemented == err.status && output.GetDataType() != input.GetDataType()) {
                //we probably could not make the reorder because there is no one supporting this precision conversion
                //lets try to convert data first using cpu_convert
                auto data = static_cast<const uint8_t *>(input.GetPtr());
                tmpBuff.resize(input.GetSize());

                const auto outPrc = MKLDNNExtensionUtils::DataTypeToIEPrecision(output.GetDataType());
                cpu_convert(data, tmpBuff.data(), MKLDNNExtensionUtils::DataTypeToIEPrecision(input.GetDataType()),
                            outPrc, input.GetShape().getElementsCount());

                MKLDNNMemory tmpMem(output.eng);
                auto tmpDesc = input.GetDesc().clone();
                tmpDesc->setPrecision(outPrc);
                tmpMem.Create(std::move(tmpDesc), tmpBuff.data());

                pReorder = std::unique_ptr<mkldnn::reorder>(new mkldnn::reorder(tmpMem.GetPrimitive(), output.GetPrimitive()));
                srcMemoryPtr = tmpMem.prim;
            } else {
                throw;
            }
        }
        if (pReorder) {
            mkldnn::stream loc_stream(output.eng, mkldnn::stream::flags::default_order);
            pReorder->execute(loc_stream, *srcMemoryPtr, *output.prim);
        } else {
            IE_THROW() << "Could not make mkldnn reorder.";
        }
    }
}



}  // namespace MKLDNNPlugin
