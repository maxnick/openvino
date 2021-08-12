// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_memory_desc.h"
#include "mkldnn_extension_utils.h"
#include <common/memory_desc_wrapper.hpp>
#include "mkldnn/ie_mkldnn.h"

namespace MKLDNNPlugin {

MKLDNNMemoryDesc::MKLDNNMemoryDesc(const mkldnn::memory::desc& desc) :
    MemoryDesc(Shape(MKLDNNExtensionUtils::convertToSizeVector(desc.dims())), Mkldnn), desc(desc) {
    if (desc.data.format_kind == dnnl::impl::format_kind::any)
        IE_THROW(Unexpected) << "Memory format any is prohibited!";
}

size_t MKLDNNMemoryDesc::getMemSizeImp() const {
    return desc.get_size();
}

size_t MKLDNNMemoryDesc::getElementOffset(size_t elemNumber) const {
    mkldnn::impl::memory_desc_wrapper wrapped(desc.data);
    return wrapped.off_l(elemNumber);
}

bool MKLDNNMemoryDesc::isCompatible(const MemoryDesc &rhs) const {
    if (MemoryDescType::Mkldnn == rhs.getType()) {
        return this->desc == rhs.as<MKLDNNMemoryDesc>()->desc;
    } else {
        return false;
    }
}

std::string MKLDNNMemoryDesc::serializeFormat() const {
    if (desc.data.format_kind == dnnl_format_kind_wino) {
        switch (desc.data.format_desc.wino_desc.wino_format) {
            case dnnl_wino_memory_format_t::dnnl_wino_wei_aaOIoi: return "wino_aaOIoi";
            case dnnl_wino_memory_format_t::dnnl_wino_wei_aaOio: return "wino_aaOio";
            case dnnl_wino_memory_format_t::dnnl_wino_wei_aaOBiOo: return "wino_aaOBiOo";
            case dnnl_wino_memory_format_t::dnnl_wino_wei_OBaaIBOIio: return "wino_OBaaIBOIio";
            default: return "wino_undef";
        }
    }
    return "undef";
}

bool MKLDNNMemoryDesc::isDefinedImp() const {
    mkldnn::impl::memory_desc_wrapper wrappedThis(desc.data);
    if (!wrappedThis.is_blocking_desc()) {
        return true;
    }

    if (wrappedThis.has_runtime_dims_or_strides()) {
        return false;
    }

    return wrappedThis.offset0() != Shape::UNDEFINED_DIM;
}

InferenceEngine::Precision MKLDNNMemoryDesc::getPrecision() const {
    return MKLDNNExtensionUtils::DataTypeToIEPrecision(desc.data_type());
}

void MKLDNNMemoryDesc::setPrecision(InferenceEngine::Precision prc) {
    desc.data.data_type = static_cast<dnnl_data_type_t>(MKLDNNExtensionUtils::IEPrecisionToDataType(prc));
}

std::unique_ptr<MemoryDesc> MKLDNNMemoryDesc::cloneWithNewDimsImp(const std::vector<size_t> &dims) const {
    IE_THROW(Unexpected) << "Cannot clone non blocked oneDNN desc with new dims";
}

size_t MKLDNNMemoryDesc::getMaxMemSize() const {
    if (desc.data.format_kind != dnnl_blocked || shape.isStatic()) {
        return getCurrentSize();
    }

    auto& maxDims = shape.getMaxDims();
    if (std::any_of(maxDims.begin(), maxDims.end(), [](size_t x){ return Shape::UNDEFINED_DIM == x; })) {
        return UNDEFINED_SIZE;
    }

    auto maxDimsDesc = cloneWithNewDims(maxDims);
    return maxDimsDesc->getCurrentSize();
}

} // namespace MKLDNNPlugin
