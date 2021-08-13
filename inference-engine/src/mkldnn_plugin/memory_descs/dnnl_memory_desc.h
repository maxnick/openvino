// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_blocked_memory_desc.h"
#include "mkldnn_extension_utils.h"

/**
 * @brief
 *
 * DnnlMemoryDesc - the descriptor of tensor representation in memory. Describes all required information
 * for proper allocation and handling tensor in some buffer. The real memory is not present, just description.
 * This object answers on question how and where data with logical index [x1, x2, .. xN] placed in real buffer.
 * In the simplest case it describe a mapping between "logical offset" and "real offset".
 *
 */

namespace MKLDNNPlugin {

/**
 * Represent internal plugin abstraction of tensor description
 *
 */

class DnnlMemoryDesc;

using DnnlMemoryDescPtr = std::unique_ptr<DnnlMemoryDesc>;

class DnnlMemoryDesc : public virtual MemoryDesc {
public:
    mkldnn::memory::data_type getDataType() const {
        return static_cast<mkldnn::memory::data_type>(desc.data.data_type);
    }

    dnnl_format_kind_t getFormatKind() const {
        return desc.data.format_kind;
    }

    std::unique_ptr<MemoryDesc> clone() const override {
        return MKLDNNPlugin::make_unique<DnnlMemoryDesc>(*this);
    }

    std::string serializeFormat() const override;

    InferenceEngine::Precision getPrecision() const override;

    void setPrecision(InferenceEngine::Precision prc) override;

    bool isCompatible(const MemoryDesc& rhs) const override;

    size_t getMaxMemSize() const override;

    // TODO [DS] phase 2: rename -> ?
    mkldnn::memory::desc getMklDesc() const {
        return desc;
    }

    bool hasLayoutType(LayoutType layoutType) const override { return false; }

    virtual bool isSame(mkldnn::memory::format_tag fmt) const { return false; }

protected:
    DnnlMemoryDesc() {}
    static constexpr size_t UNREACHABLE_DIM = std::numeric_limits<size_t>::max();

    mkldnn::memory::desc desc;

private:
    explicit DnnlMemoryDesc(const mkldnn::memory::desc& desc);

    size_t getElementOffset(size_t elemNumber) const override;

    size_t getMemSizeImp() const override;
    bool isDefinedImp() const override;
    std::unique_ptr<MemoryDesc> cloneWithNewDimsImp(const std::vector<size_t>& dims) const override;

    friend DnnlMemoryDescPtr MKLDNNExtensionUtils::makeDescriptor(const mkldnn::memory::desc &desc);
};

}  // namespace MKLDNNPlugin
