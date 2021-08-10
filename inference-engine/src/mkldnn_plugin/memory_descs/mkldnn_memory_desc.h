// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_blocked_memory_desc.h"

/**
 * @brief
 *
 * MKLDNNMemoryDesc - the descriptor of tensor representation in memory. Describes all required information
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
class MKLDNNMemoryDesc : public virtual MemoryDesc {
public:
    explicit MKLDNNMemoryDesc(const mkldnn::memory::desc& desc);

    mkldnn::memory::data_type getDataType() const {
        return static_cast<mkldnn::memory::data_type>(desc.data.data_type);
    }

    operator mkldnn::memory::desc() const;

    dnnl_format_kind_t getFormatKind() const {
        return desc.data.format_kind;
    }

    std::unique_ptr<MemoryDesc> clone() const override {
        return MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(*this);
    }

    std::string serializeFormat() const override;

    InferenceEngine::Precision getPrecision() const override;

    void setPrecision(InferenceEngine::Precision prc) override;

    bool isCompatible(const MemoryDesc& rhs) const override;

    size_t getMaxMemSize() const override;

    mkldnn::memory::desc getMklDesc() const {
        return desc;
    }

    bool hasLayoutType(LayoutType layoutType) const override { return false; }

protected:
    MKLDNNMemoryDesc(const Shape& shape) : MemoryDesc(shape, Mkldnn) {}
    static constexpr size_t UNREACHABLE_DIM = std::numeric_limits<size_t>::max();

    mkldnn::memory::desc desc;

private:
    size_t getElementOffset(size_t elemNumber) const override;

    size_t getMemSizeImp() const override;
    bool isDefinedImp() const override;
    std::unique_ptr<MemoryDesc> cloneWithNewDimsImp(const std::vector<size_t>& dims) const override;
};

using MKLDNNMemoryDescPtr = std::unique_ptr<MKLDNNMemoryDesc>;

}  // namespace MKLDNNPlugin
