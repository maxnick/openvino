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

    /**
     * Try to define original format tag use on creation
     *
     * @return format tag if was able to define it
     */
    // TODO [DS]: phase 2: move to the private section
    mkldnn::memory::format_tag getFormat() const;

    mkldnn::memory::data_type getDataType() const {
        return static_cast<mkldnn::memory::data_type>(desc.data.data_type);
    }

    // TODO [DS]: phase 2: remove!!!
    MKLDNNDims getDims() const {
        return MKLDNNDims(desc.data.dims, desc.data.ndims);
    }

    // TODO [DS]: phase 2: move to the blocked desc interface
    bool blocksExtended() const;

    // TODO [DS]: phase 2: remove
    operator bool() const {
        return getFormat() != mkldnn::memory::format_tag::any && getFormat() != mkldnn::memory::format_tag::undef;
    }

    bool operator == (const MKLDNNMemoryDesc& rhs) const;
    bool operator != (const MKLDNNMemoryDesc& rhs) const;

    operator mkldnn::memory::desc() const;

    bool isSame(mkldnn::memory::format_tag fmt) const;

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
    // void InitializePlain(const std::vector<size_t>& _dims, mkldnn::memory::data_type dataType);

    size_t getMemSizeImp() const override;
    bool isDefinedImp() const override;
    std::unique_ptr<MemoryDesc> cloneWithNewDimsImp(const std::vector<size_t>& dims) const override;
};

using MKLDNNMemoryDescPtr = std::unique_ptr<MKLDNNMemoryDesc>;

}  // namespace MKLDNNPlugin
