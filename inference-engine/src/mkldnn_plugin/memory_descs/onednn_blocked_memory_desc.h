// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "blocked_memory_desc.h"
#include "mkldnn_memory.h"

namespace MKLDNNPlugin {

class OnednnBlockedMemoryDesc : public BlockedMemoryDesc, public MKLDNNMemoryDesc {
public:
    OnednnBlockedMemoryDesc(InferenceEngine::Precision prc, const Shape& shape);

    OnednnBlockedMemoryDesc(const Shape& shape, mkldnn::memory::data_type dataType, mkldnn::memory::format_tag format);

    MemoryDescPtr clone() const override {
        return MKLDNNPlugin::make_unique<OnednnBlockedMemoryDesc>(*this);
    }

    bool isCompatible(const MemoryDesc& rhs) const override;
    bool isCompatible(const OnednnBlockedMemoryDesc& rhs) const;

    const std::vector<size_t> getBlockDims() const override;

    const std::vector<size_t> getOrder() const override;

    const std::vector<size_t> getOffsetPaddingToData() const override;

    size_t getOffsetPadding() const override;

    const std::vector<size_t> getStrides() const override;

    bool hasLayoutType(LayoutType layoutType) const override;

private:
    OnednnBlockedMemoryDesc(InferenceEngine::Precision prc, const Shape& shape, const std::vector<size_t>& blockedDims,
                            const std::vector<size_t>& order, size_t offsetPadding = 0, const std::vector<size_t>& offsetPaddingToData = {},
                            const std::vector<size_t>& strides = {});

    OnednnBlockedMemoryDesc(const mkldnn::memory::desc& _desc);

    std::unique_ptr<MemoryDesc> cloneWithNewDimsImp(const std::vector<size_t>& dims) const override;

    void InitializePlain(const Shape& shape, mkldnn::memory::data_type dataType);

    bool isPlainFormat() const;
    bool isBlockedCFormat(size_t blk_size = UNREACHABLE_DIM) const;
    bool isTailCFormat() const;

    std::vector<size_t> order;

    friend class MemoryDescUtils;
};

} // namespace MKLDNNPlugin
