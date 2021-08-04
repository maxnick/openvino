// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "blocked_memory_desc.h"
#include "mkldnn_memory.h"

namespace MKLDNNPlugin {

class OnednnBlockedMemoryDesc : public BlockedMemoryDesc, public MKLDNNMemoryDesc {
public:
    OnednnBlockedMemoryDesc(InferenceEngine::Precision prc, const Shape& shape, const std::vector<size_t>& blockedDims,
                            const std::vector<size_t>& order, size_t offsetPadding = 0, const std::vector<size_t>& offsetPaddingToData = {},
                            const std::vector<size_t>& strides = {});

    MemoryDescPtr clone() const override {
        return MKLDNNPlugin::make_unique<OnednnBlockedMemoryDesc>(*this);
    }

    bool isCompatible(const MemoryDesc& rhs) const override;

    const std::vector<size_t> getBlockDims() const override;

    const std::vector<size_t> getOrder() const override;

    const std::vector<size_t> getOffsetPaddingToData() const override;

    size_t getOffsetPadding() const override;

    const std::vector<size_t> getStrides() const override;

private:
    OnednnBlockedMemoryDesc(const mkldnn::memory::desc& desc) : MKLDNNMemoryDesc(desc),
                MemoryDesc(MKLDNNExtensionUtils::convertToSizeVector(desc.dims()), OneDnnBlocked) {}

    friend MemoryDescPtr MemoryDescUtils::makeDescriptor(const mkldnn::memory::desc &desc);
};

} // namespace MKLDNNPlugin
