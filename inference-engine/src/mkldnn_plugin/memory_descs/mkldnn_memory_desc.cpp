// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_memory_desc.h"
#include "mkldnn_extension_utils.h"
#include <common/memory_desc_wrapper.hpp>
#include "mkldnn/ie_mkldnn.h"

namespace MKLDNNPlugin {

bool MKLDNNMemoryDesc::operator==(const MKLDNNMemoryDesc &rhs) const {
    return this->desc == rhs.desc;
}

bool MKLDNNMemoryDesc::operator!=(const MKLDNNMemoryDesc &rhs) const {
    return !(*this == rhs);
}

MKLDNNMemoryDesc::operator mkldnn::memory::desc() const {
    return desc;
}

MKLDNNMemoryDesc::MKLDNNMemoryDesc(const mkldnn::memory::desc& desc) :
    MemoryDesc(Shape(MKLDNNExtensionUtils::convertToSizeVector(desc.dims())), Mkldnn), desc(desc) {
    if (desc.data.format_kind == dnnl::impl::format_kind::any)
        IE_THROW(Unexpected) << "Memory format any is prohibited!";
}

// MKLDNNMemoryDesc::MKLDNNMemoryDesc(const std::vector<size_t>& _dims, mkldnn::memory::data_type dataType, mkldnn::memory::format_tag format) :
//     MKLDNNMemoryDesc(Shape(_dims), dataType, format) {}

// MKLDNNMemoryDesc::MKLDNNMemoryDesc(const std::vector<size_t>& _dims, mkldnn::memory::data_type dataType)
//         : MemoryDesc(Shape(_dims), Mkldnn), desc() {
//     InitializePlain(_dims, dataType);
// }

static const std::map<int, std::vector<mkldnn::memory::format_tag>> form_tags_by_ndims {
    {0, {
        mkldnn::memory::format_tag::a   // TODO :: really 1d layout for scalar??
     }}, {1, {
        mkldnn::memory::format_tag::a
     }}, {2, {
        mkldnn::memory::format_tag::ab,
        mkldnn::memory::format_tag::ba
     }}, {3, {
        mkldnn::memory::format_tag::abc,
        mkldnn::memory::format_tag::acb,
        mkldnn::memory::format_tag::bac,
        mkldnn::memory::format_tag::bca,
        mkldnn::memory::format_tag::cba,

        mkldnn::memory::format_tag::Abc16a,
        mkldnn::memory::format_tag::ABc16a16b,
        mkldnn::memory::format_tag::ABc4a4b,
        mkldnn::memory::format_tag::aBc16b,
        mkldnn::memory::format_tag::aBc32b,
        mkldnn::memory::format_tag::ABc16b16a,
        mkldnn::memory::format_tag::Abc4a,
        mkldnn::memory::format_tag::aBc4b,
        mkldnn::memory::format_tag::ABc4b16a4b,
        mkldnn::memory::format_tag::ABc2b8a4b,
        mkldnn::memory::format_tag::ABc16b16a4b,
        mkldnn::memory::format_tag::ABc16b16a2b,
        mkldnn::memory::format_tag::ABc4b4a,
        mkldnn::memory::format_tag::ABc8a16b2a,
        mkldnn::memory::format_tag::ABc8a8b,
        mkldnn::memory::format_tag::ABc8a4b,
        mkldnn::memory::format_tag::aBc8b,
        mkldnn::memory::format_tag::ABc8b16a2b,
        mkldnn::memory::format_tag::ABc8b8a,
        mkldnn::memory::format_tag::Acb16a,
        mkldnn::memory::format_tag::Acb4a,
        mkldnn::memory::format_tag::Acb8a,
        mkldnn::memory::format_tag::BAc16a16b,
        mkldnn::memory::format_tag::BAc16b16a,
     }}, {4, {                                 // Popular
        mkldnn::memory::format_tag::abcd,      // plain
        mkldnn::memory::format_tag::acdb,      // tail_c
        mkldnn::memory::format_tag::aBcd8b,    // blocked 8c
        mkldnn::memory::format_tag::aBcd16b,   // blocked 16c

        mkldnn::memory::format_tag::abdc,

        mkldnn::memory::format_tag::bacd,
        mkldnn::memory::format_tag::bcda,
        mkldnn::memory::format_tag::cdba,
        mkldnn::memory::format_tag::dcab,

        mkldnn::memory::format_tag::Abcd8a,
        mkldnn::memory::format_tag::Abcd16a,
        mkldnn::memory::format_tag::Abcd32a,
        mkldnn::memory::format_tag::ABcd16a16b,
        mkldnn::memory::format_tag::aBcd32b,
        mkldnn::memory::format_tag::ABcd16b16a,
        mkldnn::memory::format_tag::aBCd16b16c,
        mkldnn::memory::format_tag::aBCd16c16b,
        mkldnn::memory::format_tag::Abcd4a,
        mkldnn::memory::format_tag::aBcd4b,
        mkldnn::memory::format_tag::ABcd4b16a4b,
        mkldnn::memory::format_tag::ABcd2b8a4b,
        mkldnn::memory::format_tag::ABcd4b4a,
        mkldnn::memory::format_tag::ABcd4a4b,
        mkldnn::memory::format_tag::aBCd4c16b4c,
        mkldnn::memory::format_tag::aBCd2c8b4c,
        mkldnn::memory::format_tag::ABcd16b16a4b,
        mkldnn::memory::format_tag::ABcd16b16a2b,
        mkldnn::memory::format_tag::aBCd16c16b4c,
        mkldnn::memory::format_tag::aBCd16c16b2c,
        mkldnn::memory::format_tag::aBCd4c4b,
        mkldnn::memory::format_tag::aBCd4b4c,
        mkldnn::memory::format_tag::ABcd8a16b2a,
        mkldnn::memory::format_tag::ABcd8a8b,
        mkldnn::memory::format_tag::ABcd8a32b,
        mkldnn::memory::format_tag::ABcd32a32b,
        mkldnn::memory::format_tag::ABcd8a4b,

        mkldnn::memory::format_tag::ABcd8b16a2b,
        mkldnn::memory::format_tag::aBCd8b16c2b,
        mkldnn::memory::format_tag::ABcd8b8a,
        mkldnn::memory::format_tag::aBCd8b8c,
        mkldnn::memory::format_tag::aBCd8b4c,
        mkldnn::memory::format_tag::aBCd8c16b2c,
        mkldnn::memory::format_tag::aBCd8c8b,

        mkldnn::memory::format_tag::ABcd4a8b8a4b,
        mkldnn::memory::format_tag::ABcd2a8b8a2b,

        mkldnn::memory::format_tag::aBdc16b,
        mkldnn::memory::format_tag::aBdc4b,
        mkldnn::memory::format_tag::aBdc8b,
        mkldnn::memory::format_tag::aCBd16b16c,
        mkldnn::memory::format_tag::aCBd16c16b,
        mkldnn::memory::format_tag::Acdb16a,
        mkldnn::memory::format_tag::Acdb4a,
        mkldnn::memory::format_tag::Acdb8a,
        mkldnn::memory::format_tag::BAcd16a16b,
        mkldnn::memory::format_tag::BAcd16b16a,
        mkldnn::memory::format_tag::ABcd32a32b,
        mkldnn::memory::format_tag::Acdb32a,
        mkldnn::memory::format_tag::aBCd2b4c2b,
        mkldnn::memory::format_tag::aBCd2c4b2c,
        mkldnn::memory::format_tag::aBCd4b8c2b,
        mkldnn::memory::format_tag::aBCd4c8b2c,
    }}, {5, {                                   // Popular
        mkldnn::memory::format_tag::abcde,      // plain
        mkldnn::memory::format_tag::acdeb,      // tail_c
        mkldnn::memory::format_tag::aBcde8b,    // blocked 8c
        mkldnn::memory::format_tag::aBcde16b,   // blocked 16c

        mkldnn::memory::format_tag::abdec,
        mkldnn::memory::format_tag::acbde,
        mkldnn::memory::format_tag::bacde,
        mkldnn::memory::format_tag::bcdea,
        mkldnn::memory::format_tag::cdeba,
        mkldnn::memory::format_tag::decab,

        mkldnn::memory::format_tag::Abcde16a,
        mkldnn::memory::format_tag::Abcde32a,
        mkldnn::memory::format_tag::ABcde16a16b,
        mkldnn::memory::format_tag::aBcde32b,
        mkldnn::memory::format_tag::ABcde16b16a,
        mkldnn::memory::format_tag::aBCde16b16c,
        mkldnn::memory::format_tag::aBCde16c16b,
        mkldnn::memory::format_tag::aBCde2c8b4c,
        mkldnn::memory::format_tag::Abcde4a,
        mkldnn::memory::format_tag::aBcde4b,
        mkldnn::memory::format_tag::ABcde4b4a,
        mkldnn::memory::format_tag::ABcde4a4b,
        mkldnn::memory::format_tag::aBCde4b4c,
        mkldnn::memory::format_tag::aBCde4c16b4c,
        mkldnn::memory::format_tag::aBCde16c16b4c,
        mkldnn::memory::format_tag::aBCde16c16b2c,
        mkldnn::memory::format_tag::aBCde4c4b,
        mkldnn::memory::format_tag::Abcde8a,
        mkldnn::memory::format_tag::ABcde8a8b,
        mkldnn::memory::format_tag::ABcde8a4b,
        mkldnn::memory::format_tag::ABcde8b16a2b,
        mkldnn::memory::format_tag::ABcde4b16a4b,
        mkldnn::memory::format_tag::ABcde2b8a4b,
        mkldnn::memory::format_tag::aBCde8b16c2b,
        mkldnn::memory::format_tag::ABcde8b8a,
        mkldnn::memory::format_tag::aBCde8b8c,
        mkldnn::memory::format_tag::aBCde8b4c,
        mkldnn::memory::format_tag::aBCde4b8c8b4c,
        mkldnn::memory::format_tag::aBCde2b8c8b2c,
        mkldnn::memory::format_tag::aBCde8c16b2c,
        mkldnn::memory::format_tag::aBCde8c8b,
        mkldnn::memory::format_tag::aBdec16b,
        mkldnn::memory::format_tag::aBdec4b,
        mkldnn::memory::format_tag::aBdec8b,
        mkldnn::memory::format_tag::aCBde16b16c,
        mkldnn::memory::format_tag::aCBde16c16b,
        mkldnn::memory::format_tag::Acdeb16a,
        mkldnn::memory::format_tag::Acdeb4a,
        mkldnn::memory::format_tag::Acdeb8a,
        mkldnn::memory::format_tag::BAcde16b16a,
        mkldnn::memory::format_tag::BAcde16a16b,
        mkldnn::memory::format_tag::aBdec32b,
        mkldnn::memory::format_tag::aBCde2b4c2b,
        mkldnn::memory::format_tag::aBCde2c4b2c,
        mkldnn::memory::format_tag::aBCde4b8c2b,
        mkldnn::memory::format_tag::aBCde4c8b2c,
    }}, {6, {                                    // Popular
        mkldnn::memory::format_tag::abcdef,      // plain
        mkldnn::memory::format_tag::acbdef,      // permute
        mkldnn::memory::format_tag::defcab,      // permute
        mkldnn::memory::format_tag::aBcdef16b,   // blocked 16c

        mkldnn::memory::format_tag::aBCdef16b16c,
        mkldnn::memory::format_tag::aBCdef16c16b,
        mkldnn::memory::format_tag::aBcdef4b,
        mkldnn::memory::format_tag::aBCdef2c8b4c,
        mkldnn::memory::format_tag::aBCdef4c4b,
        mkldnn::memory::format_tag::aBCdef4b4c,
        mkldnn::memory::format_tag::aBCdef8b8c,
        mkldnn::memory::format_tag::aBCdef8b4c,
        mkldnn::memory::format_tag::aBCdef8c16b2c,
        mkldnn::memory::format_tag::aBCdef4c16b4c,
        mkldnn::memory::format_tag::aBCdef8c8b,

        mkldnn::memory::format_tag::aBdefc16b,
        mkldnn::memory::format_tag::aCBdef16c16b,
        mkldnn::memory::format_tag::aCBdef16b16c,
        mkldnn::memory::format_tag::aBdefc4b,
        mkldnn::memory::format_tag::aBdefc8b,

        mkldnn::memory::format_tag::Abcdef4a,
        mkldnn::memory::format_tag::Abcdef8a,
        mkldnn::memory::format_tag::Abcdef16a,
        mkldnn::memory::format_tag::Abcdef32a,
        mkldnn::memory::format_tag::aBCdef2b4c2b,
        mkldnn::memory::format_tag::aBCdef2c4b2c,
        mkldnn::memory::format_tag::aBCdef4b8c2b,
        mkldnn::memory::format_tag::aBCdef4c8b2c,
        }}
};

mkldnn::memory::format_tag MKLDNNMemoryDesc::getFormat() const {
    // TODO [OneDNN]: Previously it was a field of tdesc, but now the brute
    //                force search here. Please avoid of using this method.
    const auto ndims = desc.dims().size();

    // There are no suitable format_tag for this
    if (ndims == 0 || ndims > 6)
        return mkldnn::memory::format_tag::undef;

    for (const auto fmt : form_tags_by_ndims.at(ndims)) {
        if (this->isSame(fmt))
            return fmt;
    }

    return mkldnn::memory::format_tag::undef;
}

bool MKLDNNMemoryDesc::isSame(mkldnn::memory::format_tag fmt) const {
    mkldnn::memory::desc refDesc(desc.dims(), desc.data_type(), fmt);

    if (desc.data.ndims != refDesc.data.ndims)
        return false;

    if (desc.data.format_kind != dnnl_blocked || refDesc.data.format_kind != dnnl_blocked)
        IE_THROW() << "MKLDNNMemoryDesc::isSame is not implemented for non blocked memory format";

    auto actualBlkDesc = desc.data.format_desc.blocking;
    auto refBlkDesc = refDesc.data.format_desc.blocking;
    if (actualBlkDesc.inner_nblks != refBlkDesc.inner_nblks)
        return false;

    for (size_t i = 0; i < actualBlkDesc.inner_nblks; ++i)
        if (actualBlkDesc.inner_blks[i] != refBlkDesc.inner_blks[i])
            return false;

    for (size_t i = 0; i < actualBlkDesc.inner_nblks; ++i)
        if (actualBlkDesc.inner_idxs[i] != refBlkDesc.inner_idxs[i])
            return false;

    auto actualStrides = desc.data.format_desc.blocking.strides;
    auto refStrides = refDesc.data.format_desc.blocking.strides;

    std::vector<size_t> actualOrder(desc.data.ndims);
    {
        const auto dims = desc.dims();
        std::vector<size_t> total_block_per_dim(dims.size(), 1);
        const auto &blk_desc = desc.data.format_desc.blocking;
        for (int i = 0; i < blk_desc.inner_nblks; i++) {
            total_block_per_dim[blk_desc.inner_idxs[i]] *= blk_desc.inner_blks[i];
        }
        std::vector<size_t> outer_block_dims(std::begin(dims), std::begin(dims) + dims.size());
        for (size_t i = 0; i < outer_block_dims.size(); i++) {
            outer_block_dims[i] = div_up(outer_block_dims[i], total_block_per_dim[i]);
        }

        std::iota(actualOrder.begin(), actualOrder.end(), 0);
        std::sort(actualOrder.begin(), actualOrder.end(),
                  [&actualStrides, &outer_block_dims] (size_t ind_l, size_t ind_r) {
                      return (actualStrides[ind_l] > actualStrides[ind_r]) ||
                             (actualStrides[ind_l] == actualStrides[ind_r] && outer_block_dims[ind_l] > outer_block_dims[ind_r]);
                  });
    }

    std::vector<size_t> refOrder(refDesc.data.ndims);
    {
        const auto dims = refDesc.dims();
        std::vector<size_t> total_block_per_dim(dims.size(), 1);
        const auto &blk_desc = refDesc.data.format_desc.blocking;
        for (int i = 0; i < blk_desc.inner_nblks; i++) {
            total_block_per_dim[blk_desc.inner_idxs[i]] *= blk_desc.inner_blks[i];
        }
        std::vector<size_t> outer_block_dims(std::begin(dims), std::begin(dims) + dims.size());
        for (size_t i = 0; i < outer_block_dims.size(); i++) {
            outer_block_dims[i] = div_up(outer_block_dims[i], total_block_per_dim[i]);
        }

        std::iota(refOrder.begin(), refOrder.end(), 0);
        std::sort(refOrder.begin(), refOrder.end(),
                  [&refStrides, &outer_block_dims] (size_t ind_l, size_t ind_r) {
                      return (refStrides[ind_l] > refStrides[ind_r]) ||
                             (refStrides[ind_l] == refStrides[ind_r] && outer_block_dims[ind_l] > outer_block_dims[ind_r]);
                  });
    }

    if (actualOrder != refOrder) {
        return false;
    }

    return true;
}

bool MKLDNNMemoryDesc::blocksExtended() const {
    for (int i = 0; i < desc.data.ndims; i++) {
        if (desc.data.dims[i] != desc.data.padded_dims[i])
            return true;
    }
    return false;
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
        return this->getShape() == rhs.getShape() && this->getPrecision() == rhs.getPrecision() &&
               this->desc == rhs.as<MKLDNNMemoryDesc>()->desc;
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
    auto fmt = getFormat();
    return mkldnn::utils::fmt2str(fmt);
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
