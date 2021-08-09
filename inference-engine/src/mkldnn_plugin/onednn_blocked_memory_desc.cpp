// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "onednn_blocked_memory_desc.h"
#include <dnnl_types.h>
#include <common/memory_desc_wrapper.hpp>

namespace dnnl {
namespace impl {
extern status_t fill_blocked(memory_desc_t &md, std::vector<int> &perm,
                             std::vector<int> &inner_blks,
                             std::vector<int> &inner_idxs);
} // namespace impl
} // namespace dnnl

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

OnednnBlockedMemoryDesc::OnednnBlockedMemoryDesc(InferenceEngine::Precision prc, const Shape& shape) : MKLDNNMemoryDesc(shape), MemoryDesc(shape, OneDnnBlocked) {
    InitializePlain(shape, MKLDNNExtensionUtils::IEPrecisionToDataType(prc));
}

/**
 * Construct from blocked parameters
 *
 * IE  IOhw_4i16o4i   dims(N) = {32, 64, 128, 128}
 *   blockedDims  {4, 2, 128, 128, 4, 16, 4}                      // total dims(inner, outermost, auto blocked/padded). Generally sorted by strides.
 *   strides      {8388608, 4194304,  32768, 256, 64,  4, 1}      // strides for blockedDims, growing sequence
 *   order        {1, 0,   2,   3, 1,  0, 1}                      // matching to original dims
 *
 *   All vectors blockedDims/strides/order have same size equals total num of internal blocked dims(inner_dims + outer_dims)
 *
 *   Tensor descriptor filing is not deterministic. It allows any permutation of index which keeps order of
 *   real dims spliting.
 *      for {1, 0, 2, 3, 1, 0, 1} we can swap elements [1] <=> [4]
 *      but not [0]<=>[4] because it break splitting original dims into internal blocked dims
 *   Normalization of representation: Make strides growing but keep layout same as original. Not all
 *   layout allow us to meet normalize form of tensor desc.
 *
 *   Limitation of conversion first N elements of order should be permutation of [0,1,2 ... N]
 */
OnednnBlockedMemoryDesc::OnednnBlockedMemoryDesc(InferenceEngine::Precision prc, const Shape& shape, const std::vector<size_t>& blockedDims,
                                                 const std::vector<size_t>& order, size_t offsetPadding, const std::vector<size_t>& offsetPaddingToData,
                                                 const std::vector<size_t>& strides) : MKLDNNMemoryDesc(shape), MemoryDesc(shape, OneDnnBlocked) {
    using namespace mkldnn;
    // scalar case
    if (shape.getRank() == 0) {
        desc.data.format_kind = dnnl_blocked;
        desc.data.data_type = memory::convert_to_c(MKLDNNMemory::convertToDataType(prc));
        desc.data.ndims = 1;
        desc.data.dims[0] = 1;
        desc.data.padded_dims[0] = 1;
        desc.data.format_desc.blocking.strides[0] = 1;
        desc.data.padded_offsets[0] = 0;
        desc.data.offset0 = offsetPadding;
        return;
    }

    if (order.size() != blockedDims.size()) {
        IE_THROW() << "Can not construct OnednnBlockedMemoryDesc, order and blocked dims must have equals size";
    }

    if (!offsetPaddingToData.empty() && offsetPaddingToData.size() != order.size()) {
        IE_THROW() << "Can not construct OnednnBlockedMemoryDesc, offsetPaddingToData must have equal size with order and blocked dims";
    }

    if (!strides.empty() && strides.size() != order.size()) {
        IE_THROW() << "Can not construct OnednnBlockedMemoryDesc, strides must have equal size with order and blocked dims";
    }

    if (std::any_of(order.begin(), order.end(), [](size_t val) { return val == Shape::UNDEFINED_DIM; })) {
        IE_THROW() << "OnednnBlockedMemoryDesc doesn't support undefined order.";
    }

    if (std::any_of(blockedDims.begin() + shape.getRank(), blockedDims.end(), [](size_t val) { return val == Shape::UNDEFINED_DIM; })) {
        IE_THROW() << "OnednnBlockedMemoryDesc doesn't support undefined blockedDims.";
    }

    auto dims = MKLDNNExtensionUtils::convertToDnnlDims(shape.getDims());

    size_t outer_ndims = dims.size();
    size_t inner_ndims = order.size() - dims.size();

    if (!strides.empty()) {
        bool is_descending_strides = true;
        for (int i = 1; i < strides.size(); i++) {
            is_descending_strides &= (strides[i - 1] >= strides[i]);
        }

        // TODO: That's strong constrains and can be mitigated. IE::TensorDesc allow to transpose blocked dims
        //       and may be we can achieve correct "descending strides" form which allow conversion.
        if (!is_descending_strides)
            IE_THROW() << "Can not construct OnednnBlockedMemoryDesc from strides: " << vec2str(strides);
    }

    std::vector<size_t> outer_order(outer_ndims, outer_ndims + 1); // outer_order[i] is index of stride for i-th dimension
    for (size_t i = 0; i < outer_ndims; i++) {
        outer_order[order[i]] = i;
    }
    bool outer_is_correct_permutation_of_n =
            std::find(outer_order.begin(), outer_order.end(), outer_ndims + 1) == outer_order.end();

    if (!outer_is_correct_permutation_of_n)
        IE_THROW() << "Can not construct OnednnBlockedMemoryDesc because of incorrect order: " << vec2str(order);

    if (!strides.empty() && std::none_of(strides.begin(), strides.end(), [](size_t x) { return Shape::UNDEFINED_DIM == x; })) {
        bool inner_block_are_dense = one_of(strides.back(), 0, 1);  // stride 1 - is dense case, 0 - broad casted
        for (int i = outer_ndims; i < strides.size() - 1; i++) {
            inner_block_are_dense &= (strides[i] == strides[i + 1] * blockedDims[i + 1]);
        }

        if (!inner_block_are_dense)
            IE_THROW() << "Can not construct OnednnBlockedMemoryDesc from strides: " << vec2str(strides) << " inner blocks are not dense.";
    }

    // Fill general memory desc fields
    desc.data.format_kind = dnnl_blocked;
    desc.data.extra.flags = 0;
    desc.data.data_type = memory::convert_to_c(MKLDNNMemory::convertToDataType(prc));
    desc.data.ndims = dims.size();
    desc.data.offset0 = offsetPadding;
    std::copy(dims.begin(), dims.end(), desc.data.dims);

    if (!offsetPaddingToData.empty()) {
        bool inner_pad_offsets_is_zero = std::all_of(offsetPaddingToData.begin() + outer_ndims, offsetPaddingToData.end(),
                                                     [](size_t pad) { return pad == 0; });

        if (!inner_pad_offsets_is_zero)
            IE_THROW() << "Can not construct OnednnBlockedMemoryDesc, inner pad offsets is not zero: " << vec2str(offsetPaddingToData);
        auto dnnlPaddedOffsets = MKLDNNExtensionUtils::convertToDnnlDims(offsetPaddingToData);
        std::copy(dnnlPaddedOffsets.begin(), dnnlPaddedOffsets.begin() + outer_ndims, desc.data.padded_offsets);
    } else {
        std::fill(std::begin(desc.data.padded_offsets), std::begin(desc.data.padded_offsets) + outer_ndims, 0);
    }

    std::fill(desc.data.padded_dims, desc.data.padded_dims + outer_ndims, 1);
    auto dnnlBlkDims = MKLDNNExtensionUtils::convertToDnnlDims(blockedDims);

    for (size_t i = 0; i < order.size(); i++) {
        auto idx = order[i];
        if (desc.data.padded_dims[idx] != DNNL_RUNTIME_DIM_VAL && dnnlBlkDims[i] != DNNL_RUNTIME_DIM_VAL) {
            desc.data.padded_dims[idx] *= dnnlBlkDims[i];
        } else {
            desc.data.padded_dims[idx] = DNNL_RUNTIME_DIM_VAL;
        }
    }

    // Fill blocking desc
    auto &dnn_blk_desc = desc.data.format_desc.blocking;
    dnn_blk_desc.inner_nblks = inner_ndims;
    std::copy(dnnlBlkDims.end() - inner_ndims, dnnlBlkDims.end(), dnn_blk_desc.inner_blks);
    std::copy(order.end() - inner_ndims, order.end(), dnn_blk_desc.inner_idxs);

    if (strides.empty()) {
        if (std::any_of(dnnlBlkDims.begin(), dnnlBlkDims.end(), [](memory::dim val) { return val == DNNL_RUNTIME_DIM_VAL; })) {
            std::fill(std::begin(dnn_blk_desc.strides), std::begin(dnn_blk_desc.strides) + outer_ndims, DNNL_RUNTIME_DIM_VAL);
        } else {
            //TODO [DS]: phase 2: refactor
            std::vector<memory::dim> tmpStrides(order.size());
            tmpStrides[order.size() - 1] = 1;
            for (size_t i = 2; i <= order.size(); i++) {
                tmpStrides[order.size() - i] = tmpStrides[order.size() - (i - 1)] * dnnlBlkDims[blockedDims.size() - (i - 1)];
            }
            for (size_t i = 0; i < outer_ndims; i++) {
                dnn_blk_desc.strides[i] = tmpStrides[outer_order[i]];
            }
        }
    } else {
        for (size_t i = 0; i < outer_ndims; i++) {
            auto dnnlStrides = MKLDNNExtensionUtils::convertToDnnlDims(strides);
            dnn_blk_desc.strides[i] = dnnlStrides[outer_order[i]];
        }
    }

    this->order = order;
}

OnednnBlockedMemoryDesc::OnednnBlockedMemoryDesc(const Shape& shape, mkldnn::memory::data_type dataType, mkldnn::memory::format_tag format) :
                                    MKLDNNMemoryDesc(shape), MemoryDesc(shape, OneDnnBlocked) {
    using namespace mkldnn;
    if (format == memory::format_tag::any)
        IE_THROW(Unexpected) << "Can't create mkldnn::desc with any format";

    const auto dims = shape.getDims();
    if (format == memory::format_tag::undef) {
        InitializePlain(shape, dataType);
    } else {
        if (format == memory::format_tag::x && shape.getRank() == 0) {
            desc = mkldnn::memory::desc(mkldnn::memory::dims(1, 1), dataType, format);
        } else {
            desc = mkldnn::memory::desc(MKLDNNExtensionUtils::convertToDnnlDims(dims), dataType, format);
        }

        std::vector<size_t> perm;
        std::vector<size_t> inner_blks;
        std::vector<size_t> inner_idxs;

        mkldnn::impl::memory_desc_wrapper::compute_blocking(mkldnn::memory::convert_to_c(format), perm, inner_blks, inner_idxs);

        order.swap(perm);
        order.insert(order.end(), inner_idxs.begin(), inner_idxs.end());
    }
}

const std::vector<size_t> OnednnBlockedMemoryDesc::getBlockDims() const {
    const auto dims = desc.dims();

    const auto &blk_desc = desc.data.format_desc.blocking;

    const size_t outer_ndims = dims.size();
    const size_t inner_ndims = blk_desc.inner_nblks;
    const size_t total_ndims = outer_ndims + inner_ndims;

    // total inner block size. in case of 4i16o4i will be {16, 16, 1, 1}
    std::vector<size_t> total_block_per_dim(outer_ndims, 1);
    for (int i = 0; i < inner_ndims; i++) {
        total_block_per_dim[blk_desc.inner_idxs[i]] *= blk_desc.inner_blks[i];
    }
    // blocked dims
    // [dims via new_outer_order with auto pad] U [inner_blk_dims]
    std::vector<size_t> outer_block_dims = MKLDNNExtensionUtils::convertToSizeVector(dims);
    for (size_t i = 0; i < outer_block_dims.size(); i++) {
        if (outer_block_dims[i] != Shape::UNDEFINED_DIM) {
            outer_block_dims[i] = div_up(outer_block_dims[i], total_block_per_dim[i]);
        }
    }

    // order of outer dims. In case of IOhw_ will be {1, 0, 2, 3}
    std::vector<size_t> outer_order(outer_ndims);
    std::copy(order.begin(), order.begin() + outer_ndims, outer_order.begin());

    SizeVector blk_dims(total_ndims, 0);
    std::copy(blk_desc.inner_blks, blk_desc.inner_blks + blk_desc.inner_nblks,
              blk_dims.end() - blk_desc.inner_nblks);
    std::transform(outer_order.begin(), outer_order.end(), blk_dims.begin(),
                   [&] (size_t i) { return outer_block_dims[i]; });
    return blk_dims;
}

const std::vector<size_t> OnednnBlockedMemoryDesc::getStrides() const {
    const auto dims = desc.dims();

    const auto &blk_desc = desc.data.format_desc.blocking;

    const size_t outer_ndims = dims.size();
    const size_t inner_ndims = blk_desc.inner_nblks;
    const size_t total_ndims = outer_ndims + inner_ndims;

    // strides of inner dims. In case of 4i16o4i will be {64, 4, 1}
    std::vector<size_t> inner_strides(inner_ndims, 1);
    for (size_t i = 1; i < blk_desc.inner_nblks; i++) {
        inner_strides[blk_desc.inner_nblks - 1 - i] = inner_strides[blk_desc.inner_nblks - i] * blk_desc.inner_blks[blk_desc.inner_nblks - i];
    }

    // order of outer dims. In case of IOhw_ will be {1, 0, 2, 3}
    std::vector<size_t> outer_order(outer_ndims);
    std::copy(order.begin(), order.begin() + outer_ndims, outer_order.begin());

    // blocked strides
    // [outer_strides via new_outer_order] U [inner_strides]
    SizeVector blk_strides(total_ndims, 0);
    std::copy(inner_strides.rbegin(), inner_strides.rend(), blk_strides.rbegin());
    std::transform(outer_order.begin(), outer_order.end(), blk_strides.begin(),
                   [&](size_t i) { return blk_desc.strides[i] == DNNL_RUNTIME_DIM_VAL ? Shape::UNDEFINED_DIM : blk_desc.strides[i]; });
    return blk_strides;
}

const std::vector<size_t> OnednnBlockedMemoryDesc::getOrder() const {
    return order;
}

const std::vector<size_t> OnednnBlockedMemoryDesc::getOffsetPaddingToData() const {
    return std::vector<size_t>(std::begin(desc.data.padded_offsets), std::begin(desc.data.padded_offsets) + getOrder().size());
}

size_t OnednnBlockedMemoryDesc::getOffsetPadding() const {
    return desc.data.offset0;
}

bool OnednnBlockedMemoryDesc::isCompatible(const MemoryDesc& rhs) const {
    const MemoryDesc* pRhs = &rhs;
    if (auto blockingDesc = dynamic_cast<const BlockedMemoryDesc*>(pRhs)) {
        return BlockedMemoryDesc::isCompatible(blockingDesc);
    } else {
        return false;
    }
}

static bool array_cmp_weak(const dnnl_dim_t *a1, const dnnl_dim_t *a2, size_t size) {
    for (size_t i = 0; i < size; ++i)
        if (a1[i] != a2[i] && a1[i] != DNNL_RUNTIME_DIM_VAL && a2[i] != DNNL_RUNTIME_DIM_VAL) return false;
    return true;
}

bool OnednnBlockedMemoryDesc::isCompatible(const OnednnBlockedMemoryDesc& rhs) const {
    using namespace dnnl;
    using namespace impl;
    using namespace impl::utils;
    if (this->getShape() != rhs.getShape() || this->getPrecision() != rhs.getPrecision()) {
        return false;
    }

    if (this->desc == rhs.desc) {
        return true;
    }
    memory_desc_wrapper wrappedThis(this->desc.data);
    memory_desc_wrapper wrappedRhs(rhs.desc.data);
    if (one_of(wrappedThis.format_kind(), format_kind::undef, format_kind::any))
        return false;

    const auto &blk = wrappedThis.blocking_desc();
    const auto &r_blk = wrappedRhs.blocking_desc();

    int stride_start = wrappedThis.ndims() > 0 && wrappedThis.dims()[0] == 1 ? 1 : 0;  // ignore batch axis stride if batch size == 1

    // Here is a slightly modified version of mkldnn::impl::memory_desc_wrapper::similar_to() call able to skip specific strides check
    // and use weak comparison
    return wrappedThis.ndims() == wrappedRhs.ndims()
           && wrappedThis.format_kind() == wrappedRhs.format_kind()
           && wrappedThis.data_type() == wrappedRhs.data_type()
           && array_cmp_weak(wrappedThis.dims(), wrappedRhs.dims(), wrappedThis.ndims())
           && array_cmp_weak(blk.strides + stride_start, r_blk.strides + stride_start, wrappedThis.ndims() - stride_start)
           && blk.inner_nblks == r_blk.inner_nblks
           && array_cmp(blk.inner_blks, r_blk.inner_blks, blk.inner_nblks)
           && array_cmp(blk.inner_idxs, r_blk.inner_idxs, blk.inner_nblks)
           && array_cmp_weak(wrappedThis.padded_dims(), wrappedRhs.padded_dims(), wrappedRhs.ndims())
           && array_cmp_weak(wrappedThis.padded_offsets(), wrappedRhs.padded_offsets(), wrappedThis.ndims())
           && dimsEqualWeak(wrappedThis.offset0(), wrappedRhs.offset0());
}

OnednnBlockedMemoryDesc::OnednnBlockedMemoryDesc(const mkldnn::memory::desc& mdesc) :
                MKLDNNMemoryDesc(Shape(MKLDNNExtensionUtils::convertToSizeVector(mdesc.dims()))),
                MemoryDesc(MKLDNNExtensionUtils::convertToSizeVector(mdesc.dims()), OneDnnBlocked) {
    desc = mdesc;
    if (desc.data.format_kind == dnnl::impl::format_kind::any)
        IE_THROW(Unexpected) << "Memory format any is prohibited!";

    mkldnn::impl::memory_desc_wrapper descWrapped(desc.data);
    if (!descWrapped.is_blocking_desc())
        IE_THROW(Unexpected) << "Can't create OnednnBlockedMemoryDesc from OnednnBlockedMemoryDesc";

    if (descWrapped.has_runtime_dims_or_strides()) {
        IE_THROW(Unexpected) << "Cannot calculate order from undefined dims or strides";
    }

    const auto dims = desc.dims();

    const auto &blk_desc = descWrapped.blocking_desc();

    const size_t outer_ndims = dims.size();
    const size_t inner_ndims = blk_desc.inner_nblks;
    const size_t total_ndims = outer_ndims + inner_ndims;

    // strides of inner dims. In case of 4i16o4i will be {64, 4, 1}
    std::vector<size_t> inner_strides(inner_ndims, 1);
    for (size_t i = 1; i < blk_desc.inner_nblks; i++) {
        inner_strides[blk_desc.inner_nblks - 1 - i] = inner_strides[blk_desc.inner_nblks - i] * blk_desc.inner_blks[blk_desc.inner_nblks - i];
    }

    // total inner block size. in case of 4i16o4i will be {16, 16, 1, 1}
    std::vector<size_t> total_block_per_dim(outer_ndims, 1);
    for (int i = 0; i < inner_ndims; i++) {
        total_block_per_dim[blk_desc.inner_idxs[i]] *= blk_desc.inner_blks[i];
    }
    std::vector<size_t> outer_block_dims(std::begin(dims), std::begin(dims) + outer_ndims);
    for (size_t i = 0; i < outer_block_dims.size(); i++) {
        outer_block_dims[i] = div_up(outer_block_dims[i], total_block_per_dim[i]);
    }

    // order of outer dims. In case of IOhw_ will be {1, 0, 2, 3}
    std::vector<size_t> outer_order(outer_ndims);
    std::iota(outer_order.begin(), outer_order.end(), 0);
    std::sort(outer_order.begin(), outer_order.end(),
              [&blk_desc, &outer_block_dims](size_t ind_l, size_t ind_r) {
                  return (blk_desc.strides[ind_l] > blk_desc.strides[ind_r]) ||
                         (blk_desc.strides[ind_l] == blk_desc.strides[ind_r] && outer_block_dims[ind_l] > outer_block_dims[ind_r]);
              });


    // blocked order
    // [new_outer_order] U [inner_idxs]
    SizeVector blk_order(total_ndims, 0);
    std::copy(outer_order.begin(), outer_order.end(), blk_order.begin());
    std::copy(blk_desc.inner_idxs, blk_desc.inner_idxs + blk_desc.inner_nblks, blk_order.begin() + dims.size());
    order.swap(blk_order);
}

bool OnednnBlockedMemoryDesc::hasLayoutType(LayoutType layoutType) const {
    switch (layoutType) {
        case LayoutType::ncsp:
            return isPlainFormat();
        case LayoutType::nspc:
            return isTailCFormat();
        case LayoutType::nCsp8c:
            return isBlockedCFormat(8);
        case LayoutType::nCsp16c:
            return isBlockedCFormat(16);
        default:
            return false;
    }
}

// TODO [DS]: move to BlockedMemoryDesc after caching
bool OnednnBlockedMemoryDesc::isPlainFormat() const {
    if (desc.data.format_kind != dnnl_blocked)
        return false;

    if (shape.getRank() != order.size()) {
        return false;
    }
    for (size_t i = 0; i < order.size(); ++i) {
        if (order[i] != i) {
            return false;
        }
    }
    return true;
}

bool OnednnBlockedMemoryDesc::isBlockedCFormat(size_t blk_size) const {
    const auto &blocking = desc.data.format_desc.blocking;

    if (desc.data.format_kind !=dnnl_blocked ||
        blocking.inner_nblks != 1 ||
        blocking.inner_idxs[0] != 1)
        return false;

    if ((order.size() - shape.getRank()) != 1) {
        return false;
    }
    for (size_t i = 0; i < order.size() - 1; ++i) {
        if (order[i] != i) {
            return false;
        }
    }
    if (blk_size != UNREACHABLE_DIM && blk_size != blocking.inner_blks[0]) {
            return false;
    }

    return true;
}

bool OnednnBlockedMemoryDesc::isTailCFormat() const {
    if (desc.data.format_kind != dnnl_blocked)
        return false;

    if (shape.getRank() < 3) {
        return false;
    }
    if (shape.getRank() != order.size()) {
        return false;
    }
    if (!std::is_sorted(order.begin(), --order.end())) {
        return false;
    }
    if (order.back() != 1) {
        return false;
    }
    return true;
}

std::unique_ptr<MemoryDesc> OnednnBlockedMemoryDesc::cloneWithNewDimsImp(const std::vector<size_t> &dims) const {
    IE_THROW(Unexpected) << "Cannot clone non blocked oneDNN desc with new dims";
    using namespace dnnl::impl::utils;
    auto mklDims = MKLDNNExtensionUtils::convertToDnnlDims(dims);
    mkldnn::memory::desc newMklDesc = desc;
    array_copy(newMklDesc.data.dims, mklDims.data(), mklDims.size());
    std::vector<int> perm(order.begin(), order.begin() + mklDims.size());
    auto& blockingDesc = newMklDesc.data.format_desc.blocking;
    auto numInnerBlks = blockingDesc.inner_nblks;
    std::vector<int> innerBlks(std::begin(blockingDesc.inner_blks), std::begin(blockingDesc.inner_blks) + numInnerBlks);
    std::vector<int> innerIdxs(std::begin(blockingDesc.inner_idxs), std::begin(blockingDesc.inner_idxs) + numInnerBlks);
    auto retCode = dnnl::impl::fill_blocked(newMklDesc.data, perm, innerBlks, innerIdxs);
    if (retCode != dnnl::impl::status::success) {
        IE_THROW() << "Can not clone OnednnBlockedMemoryDesc with dims: " << dims2str(dims);
    }
    return MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(newMklDesc);
}

void OnednnBlockedMemoryDesc::InitializePlain(const Shape& shape, mkldnn::memory::data_type dataType) {
    const auto ndims = shape.getRank();
    const auto dims = shape.getDims();
    mkldnn::memory::dims plain_strides;
    if (std::any_of(dims.begin(), dims.end(), [](size_t val) { return val == Shape::UNDEFINED_DIM; })) {
        plain_strides.resize(ndims, DNNL_RUNTIME_DIM_VAL);
    } else {
        plain_strides.resize(ndims, 1);
        for (size_t i = 1; i < ndims; i++) {
            plain_strides[ndims - i -1] = plain_strides[ndims - i] * dims[ndims - i];
        }
    }

    desc = {MKLDNNExtensionUtils::convertToDnnlDims(dims), dataType, plain_strides};

    order.resize(ndims);
    std::iota(order.begin(), order.end(), 0);
}
