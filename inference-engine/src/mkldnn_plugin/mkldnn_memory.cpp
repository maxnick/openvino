// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <algorithm>
#include <numeric>
#include <unordered_set>

#include "utils/general_utils.h"

#include <mkldnn_types.h>
#include <dnnl_types.h>
#include <common/memory_desc_wrapper.hpp>
#include "mkldnn_memory.h"
#include "mkldnn_extension_utils.h"
#include "nodes/common/cpu_memcpy.h"
#include "nodes/common/cpu_convert.h"
#include "mkldnn/ie_mkldnn.h"
#include "cpu_shape.h"
#include "memory_descs/onednn_blocked_memory_desc.h"

using namespace InferenceEngine;
using namespace mkldnn;

namespace MKLDNNPlugin {
namespace {
    inline void setSubnormalsToZero(float *data, size_t size) {
        uint32_t *u32data = reinterpret_cast<uint32_t *>(data);
        for (size_t i = 0; i < size; ++i) {
            if ((u32data[i] & (0xFF << 23)) == 0) {
                u32data[i] = 0;
            }
        }
    }
}   // namespace

MKLDNNMemory::MKLDNNMemory(const mkldnn::engine& eng) : eng(eng) {}

size_t MKLDNNMemory::GetSize() const {
    uint8_t itemSize = MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type(GetDataType()));
    return GetElementsCount() * itemSize;
}

size_t MKLDNNMemory::GetElementsCount() const {
    auto desc = GetDescriptor();
    std::vector<int> dims(desc.data.padded_dims,
                          desc.data.padded_dims + desc.data.ndims);
    return std::accumulate(std::begin(dims), std::end(dims), (size_t) 1, std::multiplies<size_t>());
}

void MKLDNNMemory::Create(const memory::dims& dims, memory::data_type data_type, memory::format_tag format, const void* data) {
    if (format == memory::format_tag::undef) {
        format = memory::format_tag::any;
    }

    memory::desc desc = OnednnBlockedMemoryDesc(Shape(MKLDNNExtensionUtils::convertToSizeVector(dims)), data_type, format);

    Create(desc, data);
}

void MKLDNNMemory::Create(const mkldnn::memory::desc& desc, const void *data, bool pads_zeroing) {
    if (data == nullptr) {
        prim.reset(new memory(desc, eng));

        size_t real_size = 0;
        if (desc.data.format_kind == dnnl_format_kind_wino)
            return;
        auto desc_loc = prim->get_desc().data;
        if (desc_loc.ndims > 0) {
            real_size = static_cast<size_t>(desc_loc.padded_dims[0]);
            for (int i = 1; i < desc_loc.ndims; i++) {
                real_size *= desc_loc.padded_dims[i];
            }
        }
    } else {
        // MKLDNN accepts not a const data, probably need to remove some level of consteness in a call stack

        // ========================
        // Equivalent of constructor memory(const primitive_desc &desc, void *hdl)
        // but with ability to skipp pads zeroing.
        prim.reset(new memory(desc, eng, DNNL_MEMORY_NONE));
        if (pads_zeroing)
            prim->set_data_handle(const_cast<void*>(data));
        else
            prim->set_data_handle_no_pads_proc(const_cast<void*>(data));
        //
        // ========================
    }
}

void MKLDNNMemory::Create(const MemoryDesc &desc, const void *data, bool pads_zeroing) {
    Create(desc.clone(), data, pads_zeroing);
}

void MKLDNNMemory::Create(MemoryDescPtr desc, const void* data, bool pads_zeroing) {
    pMemDesc = std::move(desc);
    if (nullptr != data) {
        useExternalStorage = true;
    } else {
        useExternalStorage = false;
    }

    if (pMemDesc->isDefined()) {
        const auto a = *MemoryDescUtils::convertToMKLDNNMemoryDesc(*pMemDesc);
        Create(mkldnn::memory::desc(a), data, pads_zeroing);
    } else {
        //delayed dynamic allocation
        size_t maxMemSize = pMemDesc->getMaxMemSize();
        size_t dummySize = MemoryDesc::UNDEFINED_SIZE == maxMemSize ? 1 : maxMemSize;
        OnednnBlockedMemoryDesc dummyDesc(InferenceEngine::Precision::U8, Shape(InferenceEngine::SizeVector{dummySize}));
        Create(mkldnn::memory::desc(dummyDesc), data, false);  // no pads zeroing
    }
    size_t newUpperBound = prim->get_desc().get_size();
    if (newUpperBound > memUpperBound) {
        memUpperBound = newUpperBound;
    }
}


void MKLDNNMemory::reorderData(const MKLDNNMemory &input, const MKLDNNMemory &output, size_t size) {
    if (size != 0)
        IE_ASSERT(size <= output.GetDescriptor().get_size());
    if (input.GetDescriptor() == output.GetDescriptor()) {
        auto srcPtr = static_cast<uint8_t*>(input.GetPtr());
        auto dstPtr = static_cast<uint8_t*>(output.GetPtr());

        auto copySize = size == 0 ? output.GetSize() : size;
        cpu_memcpy(dstPtr, srcPtr, copySize);
    } else {
        std::unique_ptr<mkldnn::reorder> pReorder;
        std::shared_ptr<memory> srcMemoryPtr;
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

                cpu_convert(data, tmpBuff.data(), MKLDNNExtensionUtils::DataTypeToIEPrecision(input.GetDataType()),
                            MKLDNNExtensionUtils::DataTypeToIEPrecision(output.GetDataType()), input.GetElementsCount());

                MKLDNNMemory tmpMem(output.eng);
                tmpMem.Create(input.GetDims(), output.GetDataType(), input.GetMKLDNNDesc().getFormat(), tmpBuff.data());

                pReorder = std::unique_ptr<mkldnn::reorder>(new mkldnn::reorder(tmpMem.GetPrimitive(), output.GetPrimitive()));
                srcMemoryPtr = tmpMem.prim;
            } else {
                throw;
            }
        }
        if (pReorder) {
            mkldnn::stream loc_stream(output.eng, stream::flags::default_order);
            pReorder->execute(loc_stream, *srcMemoryPtr, *output.prim);
        } else {
            IE_THROW() << "Could not make mkldnn reorder.";
        }
    }
}

// TODO: It should be done via wrap into Memory;
void MKLDNNMemory::SetData(memory::data_type dataType, memory::format_tag format, const void* data, size_t size, bool ftz) const {
    IE_ASSERT(!one_of(format, memory::format_tag::undef, memory::format_tag::any));

    auto dst_desc = GetDescriptor();
    memory::desc src_desc{dst_desc.dims(), dataType, format};

    IE_ASSERT(size <= dst_desc.get_size());

    if (dst_desc == src_desc) {
        uint8_t itemSize = MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type(dataType));
        uint8_t* dataPtr = static_cast<uint8_t*>(GetData());
        // We cannot support strides for i/o blobs because it affects performance.
        dataPtr += itemSize * prim->get_desc().data.offset0;
        cpu_memcpy(dataPtr, data, size);
    } else {
        auto memData = this->GetDescriptor().data;
        memory::dims dims(memData.dims, memData.dims + memData.ndims);

        MKLDNNMemory src(this->eng);
        src.Create(dims, dataType, format, data);

        reorderData(src, *this);
    }
    if (ftz
        && dataType == memory::data_type::f32
        && prim->get_desc().data.format_kind != dnnl_format_kind_wino
        && GetDataType() != memory::data_type::bf16) {
        // Internal blobs haven't strides yet.
        auto *memData = static_cast<float *>(GetData());
        memData += prim->get_desc().data.offset0;
        setSubnormalsToZero(memData, GetSize() / sizeof(float));
    }
}

void MKLDNNMemory::SetData(const MKLDNNMemory& src, size_t size, bool ftz) const {
    reorderData(src, *this, size);

    if (ftz
        && src.GetDataType() == memory::data_type::f32
        && prim->get_desc().data.format_kind != dnnl_format_kind_wino
        && GetDataType() != memory::data_type::bf16) {
        // Internal blobs haven't strides yet.
        auto *memData = static_cast<float *>(GetData());
        memData += prim->get_desc().data.offset0;
        setSubnormalsToZero(memData, GetSize() / sizeof(float));
    }
}

void MKLDNNMemory::FillZero() {
    void* dataPtr = GetData();
    memset(dataPtr, 0, GetSize());
}

memory::format_tag MKLDNNMemory::GetPlainFormatByRank(size_t rank) {
    switch (rank) {
        case 0:
        case 1:
            return memory::format_tag::a;
        case 2:
            return memory::format_tag::ab;
        case 3:
            return memory::format_tag::abc;
        case 4:
            return memory::format_tag::abcd;
        case 5:
            return memory::format_tag::abcde;
        case 6:
            return memory::format_tag::abcdef;
        default:
            return memory::format_tag::undef;
    }
}

InferenceEngine::Layout MKLDNNMemory::GetPlainLayout(const memory::dims& dims) {
    switch (dims.size()) {
        case 0: return Layout::SCALAR;
        case 1: return Layout::C;
        case 2: return Layout::NC;
        case 3: return Layout::CHW;
        case 4: return Layout::NCHW;
        case 5: return Layout::NCDHW;
        default:
            return Layout::BLOCKED;
    }
}

Precision MKLDNNMemory::convertToIePrec(memory::data_type dataType) {
    return MKLDNNExtensionUtils::DataTypeToIEPrecision(dataType);
}

memory::data_type MKLDNNMemory::convertToDataType(const InferenceEngine::Precision &precision) {
    return MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
}

memory::format_tag MKLDNNMemory::Convert(const InferenceEngine::Layout layout) {
    switch (layout) {
        case NCHW:
            return memory::format_tag::nchw;
        case NHWC:
            return memory::format_tag::nhwc;
        case NCDHW:
            return memory::format_tag::ncdhw;
        case NDHWC:
            return memory::format_tag::ndhwc;
        case CHW:
            return memory::format_tag::tnc;
        case NC:
            return memory::format_tag::nc;
        case C:
            return memory::format_tag::x;
        case SCALAR:
            return memory::format_tag::x;
        default:
            return memory::format_tag::undef;
    }
}

std::string MKLDNNMemory::formatToString(memory::format_tag fmt) {
    return mkldnn::utils::fmt2str(fmt);
}

void *MKLDNNMemory::GetPtr() const  {
    auto ptr = static_cast<uint8_t*>(GetData());
    auto md = GetDescriptor().data;
    mkldnn::impl::memory_desc_wrapper wrapper(md);
    ptr += wrapper.offset0() * wrapper.data_type_size();
    return ptr;
}

template<>
MKLDNNMemoryDescPtr MKLDNNMemory::GetDescWithType<MKLDNNMemoryDesc, 0, 0>() const {
    if (pMemDesc->getType() & MemoryDescType::Mkldnn) {
        return std::unique_ptr<MKLDNNMemoryDesc>(dynamic_cast<MKLDNNMemoryDesc *>(pMemDesc->clone().release()));
    } else if (pMemDesc->getType() == MemoryDescType::CpuBlocked) {
        return MemoryDescUtils::convertToMKLDNNMemoryDesc(*(pMemDesc->as<BlockedMemoryDesc>()));
    } else {
        IE_THROW() << "Can not convert unsupported memory descriptor";
    }
}

void MKLDNNMemory::redefineDesc(const MemoryDesc& desc) {
    redefineDesc(desc.clone());
}

void MKLDNNMemory::redefineDesc(MemoryDescPtr desc) {
    if (useExternalStorage) {
        size_t descMaxSize = desc->getMaxMemSize();
        if (MemoryDesc::UNDEFINED_SIZE == descMaxSize) {
            IE_THROW() << "Can not reset descriptor, memory upper bound is unknown.";
        }
        if (descMaxSize <= memUpperBound) {
            this->Create(std::move(desc), prim->get_data_handle(), false);
        } else {
            this->Create(std::move(desc), nullptr, false);
        }
    } else {
        this->Create(std::move(desc), nullptr, false);
    }
}

template<>
BlockedMemoryDescPtr MKLDNNMemory::GetDescWithType<BlockedMemoryDesc, 0, 0>() const {
    if (pMemDesc->getType() & MemoryDescType::Blocked) {
        return std::unique_ptr<BlockedMemoryDesc>(dynamic_cast<BlockedMemoryDesc *>(pMemDesc->clone().release()));
    } else {
        IE_THROW() << "Can not convert unsupported memory descriptor";
    }
}

}  // namespace MKLDNNPlugin
