// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "blob_dump.h"
#include "blob_factory.hpp"
#include "mkldnn_memory.h"
#include "mkldnn_extension_utils.h"
#include <nodes/common/cpu_memcpy.h>

#include "common/memory_desc_wrapper.hpp"

#include <fstream>
#include <cpu_memory_desc_utils.h>

using namespace InferenceEngine;

namespace MKLDNNPlugin {

// IEB file format routine
static unsigned char IEB_MAGIC[4] = {'I', 'E', 'B', '0'};
static unsigned char NO_SCALES = 0xFF;

struct IEB_HEADER {
    unsigned char magic[4];
    unsigned char ver[2];

    unsigned char precision;  // 0-8
    unsigned char ndims;
    unsigned int  dims[7];  // max is 7-D blob

    unsigned char scaling_axis;  // FF - no scaling
    unsigned char reserved[3];

    unsigned long data_offset;
    unsigned long data_size;
    unsigned long scaling_data_offset;
    unsigned long scaling_data_size;
};

static IEB_HEADER prepare_header(const MKLDNNMemoryDesc& desc) {
    IEB_HEADER header = {};

    header.magic[0] = IEB_MAGIC[0];
    header.magic[1] = IEB_MAGIC[1];
    header.magic[2] = IEB_MAGIC[2];
    header.magic[3] = IEB_MAGIC[3];

    // IEB file format version 0.1
    header.ver[0] = 0;
    header.ver[1] = 1;

    header.precision = desc.getPrecision();

    if (desc.getShape().getRank() > 7)
        IE_THROW() << "Dumper support max 7D blobs";

    header.ndims = desc.getShape().getRank();
    const auto &dims = desc.getShape().getStaticDims();
    for (int i = 0; i < header.ndims; i++)
        header.dims[i] = dims[i];

    header.scaling_axis = NO_SCALES;

    return header;
}

static MKLDNNMemoryDesc parse_header(IEB_HEADER &header) {
    if (header.magic[0] != IEB_MAGIC[0] ||
        header.magic[1] != IEB_MAGIC[1] ||
        header.magic[2] != IEB_MAGIC[2] ||
        header.magic[3] != IEB_MAGIC[3])
        IE_THROW() << "Dumper cannot parse file. Wrong format.";

    if (header.ver[0] != 0 ||
        header.ver[1] != 1)
        IE_THROW() << "Dumper cannot parse file. Unsupported IEB format version.";

    const auto prc = MKLDNNExtensionUtils::IEPrecisionToDataType(Precision(static_cast<Precision::ePrecision>(header.precision)));
    SizeVector dims(header.ndims);
    for (int i = 0; i < header.ndims; i++)
        dims[i] = header.dims[i];

    return MKLDNNMemoryDesc{MKLDNNDims(dims), prc, MKLDNNMemory::GetPlainFormatByRank(dims.size()) };
}

static void prepare_plain_data(const MKLDNNMemoryPtr &memory, std::vector<uint8_t> &data) {
    const auto mdesc = memory->GetDescWithType<MKLDNNMemoryDesc>();
    const auto size = memory->GetSize();
    data.resize(size);

    // check if it already plain
    if (mdesc.checkGeneralLayout(GeneralLayout::ncsp)) {
        cpu_memcpy(data.data(), reinterpret_cast<const uint8_t*>(memory->GetPtr()), size);
        return;
    }

    // Copy to plain
    mkldnn::memory::desc desc = mdesc;
    mkldnn::impl::memory_desc_wrapper mem_wrp(desc.data);
    size_t data_size = mem_wrp.nelems();
    const void *ptr = memory->GetData();

    switch (mdesc.getPrecision()) {
        case Precision::FP32:
        case Precision::I32: {
            auto *pln_blob_ptr = reinterpret_cast<int32_t *>(data.data());
            auto *blob_ptr = reinterpret_cast<const int32_t *>(ptr);
            for (size_t i = 0; i < data_size; i++)
                pln_blob_ptr[i] = blob_ptr[mem_wrp.off_l(i)];
            break;
        }
        case Precision::BF16: {
            auto *pln_blob_ptr = reinterpret_cast<int16_t *>(data.data());
            auto *blob_ptr = reinterpret_cast<const int16_t *>(ptr);
            for (size_t i = 0; i < data_size; i++)
                pln_blob_ptr[i] = blob_ptr[mem_wrp.off_l(i)];
            break;
        }
        case Precision::I8:
        case Precision::U8: {
            auto *pln_blob_ptr = reinterpret_cast<int8_t*>(data.data());
            auto *blob_ptr = reinterpret_cast<const int8_t *>(ptr);
            for (size_t i = 0; i < data_size; i++)
                pln_blob_ptr[i] = blob_ptr[mem_wrp.off_l(i)];
            break;
        }
        default:
            IE_THROW() << "Dumper. Unsupported precision";
    }
}

void BlobDumper::dump(std::ostream &stream) const {
    if (memory == nullptr)
        IE_THROW() << "Dumper cannot dump. Memory is not allocated.";

    IEB_HEADER header = prepare_header(memory->GetDescWithType<MKLDNNMemoryDesc>());
    std::vector<uint8_t> data;
    prepare_plain_data(this->memory, data);

    header.data_offset = sizeof(header);
    header.data_size = data.size();
    header.scaling_data_offset = 0;
    header.scaling_data_size = 0;

    stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
    stream.write(reinterpret_cast<char*>(data.data()), data.size());
}

void BlobDumper::dumpAsTxt(std::ostream &stream) const {
    if (memory == nullptr)
        IE_THROW() << "Dumper cannot dump. Memory is not allocated.";

    const auto dims = memory->GetDims();

    // Header like "U8 4D shape: 2 3 224 224 ()
    stream << MKLDNNExtensionUtils::DataTypeToIEPrecision(memory->GetDataType()).name() << " "
           << dims.size() << "D "
           << "shape: ";
    for (size_t d : dims) stream << d << " ";
    stream << "(" << memory->GetElementsCount() << ")" <<
    " by address 0x" << std::hex << reinterpret_cast<const long long *>(memory->GetData()) << std::dec <<std::endl;

    mkldnn::memory::desc mkldnnDesc = memory->GetDescWithType<MKLDNNMemoryDesc>();
    mkldnn::impl::memory_desc_wrapper mem_wrp(mkldnnDesc.data);
    const void *ptr = memory->GetData();

    size_t data_size = memory->GetElementsCount();
    switch (mkldnnDesc.data_type()) {
        case mkldnn::memory::data_type::f32 : {
            auto *blob_ptr = reinterpret_cast<const float*>(ptr);
            for (size_t i = 0; i < data_size; i++)
                stream << blob_ptr[mem_wrp.off_l(i)] << std::endl;
            break;
        }
        case mkldnn::memory::data_type::bf16:
        {
            auto *blob_ptr = reinterpret_cast<const int16_t*>(ptr);
            for (size_t i = 0; i < data_size; i++) {
                int i16n = blob_ptr[mem_wrp.off_l(i)];
                i16n = i16n << 16;
                float fn = *(reinterpret_cast<const float *>(&i16n));
                stream << fn << std::endl;
            }
            break;
        }
        case mkldnn::memory::data_type::s32: {
            auto *blob_ptr = reinterpret_cast<const int32_t*>(ptr);
            for (size_t i = 0; i < data_size; i++)
                stream << blob_ptr[mem_wrp.off_l(i)] << std::endl;
            break;
        }
        case mkldnn::memory::data_type::s8: {
            auto *blob_ptr = reinterpret_cast<const int8_t*>(ptr);
            for (size_t i = 0; i < data_size; i++)
                stream << static_cast<int>(blob_ptr[mem_wrp.off_l(i)]) << std::endl;
            break;
        }
        case mkldnn::memory::data_type::u8: {
            auto *blob_ptr = reinterpret_cast<const uint8_t*>(ptr);
            for (size_t i = 0; i < data_size; i++)
                stream << static_cast<int>(blob_ptr[mem_wrp.off_l(i)]) << std::endl;
            break;
        }
        default:
            IE_THROW() << "Dumper. Unsupported precision";
    }
}

BlobDumper BlobDumper::read(std::istream &stream) {
    IEB_HEADER header;
    stream.read(reinterpret_cast<char*>(&header), sizeof(header));

    const auto desc = parse_header(header);

    BlobDumper res(desc);
    stream.seekg(header.data_offset, stream.beg);
    stream.read(reinterpret_cast<char *>(res.getDataPtr()), header.data_size);

    return res;
}

BlobDumper BlobDumper::read(const std::string &file_path) {
    std::ifstream file;
    file.open(file_path);
    if (!file.is_open())
        IE_THROW() << "Dumper cannot open file " << file_path;

    auto res = read(file);
    file.close();
    return res;
}

void BlobDumper::dump(const std::string &dump_path) const {
    std::ofstream dump_file;
    dump_file.open(dump_path);
    if (!dump_file.is_open())
        IE_THROW() << "Dumper cannot create dump file";

    dump(dump_file);
    dump_file.close();
}

void BlobDumper::dumpAsTxt(const std::string& dump_path) const {
    std::ofstream dump_file;
    dump_file.open(dump_path);
    if (!dump_file.is_open())
        IE_THROW() << "Dumper cannot create dump file";

    dumpAsTxt(dump_file);
    dump_file.close();
}

}  // namespace MKLDNNPlugin
