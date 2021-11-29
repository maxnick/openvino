// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <vector>
#include <memory>
#include <caseless.hpp>
#include <cpu_lru_cache.h>

namespace MKLDNNPlugin {

#define MAX_ELTWISE_INPUTS 7
#define MAX_ELTWISE_DIM_RANK 12

struct jit_eltwise_params {
    size_t inputs_number;
    size_t input_size;

    InferenceEngine::Precision src_prc[MAX_ELTWISE_INPUTS];
    InferenceEngine::Precision dst_prc;

    VectorDims dims;
    VectorDims src_offsets[MAX_ELTWISE_INPUTS];
    VectorDims dst_offsets;
    VectorDims oc_offsets;

    size_t src_size[MAX_ELTWISE_INPUTS];
    size_t dst_size;
    size_t oc_size;

    size_t work_amount;
};

struct jit_eltwise_call_args_ptrs {
    const void *src_ptr[MAX_ELTWISE_INPUTS];
    void *dst_ptr;
    //ptr to flat buffer of postop data pointer
    const void** post_op_data;
};

struct jit_eltwise_call_args_indexes {
    size_t indexes[MAX_ELTWISE_DIM_RANK];
};

class MKLDNNEltwiseNode;

struct jit_uni_eltwise_kernel {
    void (*ker_)(const jit_eltwise_call_args_ptrs*, const jit_eltwise_call_args_indexes*);

    void operator()(const jit_eltwise_call_args_ptrs* const_args, const jit_eltwise_call_args_indexes* indexes) {
        assert(ker_);
        ker_(const_args, indexes);
    }

    explicit jit_uni_eltwise_kernel(jit_eltwise_params jep) : ker_(nullptr), jep_(std::move(jep)) {}
    virtual ~jit_uni_eltwise_kernel() {}

    virtual void create_ker() = 0;

    jit_eltwise_params jep_;
};

class MKLDNNEltwiseNode : public MKLDNNNode {
public:
    struct EltwiseData {
        Algorithm algo;
        mkldnn::algorithm mkldnnAlgorithm;
        float alpha;
        float beta;
        float gamma;

        bool operator==(const EltwiseData& rhs) const noexcept;
    };

    class IEltwiseExecutor {
    public:
        IEltwiseExecutor() = default;
        virtual void exec(const jit_eltwise_call_args_ptrs &args_ptrs, const VectorDims &dims_out) = 0;
        virtual size_t getBatchDimIdx() const = 0;
        virtual const VectorDims& getOutDims() const = 0;
        virtual ~IEltwiseExecutor() = default;
    };

    struct EltwiseKey {
        std::vector<EltwiseData> eltwise_data;
        std::vector<Type> ops_list;
        VectorDims outBlkDims;
        VectorDims outOrder;
        std::vector<VectorDims> dims_in;
        std::vector<InferenceEngine::Precision> inpPrc;
        InferenceEngine::Precision outPrc;
        mkldnn::post_ops post_ops;
        bool useDynBatch;
        bool isOptimized;

        size_t hash() const;
        bool operator==(const EltwiseKey& rhs) const noexcept;
    };

public:
    MKLDNNEltwiseNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void selectOptimalPrimitiveDescriptor() override;
    void initOptimalPrimitiveDescriptor() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;
    bool canBeInPlace() const override;
    bool canFuse(const MKLDNNNodePtr& node) const override;
    void appendPostOps(mkldnn::post_ops& ops, const VectorDims &postOpDims, int align = -1) override;
    void appendBinPostOps(mkldnn::post_ops& ops, const VectorDims &postOpDims, std::vector<MKLDNNMemoryPtr>& binaryPostOpsMem) override;
    void fuseInto(MKLDNNNodePtr& parentNode) override;
    InferenceEngine::Precision getRuntimePrecision() const override;

    float getAlpha() const { return alpha; }
    float getBeta() const { return beta; }
    float getGamma() const { return gamma; }
    MKLDNNMemoryPtr scalesMemory;
    MKLDNNMemoryPtr shiftsMemory;
    mkldnn::algorithm getMKLDNNAlgorithm() const { return mkldnnAlgorithm; }

    bool isWithBroadcast();
    bool isSpecialConvolutionAddFusing() const { return specialConvolutionAddFusing; }

    void createPrimitive() override;

    std::vector<VectorDims> shapeInfer() const override;
    bool needPrepareParams() const override;
    void prepareParams() override;

    void executeDynamicImpl(mkldnn::stream strm) override { execute(strm); }

    enum Policy {
        PerChannel,
        PerTensor,
        Undefined,
    };

    Policy getPolicy() const { return policy; }

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;


private:
    using executorPtr = std::shared_ptr<IEltwiseExecutor>;
    executorPtr execPtr = nullptr;

    using cacheType = LruCache<EltwiseKey, executorPtr>;
    static cacheType cache;

    Policy policy;

    mkldnn::algorithm mkldnnAlgorithm = mkldnn::algorithm::undef;

    bool canUseOptimizedImpl = false;
    bool isDynBatchEnabled = false;
    bool specialConvolutionAddFusing = false;
    size_t inputNum = 0;
    std::vector<ptrdiff_t> start_offset_in = {};
    ptrdiff_t start_offset_out = 0;

    // blocked dims for which kernel compiled and params prepared
    std::vector<VectorDims> currentInBlkDims = {};

    float alpha = 0;
    float beta = 0;
    float gamma = 0;

    std::vector<float> scales = {};
    std::vector<float> shifts = {};
    std::vector<float> scalesBuffer = {};
    std::vector<float> shiftsBuffer = {};

    std::vector<MKLDNNMemoryPtr> memPtrs = {};
    std::vector<std::vector<const float*>> fqDataPtrs;

    using Initializer = std::function<void(const std::shared_ptr<ngraph::Node>&, MKLDNNEltwiseNode& node)>;
    static const std::map<const ngraph::DiscreteTypeInfo, Initializer> initializers;

    static Policy determinePolicy(const std::shared_ptr<ngraph::Node>& op);

    size_t getOpInputsNum() const;
};

}  // namespace MKLDNNPlugin
