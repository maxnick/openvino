// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>

#include <memory>
#include <string>
#include <vector>

namespace MKLDNNPlugin {

enum MatrixNmsSortResultType {
    CLASSID,  // sort selected boxes by class id (ascending) in each batch element
    SCORE,    // sort selected boxes by score (descending) in each batch element
    NONE      // do not guarantee the order in each batch element
};

enum MatrixNmsDecayFunction { GAUSSIAN, LINEAR };

class MKLDNNMatrixNmsNode : public MKLDNNNode {
public:
    MKLDNNMatrixNmsNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr& cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override {};
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    // input
    static const size_t NMS_BOXES = 0;
    static const size_t NMS_SCORES = 1;

    // output
    static const size_t NMS_SELECTED_OUTPUTS = 0;
    static const size_t NMS_SELECTED_INDICES = 1;
    static const size_t NMS_VALID_OUTPUTS = 2;

    size_t m_numBatches;
    size_t m_numBoxes;
    size_t m_numClasses;
    size_t m_maxBoxesPerBatch;

    MatrixNmsSortResultType m_sortResultType;
    bool m_sortResultAcrossBatch;
    float m_scoreThreshold;
    int m_nmsTopk;
    int m_keepTopk;
    int m_backgroundClass;
    MatrixNmsDecayFunction m_decayFunction;
    float m_gaussianSigma;
    float m_postThreshold;
    bool m_normalized;

    struct Rectangle {
        Rectangle(float x_left, float y_left, float x_right, float y_right) : x1 {x_left}, y1 {y_left}, x2 {x_right}, y2 {y_right} {}

        Rectangle() = default;

        float x1 = 0.0f;
        float y1 = 0.0f;
        float x2 = 0.0f;
        float y2 = 0.0f;
    };

    struct BoxInfo {
        BoxInfo(const Rectangle& r, int64_t idx, float sc, int64_t batch_idx, int64_t class_idx)
            : box {r}, index {idx}, batchIndex {batch_idx}, classIndex {class_idx}, score {sc} {}

        BoxInfo() = default;

        Rectangle box;
        int64_t index = -1;
        int64_t batchIndex = -1;
        int64_t classIndex = -1;
        float score = 0.0f;
    };
    std::string errorPrefix;
    const std::string inType = "input", outType = "output";
    std::vector<int64_t> m_numPerBatch;
    std::vector<std::vector<int64_t>> m_numPerBatchClass;
    std::vector<BoxInfo> m_filteredBoxes;
    std::vector<int> m_classOffset;
    size_t m_realNumClasses;
    size_t m_realNumBoxes;
    float (*m_decay_fn)(float, float, float);
    void checkPrecision(const InferenceEngine::Precision prec, const std::vector<InferenceEngine::Precision> precList, const std::string name,
                        const std::string type);

    size_t nmsMatrix(const float* boxesData, const float* scoresData, BoxInfo* filterBoxes, const int64_t batchIdx, const int64_t classIdx);
};

}  // namespace MKLDNNPlugin
