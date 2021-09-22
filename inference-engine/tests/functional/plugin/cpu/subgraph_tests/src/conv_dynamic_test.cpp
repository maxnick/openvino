// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

using namespace ngraph;
using namespace InferenceEngine;
using namespace testing;

TEST(DynamicTests, DynamicConv) {
    std::shared_ptr<InferenceEngine::Core> ie = PluginCache::get().ie();
    auto targetDevice = CommonTestUtils::DEVICE_CPU;
    std::map<std::string, std::string> configuration;

    const std::string param_name1 = "Param_1";

    auto inputParams = builder::makeParams(element::f32, {Shape{1, 9, 7, 7}});
    auto paramOuts = helpers::convert2OutputVector(helpers::castOps2Nodes<op::Parameter>(inputParams));

    inputParams.front()->set_friendly_name(param_name1);

    std::shared_ptr<Node> conv;
    {
        const std::vector<size_t> kernelSize = {3, 3};
        const std::vector<size_t> strides = {1, 1};
        const std::vector<ptrdiff_t> padBegin = {0, 0};
        const std::vector<ptrdiff_t> padEnd = {0, 0};
        const std::vector<size_t> dilation = {1, 1};
        const size_t numOutChannels = 16;
        const op::PadType paddingType = op::PadType::EXPLICIT;
        conv = builder::makeConvolution(paramOuts[0], element::f32, kernelSize, strides, padBegin, padEnd, dilation, paddingType, numOutChannels);
    }

    ngraph::ResultVector results;

    for (int i = 0; i < conv->get_output_size(); i++)
        results.push_back(std::make_shared<ngraph::opset1::Result>(conv->output(i)));

    auto function = std::make_shared<ngraph::Function>(results, inputParams, "Dynamic conv");


    const std::string param_name2 = "Param_2";

    InferenceEngine::CNNNetwork cnnNet(function);
    std::map<std::string, ngraph::PartialShape> shapes;
    shapes[param_name1] = {ngraph::Dimension::dynamic(), 9, ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic()};
    //shapes[param_name1] = Shape{1, 3, 7, 7};
    //shapes[param_name2] = {ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic()};
    cnnNet.reshape(shapes);
    // Load CNNNetwork to target plugins
    auto execNet = ie->LoadNetwork(cnnNet, targetDevice, configuration);
    // Create InferRequest
    InferenceEngine::InferRequest req;
    ASSERT_NO_THROW(req = execNet.CreateInferRequest());
    InferenceEngine::StatusCode sts;

    InferenceEngine::Blob::Ptr blob1 = make_blob_with_precision({InferenceEngine::Precision::FP32, {1, 9, 7, 7}, InferenceEngine::Layout::NCHW});
    blob1->allocate();
    float *data1 = blob1->buffer().as<float *>();
    for (size_t i = 0; i < blob1->size(); i++)
        data1[i] = 1;
//    InferenceEngine::Blob::Ptr blob2 = make_blob_with_precision({InferenceEngine::Precision::FP32, {1, 3, 1, 8}, InferenceEngine::Layout::NCHW});
//    blob2->allocate();
//    float *data2 = blob2->buffer().as<float *>();
//    for (size_t i = 0; i < blob2->size(); i++)
//        data2[i] = 2;

    ASSERT_NO_THROW(req.SetBlob(param_name1, blob1));
//    ASSERT_NO_THROW(req.SetBlob(param_name2, blob2));
    req.Infer();
    sts = req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY);
    ASSERT_EQ(InferenceEngine::StatusCode::OK, sts);
    auto blob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first);
    ASSERT_EQ((InferenceEngine::SizeVector{1, 16, 5, 5}), blob->getTensorDesc().getDims());
    const float *out = blob->cbuffer().as<const float *>();
    for (size_t i = 0; i < blob->size(); i++)
        std::cout << out[i] << " ";
    std::cout << std::endl;

    blob1->setShape({5, 9, 15, 15});
//    blob2->setShape({2, 5, 1, 1});
    data1 = blob1->buffer().as<float *>();
    for (size_t i = 0; i < blob1->size(); i++)
        data1[i] = 3;
//    data2 = blob2->buffer().as<float *>();
//    for (size_t i = 0; i < blob2->size(); i++)
//        data2[i] = 4;
//
    ASSERT_NO_THROW(req.SetBlob(param_name1, blob1));
//    ASSERT_NO_THROW(req.SetBlob(param_name2, blob2));
    req.Infer();
    sts = req.Wait(InferenceEngine::InferRequest::WaitMode::RESULT_READY);
    ASSERT_EQ(InferenceEngine::StatusCode::OK, sts);
    blob = req.GetBlob(cnnNet.getOutputsInfo().begin()->first);
    ASSERT_EQ((InferenceEngine::SizeVector{5, 16, 13, 13}), blob->getTensorDesc().getDims());
    out = blob->cbuffer().as<const float *>();
    for (size_t i = 0; i < blob->size(); i++)
        std::cout << out[i] << " ";
    std::cout << std::endl;
}