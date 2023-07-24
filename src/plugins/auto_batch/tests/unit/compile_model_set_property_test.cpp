// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "mock_common.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "openvino/core/dimension_tracker.hpp"
#include "unit_test_utils/mocks/cpp_interfaces/interface/mock_icore.hpp"

using ::testing::_;
using ::testing::AnyNumber;
using ::testing::AtLeast;
using ::testing::Eq;
using ::testing::MatcherCast;
using ::testing::Matches;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::StrEq;
using ::testing::StrNe;
using ::testing::Throw;

using namespace ov::mock_autobatch_plugin;

using set_property_param = std::tuple<ov::AnyMap,  // Property need to be set
                                      bool>;       // Throw exception

class CompileModelSetPropertyTest : public ::testing::TestWithParam<set_property_param> {
public:
    ov::AnyMap m_properities;
    bool m_throw_exception;
    std::shared_ptr<NiceMock<MockICore>> m_core;
    std::shared_ptr<NiceMock<MockAutoBatchInferencePlugin>> m_plugin;
    std::shared_ptr<ov::Model> m_model;

    // Mock execNetwork
    ov::SoPtr<MockICompiledModel> m_mock_compile_model;
    std::shared_ptr<MockICompiledModel> m_mock_i_compile_model;
    std::shared_ptr<NiceMock<MockIPlugin>> m_hardware_plugin;

    std::shared_ptr<ov::ICompiledModel> m_auto_batch_compile_model;

public:
    static std::string getTestCaseName(testing::TestParamInfo<set_property_param> obj) {
        ov::AnyMap properities;
        bool throw_exception;
        std::tie(properities, throw_exception) = obj.param;

        std::string res;
        for (auto& c : properities) {
            res += "_" + c.first + "_" + c.second.as<std::string>();
        }
        if (throw_exception)
            res += "throw";

        return res;
    }

    void TearDown() override {
        m_core.reset();
        m_plugin.reset();
        m_model.reset();
        m_mock_i_compile_model.reset();
        m_mock_compile_model = {};
        m_auto_batch_compile_model.reset();
    }

    void SetUp() override {
        std::tie(m_properities, m_throw_exception) = this->GetParam();
        m_model = ngraph::builder::subgraph::makeMultiSingleConv();
        m_core = std::shared_ptr<NiceMock<MockICore>>(new NiceMock<MockICore>());
        m_plugin =
            std::shared_ptr<NiceMock<MockAutoBatchInferencePlugin>>(new NiceMock<MockAutoBatchInferencePlugin>());
        m_plugin->set_core(m_core);
        m_hardware_plugin = std::shared_ptr<NiceMock<MockIPlugin>>(new NiceMock<MockIPlugin>());
        m_mock_i_compile_model = std::make_shared<NiceMock<MockICompiledModel>>(m_model, m_hardware_plugin);
        m_mock_compile_model = {m_mock_i_compile_model, {}};

        ON_CALL(*m_core,
                compile_model(MatcherCast<const std::shared_ptr<const ov::Model>&>(_),
                              MatcherCast<const std::string&>(_),
                              _))
            .WillByDefault(Return(m_mock_compile_model));

        ON_CALL(*m_core,
                compile_model(MatcherCast<const std::shared_ptr<const ov::Model>&>(_),
                              MatcherCast<const ov::SoPtr<ov::IRemoteContext>&>(_),
                              _))
            .WillByDefault(Return(m_mock_compile_model));

        ON_CALL(*m_core, get_property(_, StrEq("PERFORMANCE_HINT")))
            .WillByDefault(Return(ov::hint::PerformanceMode::THROUGHPUT));

        ON_CALL(*m_core, get_property(_, StrEq("OPTIMAL_BATCH_SIZE"), _))
            .WillByDefault(Return(static_cast<unsigned int>(16)));

        ON_CALL(*m_core, get_property(_, StrEq("PERFORMANCE_HINT_NUM_REQUESTS")))
            .WillByDefault(Return(static_cast<uint32_t>(12)));

        ON_CALL(*m_core, get_property(_, StrEq("GPU_MEMORY_STATISTICS"), _))
            .WillByDefault([](const std::string& device, const std::string& key, const ov::AnyMap& options) {
                std::map<std::string, uint64_t> ret = {{"xyz", 1024}};
                return ret;
            });

        ON_CALL(*m_core, get_property(_, StrEq("GPU_DEVICE_TOTAL_MEM_SIZE"), _)).WillByDefault(Return("10240"));

        const ov::AnyMap configs = {{"AUTO_BATCH_TIMEOUT", "200"}, {"AUTO_BATCH_DEVICE_CONFIG", "CPU(16)"}};

        ASSERT_NO_THROW(m_auto_batch_compile_model = m_plugin->compile_model(m_model, configs));
    }
};

TEST_P(CompileModelSetPropertyTest, CompileModelSetPropertyTestCase) {
    if (m_throw_exception)
        ASSERT_ANY_THROW(m_auto_batch_compile_model->set_property(m_properities));
    else
        ASSERT_NO_THROW(m_auto_batch_compile_model->set_property(m_properities));
}

const std::vector<set_property_param> compile_model_set_property_param_test = {
    set_property_param{{{CONFIG_KEY(AUTO_BATCH_TIMEOUT), std::uint32_t(100)}}, false},
    set_property_param{{{"INCORRECT_CONFIG", 2}}, true},
};

INSTANTIATE_TEST_SUITE_P(smoke_AutoBatch_BehaviorTests,
                         CompileModelSetPropertyTest,
                         ::testing::ValuesIn(compile_model_set_property_param_test),
                         CompileModelSetPropertyTest::getTestCaseName);
