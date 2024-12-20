package development

import "github.com/theapemachine/amsh/utils"

type Testing struct {
	TestCases      []TestCase `json:"test_cases" jsonschema:"description=List of test cases covering system functionalities"`
	TestStrategy   string     `json:"test_strategy" jsonschema:"description=Overall testing strategy, e.g., unit, integration, performance testing"`
	QualityMetrics []string   `json:"quality_metrics" jsonschema:"description=List of metrics used to assess code quality"`
}

type TestCase struct {
	Description    string `json:"description" jsonschema:"description=Description of the test case"`
	ExpectedResult string `json:"expected_result" jsonschema:"description=Expected outcome of the test"`
}

func (testing *Testing) GenerateSchema() string {
	return utils.GenerateSchema[Testing]()
}
