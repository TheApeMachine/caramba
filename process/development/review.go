package development

import "github.com/theapemachine/amsh/utils"

type Review struct {
	CodeReviewProcess  string   `json:"code_review_process" jsonschema:"description=Process for reviewing code and providing feedback"`
	PeerReview         string   `json:"peer_review" jsonschema:"description=Process for peer validation and approval"`
	ValidationCriteria []string `json:"validation_criteria" jsonschema:"description=Criteria for validating the final product"`
}

func (review *Review) GenerateSchema() string {
	return utils.GenerateSchema[Review]()
}
