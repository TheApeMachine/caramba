package process

// import (
// 	"github.com/theapemachine/amsh/utils"
// )

// /*
// NarrativeAnalysis transforms abstract patterns into coherent storylines.
// */
// type NarrativeAnalysis struct {
// 	StoryElements []StoryElement `json:"story_elements" jsonschema:"title=Story Elements,description=Building blocks of the narrative,required"`
// 	StoryFlow     StoryFlow      `json:"story_flow" jsonschema:"title=Story Flow,description=Narrative progression and connections,required"`
// 	ThemeMapping  ThemeMapping   `json:"theme_mapping" jsonschema:"title=Theme Mapping,description=Abstract concepts mapped to narrative elements,required"`
// }

// func (narrative *NarrativeAnalysis) SystemPrompt(key string) string {
// 	return utils.SystemPrompt(key, "narrative", utils.GenerateSchema[NarrativeAnalysis]())
// }

// type StoryElement struct {
// 	ID           string     `json:"id" jsonschema:"required,description=Unique identifier for the element"`
// 	Type         string     `json:"type" jsonschema:"required,description=Type of story element (character, event, concept)"`
// 	Description  string     `json:"description" jsonschema:"required,description=Human-friendly description"`
// 	AbstractLink []string   `json:"abstract_link" jsonschema:"required,description=Links to abstract concepts from deep analysis"`
// 	Properties   Properties `json:"properties" jsonschema:"required,description=Additional narrative properties"`
// }

// type StoryFlow struct {
// 	Sequences   []Sequence   `json:"sequences" jsonschema:"required,description=Ordered sequences of story elements"`
// 	Connections []Connection `json:"connections" jsonschema:"required,description=Relationships between elements"`
// 	Progression float64      `json:"progression" jsonschema:"required,description=Overall narrative progression"`
// }

// /*
// AnalogyAnalysis creates relatable comparisons for abstract concepts.
// */
// type AnalogyAnalysis struct {
// 	Analogies     []Analogy      `json:"analogies" jsonschema:"required,description=Set of analogies for abstract concepts"`
// 	Mappings      []AnalogyMap   `json:"mappings" jsonschema:"required,description=How abstract concepts map to familiar ideas"`
// 	Relationships []Relationship `json:"relationships" jsonschema:"required,description=How analogies relate to each other"`
// }

// func (analogy *AnalogyAnalysis) SystemPrompt(key string) string {
// 	return utils.SystemPrompt(key, "analogy", utils.GenerateSchema[AnalogyAnalysis]())
// }

// type Analogy struct {
// 	ID            string  `json:"id" jsonschema:"required,description=Unique identifier for the analogy"`
// 	SourceConcept string  `json:"source_concept" jsonschema:"required,description=The familiar concept used in the analogy"`
// 	TargetConcept string  `json:"target_concept" jsonschema:"required,description=The abstract concept being explained"`
// 	Explanation   string  `json:"explanation" jsonschema:"required,description=Human-friendly explanation"`
// 	Strength      float64 `json:"strength" jsonschema:"required,description=How well the analogy fits"`
// }

// /*
// PracticalAnalysis converts abstract insights into actionable steps.
// */
// type PracticalAnalysis struct {
// 	Actions        []Action       `json:"actions" jsonschema:"required,description=Concrete actions derived from abstract insights"`
// 	Dependencies   []Dependency   `json:"dependencies" jsonschema:"required,description=Relationships between actions"`
// 	Implementation Implementation `json:"implementation" jsonschema:"required,description=How to implement the actions"`
// }

// func (practical *PracticalAnalysis) SystemPrompt(key string) string {
// 	return utils.SystemPrompt(key, "practical", utils.GenerateSchema[PracticalAnalysis]())
// }

// /*
// ContextAnalysis grounds abstract insights in user's specific situation.
// */
// type ContextAnalysis struct {
// 	Relevance       []RelevanceMap   `json:"relevance" jsonschema:"required,description=How insights apply to context"`
// 	Constraints     []Constraint     `json:"constraints" jsonschema:"required,description=Contextual limitations"`
// 	Opportunities   []Opportunity    `json:"opportunities" jsonschema:"required,description=Context-specific possibilities"`
// 	Recommendations []Recommendation `json:"recommendations" jsonschema:"required,description=Context-aware suggestions"`
// }

// func (context *ContextAnalysis) SystemPrompt(key string) string {
// 	return utils.SystemPrompt(key, "context", utils.GenerateSchema[ContextAnalysis]())
// }

// type RelevanceMap struct {
// 	InsightID     string  `json:"insight_id" jsonschema:"required,description=ID of the abstract insight"`
// 	Applicability float64 `json:"applicability" jsonschema:"required,description=How relevant the insight is"`
// 	Adaptation    string  `json:"adaptation" jsonschema:"required,description=How to adapt the insight"`
// 	Context       string  `json:"context" jsonschema:"required,description=Specific contextual factors"`
// }

// type Constraint struct {
// 	ID           string   `json:"id" jsonschema:"required,description=Unique identifier for the constraint"`
// 	Type         string   `json:"type" jsonschema:"required,description=Type of constraint (technical, resource, time, etc)"`
// 	Description  string   `json:"description" jsonschema:"required,description=Human-readable description of the constraint"`
// 	Severity     float64  `json:"severity" jsonschema:"required,description=How limiting this constraint is"`
// 	Flexibility  float64  `json:"flexibility" jsonschema:"required,description=How negotiable this constraint is"`
// 	DeepInsights []string `json:"deep_insights" jsonschema:"required,description=Links to relevant deep analysis insights"`
// }

// type Opportunity struct {
// 	ID           string   `json:"id" jsonschema:"required,description=Unique identifier for the opportunity"`
// 	Type         string   `json:"type" jsonschema:"required,description=Type of opportunity (innovation, optimization, etc)"`
// 	Description  string   `json:"description" jsonschema:"required,description=Human-readable description of the opportunity"`
// 	Potential    float64  `json:"potential" jsonschema:"required,description=Estimated value of the opportunity"`
// 	Feasibility  float64  `json:"feasibility" jsonschema:"required,description=How achievable this opportunity is"`
// 	Requirements []string `json:"requirements" jsonschema:"required,description=What's needed to realize this opportunity"`
// }

// type Recommendation struct {
// 	ID            string   `json:"id" jsonschema:"required,description=Unique identifier for the recommendation"`
// 	Priority      int      `json:"priority" jsonschema:"required,description=Importance ranking"`
// 	Description   string   `json:"description" jsonschema:"required,description=Human-readable recommendation"`
// 	Rationale     string   `json:"rationale" jsonschema:"required,description=Why this is recommended"`
// 	Prerequisites []string `json:"prerequisites" jsonschema:"required,description=What needs to be in place first"`
// 	Impact        float64  `json:"impact" jsonschema:"required,description=Expected effect of this recommendation"`
// }

// func (recommendation *Recommendation) SystemPrompt(key string) string {
// 	return utils.SystemPrompt(key, "recommendation", utils.GenerateSchema[Recommendation]())
// }

// type Implementation struct {
// 	ID          string   `json:"id" jsonschema:"required,description=Unique identifier for the implementation"`
// 	Description string   `json:"description" jsonschema:"required,description=How to implement the actions"`
// 	Steps       []string `json:"steps" jsonschema:"required,description=Ordered list of implementation steps"`
// 	Resources   []string `json:"resources" jsonschema:"required,description=Required resources for implementation"`
// 	Timeline    string   `json:"timeline" jsonschema:"required,description=Expected implementation schedule"`
// 	RiskFactors []string `json:"risk_factors" jsonschema:"required,description=Potential implementation risks"`
// }

// type Sequence struct {
// 	ID           string   `json:"id" jsonschema:"required,description=Unique identifier for the sequence"`
// 	Description  string   `json:"description" jsonschema:"required,description=Sequence description"`
// 	OrderedSteps []string `json:"ordered_steps" jsonschema:"required,description=Steps in correct order"`
// 	Duration     string   `json:"duration" jsonschema:"required,description=Expected duration of sequence"`
// 	Dependencies []string `json:"dependencies" jsonschema:"required,description=What this sequence depends on"`
// }

// type Connection struct {
// 	ID          string  `json:"id" jsonschema:"required,description=Unique identifier for the connection"`
// 	FromID      string  `json:"from_id" jsonschema:"required,description=Source element ID"`
// 	ToID        string  `json:"to_id" jsonschema:"required,description=Target element ID"`
// 	Type        string  `json:"type" jsonschema:"required,description=Type of connection"`
// 	Description string  `json:"description" jsonschema:"required,description=Nature of the connection"`
// 	Strength    float64 `json:"strength" jsonschema:"required,description=Connection strength"`
// }

// type Relationship struct {
// 	ID            string   `json:"id" jsonschema:"required,description=Unique identifier for the relationship"`
// 	Type          string   `json:"type" jsonschema:"required,description=Type of relationship"`
// 	Description   string   `json:"description" jsonschema:"required,description=Nature of the relationship"`
// 	Elements      []string `json:"elements" jsonschema:"required,description=Related elements"`
// 	Bidirectional bool     `json:"bidirectional" jsonschema:"required,description=Whether relationship works both ways"`
// }

// type AnalogyMap struct {
// 	ID              string   `json:"id" jsonschema:"required,description=Unique identifier for the analogy map"`
// 	AbstractConcept string   `json:"abstract_concept" jsonschema:"required,description=The complex concept being mapped"`
// 	FamiliarConcept string   `json:"familiar_concept" jsonschema:"required,description=The relatable concept"`
// 	Description     string   `json:"description" jsonschema:"required,description=How the mapping works"`
// 	Limitations     []string `json:"limitations" jsonschema:"required,description=Where the analogy breaks down"`
// }

// type ThemeMapping struct {
// 	ID               string   `json:"id" jsonschema:"required,description=Unique identifier for the theme mapping"`
// 	Theme            string   `json:"theme" jsonschema:"required,description=The narrative theme"`
// 	AbstractConcepts []string `json:"abstract_concepts" jsonschema:"required,description=Related abstract concepts"`
// 	Description      string   `json:"description" jsonschema:"required,description=How concepts map to theme"`
// 	Examples         []string `json:"examples" jsonschema:"required,description=Concrete examples of the theme"`
// }

// type Dependency struct {
// 	ID          string `json:"id" jsonschema:"required,description=Unique identifier for the dependency"`
// 	SourceID    string `json:"source_id" jsonschema:"required,description=ID of the dependent element"`
// 	TargetID    string `json:"target_id" jsonschema:"required,description=ID of the element depended upon"`
// 	Type        string `json:"type" jsonschema:"required,description=Type of dependency"`
// 	Description string `json:"description" jsonschema:"required,description=Nature of the dependency"`
// 	Critical    bool   `json:"critical" jsonschema:"required,description=Whether this is a blocking dependency"`
// }
