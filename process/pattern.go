package process

// import (
// 	"time"

// 	"github.com/theapemachine/amsh/utils"
// )

// type GlobalPattern struct {
// 	ID           string     `json:"id" jsonschema:"required,title=Id,description=Unique identifier for the global pattern"`
// 	Layers       []string   `json:"layers" jsonschema:"required,title=Layers,description=Layers containing the global pattern"`
// 	Pattern      Pattern    `json:"pattern" jsonschema:"required,title=Pattern,description=Pattern of the global pattern"`
// 	Significance float64    `json:"significance" jsonschema:"required,title=Significance,description=Significance of the global pattern"`
// 	Support      []Evidence `json:"support" jsonschema:"required,title=Support,description=Support for the global pattern"`
// }

// type Scale struct {
// 	Level      int       `json:"level" jsonschema:"required,title=Level,description=Level of the scale"`
// 	Resolution float64   `json:"resolution" jsonschema:"required,title=Resolution,description=Resolution of the scale"`
// 	Patterns   []Pattern `json:"patterns" jsonschema:"required,title=Patterns,description=Patterns at this scale"`
// 	Metrics    Metrics   `json:"metrics" jsonschema:"required,title=Metrics,description=Metrics for the scale"`
// }

// /*
// EmergentPatterns represents higher-order patterns that emerge from interactions.
// */
// type EmergentPatterns struct {
// 	Patterns         []Pattern         `json:"patterns" jsonschema:"title=Patterns,description=Discovered higher-order patterns,required"`
// 	EmergenceRules   []EmergenceRule   `json:"emergence_rules" jsonschema:"title=EmergenceRules,description=Rules governing pattern formation,required"`
// 	StabilityMetrics []StabilityMetric `json:"stability_metrics" jsonschema:"title=StabilityMetrics,description=Measures of pattern stability,required"`
// }

// type EmergenceRule struct {
// 	ID           string      `json:"id" jsonschema:"required,title=Id,description=Unique identifier for the emergence rule"`
// 	Components   []Pattern   `json:"components" jsonschema:"required,title=Components,description=Components of the emergence rule"`
// 	Interactions []Relation  `json:"interactions" jsonschema:"required,title=Interactions,description=Interactions between components"`
// 	Outcome      Pattern     `json:"outcome" jsonschema:"required,title=Outcome,description=Outcome of the emergence rule"`
// 	Conditions   []Predicate `json:"conditions" jsonschema:"required,title=Conditions,description=Conditions for the emergence rule"`
// }

// type StabilityMetric struct {
// 	Type      string        `json:"type" jsonschema:"required,title=Type,description=Type of the stability metric"`
// 	Value     float64       `json:"value" jsonschema:"required,title=Value,description=Value of the stability metric"`
// 	Threshold float64       `json:"threshold" jsonschema:"required,title=Threshold,description=Threshold for the stability metric"`
// 	Window    time.Duration `json:"window" jsonschema:"required,title=Window,description=Window for the stability metric"`
// }

// /*
// UnifiedPerspective represents a coherent view across all structures.
// */
// type UnifiedPerspective struct {
// 	GlobalPatterns []GlobalPattern  `json:"global_patterns" jsonschema:"required,title=GlobalPatterns,description=Patterns visible across all layers"`
// 	Coherence      float64          `json:"coherence" jsonschema:"required,title=Coherence,description=Measure of overall integration"`
// 	Insights       []UnifiedInsight `json:"insights" jsonschema:"required,title=Insights,description=Understanding derived from the whole"`
// }

// type PatternAnalysis struct {
// 	FractalStructure FractalStructure `json:"fractal_structure" jsonschema:"title=FractalStructure,description=Self-similar patterns at different scales,required"`
// 	EmergentPatterns EmergentPatterns `json:"emergent_patterns" jsonschema:"title=EmergentPatterns,description=Higher-order patterns that emerge from interactions,required"`
// }

// func (pa *PatternAnalysis) SystemPrompt(key string) string {
// 	return utils.SystemPrompt(key, "pattern", utils.GenerateSchema[PatternAnalysis]())
// }
