package tensor

import "github.com/theapemachine/amsh/utils"

/*
Process represents a tensor-based thought or reasoning process.
*/
type Process struct {
	Dimensions   []Dimension   `json:"dimensions" jsonschema:"title=Dimensions,description=Different aspects of relationship space,required"`
	TensorFields []TensorField `json:"tensor_fields" jsonschema:"title=TensorFields,description=Multi-dimensional relationship patterns,required"`
	Projections  []Projection  `json:"projections" jsonschema:"title=Projections,description=Lower-dimensional views of the tensor space,required"`
}

func NewProcess() *Process {
	return &Process{}
}

type Dimension struct {
	ID         string    `json:"id" jsonschema:"required,title=ID,description=Unique identifier for the dimension"`
	Name       string    `json:"name" jsonschema:"required,title=Name,description=Name of the dimension"`
	Scale      []float64 `json:"scale" jsonschema:"required,title=Scale,description=Scale values for the dimension"`
	Resolution float64   `json:"resolution" jsonschema:"required,title=Resolution,description=Granularity of the dimension"`
	Boundaries []float64 `json:"boundaries" jsonschema:"required,title=Boundaries,description=Min and max values"`
}

type TensorField struct {
	ID           string                 `json:"id" jsonschema:"required,title=ID,description=Unique identifier for the tensor field"`
	DimensionIDs []string               `json:"dimension_ids" jsonschema:"required,title=DimensionIDs,description=IDs of dimensions"`
	Values       []float64              `json:"values" jsonschema:"required,title=Values,description=Flattened tensor values"`
	Shape        []int                  `json:"shape" jsonschema:"required,title=Shape,description=Shape of the tensor"`
	Metadata     map[string]interface{} `json:"metadata" jsonschema:"title=Metadata,description=Additional metadata"`
}

type Projection struct {
	ID             string    `json:"id" jsonschema:"required,title=ID,description=Unique identifier for the projection"`
	SourceDimIDs   []string  `json:"source_dimension_ids" jsonschema:"required,title=SourceDimensionIDs,description=IDs of source dimensions"`
	TargetDimIDs   []string  `json:"target_dimension_ids" jsonschema:"required,title=TargetDimensionIDs,description=IDs of target dimensions"`
	ProjectionType string    `json:"projection_type" jsonschema:"required,title=ProjectionType,description=Type of projection"`
	Matrix         []float64 `json:"matrix" jsonschema:"required,title=Matrix,description=Projection matrix"`
}

func (ta *Process) GenerateSchema() string {
	return utils.GenerateSchema[Process]()
}
