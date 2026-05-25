package schema

import (
	"fmt"
	"strings"

	"github.com/gofiber/fiber/v3"
	"github.com/theapemachine/caramba/pkg/asset"
)

/*
Block serves block and model schemas to the frontend node graph editor.
Blocks are pre-wired groups of operations that appear as a single collapsed
node in the graph with exposed external ports only.
*/
type Block struct{}

/*
NewBlock creates a Block schema handler.
*/
func NewBlock() *Block {
	return &Block{}
}

/*
Request returns all block and model schemas as JSON, keyed by op identifier.
*/
func (block *Block) Request(ctx fiber.Ctx) error {
	schemas, err := asset.Walk("template/block")

	if err != nil {
		return ctx.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	modelSchemas, modelErr := asset.Walk("template/model")

	if modelErr != nil {
		return ctx.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": modelErr.Error(),
		})
	}

	for key, schema := range modelSchemas {
		schemas[key] = schema
	}

	for key, schema := range schemas {
		if schema.System == nil || len(schema.System.Topology.Nodes) == 0 {
			continue
		}

		schema.System.Topology.Nodes = expandTopology(schema.System.Topology.Nodes)
		schemas[key] = schema
	}

	return ctx.JSON(schemas)
}

func expandTopology(nodes []asset.TopologyNode) []asset.TopologyNode {
	expanded := make([]asset.TopologyNode, 0, len(nodes))

	for _, node := range nodes {
		repeatCount, hasRepeat := repeatCount(node.Repeat)

		if !hasRepeat {
			expanded = append(expanded, node)
			continue
		}

		for index := range repeatCount {
			for _, templateNode := range node.Template {
				expanded = append(expanded, asset.TopologyNode{
					ID:     replaceVars(templateNode.ID, node.Index, index),
					Op:     replaceVars(templateNode.Op, node.Index, index),
					In:     replaceVarsSlice(templateNode.In, node.Index, index),
					Out:    replaceVarsSlice(templateNode.Out, node.Index, index),
					Config: replaceVarsMap(templateNode.Config, node.Index, index),
				})
			}
		}
	}

	return expanded
}

func repeatCount(repeat any) (int, bool) {
	switch value := repeat.(type) {
	case int:
		return value, value > 0
	case int64:
		count := int(value)
		return count, count > 0
	case float64:
		count := int(value)
		return count, count > 0
	default:
		return 0, false
	}
}

func replaceVars(value string, indexVar string, index int) string {
	value = strings.ReplaceAll(value, fmt.Sprintf("${%s}", indexVar), fmt.Sprintf("%d", index))
	value = strings.ReplaceAll(value, fmt.Sprintf("${next_%s}", indexVar), fmt.Sprintf("%d", index+1))

	return value
}

func replaceVarsSlice(values []string, indexVar string, index int) []string {
	result := make([]string, 0, len(values))

	for _, value := range values {
		result = append(result, replaceVars(value, indexVar, index))
	}

	return result
}

func replaceVarsMap(values map[string]any, indexVar string, index int) map[string]any {
	if values == nil {
		return nil
	}

	result := make(map[string]any, len(values))

	for key, value := range values {
		stringValue, isString := value.(string)

		if isString {
			result[key] = replaceVars(stringValue, indexVar, index)
			continue
		}

		result[key] = value
	}

	return result
}
