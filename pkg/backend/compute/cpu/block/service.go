package block

import (
	"maps"

	"fmt"
	"strings"

	"github.com/gofiber/fiber/v3"
	"github.com/theapemachine/caramba/pkg/asset"
)

/*
Service serves block schemas to the frontend node graph editor.
Blocks are pre-wired groups of operations that appear as a single
collapsed node in the graph with exposed external ports only.
*/
type Service struct{}

/*
NewService creates a new Service.
*/
func NewService() *Service {
	return &Service{}
}

/*
Request returns all block schemas as JSON, keyed by op identifier.
*/
func (service *Service) Request(ctx fiber.Ctx) error {
	schemas, err := asset.Walk("template/block")

	if err != nil {
		return ctx.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	modelSchemas, err := asset.Walk("template/model")
	if err != nil {
		return ctx.Status(fiber.StatusInternalServerError).JSON(fiber.Map{
			"error": err.Error(),
		})
	}

	maps.Copy(schemas, modelSchemas)

	for key, schema := range schemas {
		if schema.System != nil {
			schema.System.Topology.Nodes = expandTopology(schema.System.Topology.Nodes)
			schemas[key] = schema
		}
	}

	return ctx.JSON(schemas)
}

func expandTopology(nodes []asset.TopologyNode) []asset.TopologyNode {
	var expanded []asset.TopologyNode
	for _, node := range nodes {
		if node.Repeat > 0 {
			for i := 0; i < node.Repeat; i++ {
				for _, tpl := range node.Template {
					expandedNode := asset.TopologyNode{
						ID:     replaceVars(tpl.ID, node.Index, i),
						Op:     replaceVars(tpl.Op, node.Index, i),
						In:     replaceVarsSlice(tpl.In, node.Index, i),
						Out:    replaceVarsSlice(tpl.Out, node.Index, i),
						Config: replaceVarsMap(tpl.Config, node.Index, i),
					}
					expanded = append(expanded, expandedNode)
				}
			}
		} else {
			expanded = append(expanded, node)
		}
	}
	return expanded
}

func replaceVars(s string, indexVar string, i int) string {
	s = strings.ReplaceAll(s, fmt.Sprintf("${%s}", indexVar), fmt.Sprintf("%d", i))
	s = strings.ReplaceAll(s, fmt.Sprintf("${next_%s}", indexVar), fmt.Sprintf("%d", i+1))
	return s
}

func replaceVarsSlice(s []string, indexVar string, i int) []string {
	var res []string
	for _, v := range s {
		res = append(res, replaceVars(v, indexVar, i))
	}
	return res
}

func replaceVarsMap(m map[string]any, indexVar string, i int) map[string]any {
	if m == nil {
		return nil
	}
	res := make(map[string]any)
	for k, v := range m {
		if str, ok := v.(string); ok {
			res[k] = replaceVars(str, indexVar, i)
		} else {
			res[k] = v
		}
	}
	return res
}
