package tasks

import (
	"fmt"

	"github.com/charmbracelet/log"
	"github.com/theapemachine/caramba/ai/drknow"
	"github.com/theapemachine/caramba/provider"
	"github.com/theapemachine/caramba/tools"
)

type Web struct {
	qdrant *tools.Qdrant
	neo4j  *tools.Neo4j
}

func NewWeb() *Web {
	return &Web{
		qdrant: tools.NewQdrant("web_content", 1536),
		neo4j:  tools.NewNeo4j(),
	}
}

func (task *Web) Execute(ctx *drknow.Context, args map[string]any) Bridge {
	log.Info("Starting Web task", "args", args)

	// Initialize memory stores
	if err := task.qdrant.Initialize(); err != nil {
		log.Error("Failed to initialize Qdrant", "error", err)
	}

	// First check if we have this URL in memory
	url, ok := args["url"].(string)
	if !ok || url == "" {
		errMsg := "URL is required"
		log.Error(errMsg)
		ctx.AddMessage(provider.NewMessage(provider.RoleAssistant, errMsg))
		return nil
	}

	// Try to recall from memory first
	memoryArgs := map[string]any{
		"query": url,
	}
	if result := task.qdrant.Use(ctx.Identity.Ctx, memoryArgs); result != "No results found" {
		log.Info("Found content in memory", "url", url)
		ctx.AddMessage(provider.NewMessage(provider.RoleAssistant, result))
		return nil
	}

	browser := tools.NewBrowser()
	defer browser.Close()

	// First handle scroll if requested
	if action, ok := args["action"].(string); ok && action == "scroll" {
		scrollArgs := map[string]any{
			"url":    url,
			"action": "scroll",
		}
		if _, err := browser.Run(scrollArgs); err != nil {
			errMsg := fmt.Sprintf("Scroll error: %v", err)
			log.Error(errMsg)
			ctx.AddMessage(provider.NewMessage(provider.RoleAssistant, errMsg))
			return nil
		}
	}

	// Extract content using provided JavaScript
	if js, ok := args["javascript"].(string); ok && js != "" {
		contentArgs := map[string]any{
			"url":        url,
			"javascript": js,
		}
		result, err := browser.Run(contentArgs)
		if err != nil {
			errMsg := fmt.Sprintf("Browser error: %v", err)
			log.Error(errMsg)
			ctx.AddMessage(provider.NewMessage(provider.RoleAssistant, errMsg))
			return nil
		}

		if result == "" {
			log.Warn("No content extracted from page")
			ctx.AddMessage(provider.NewMessage(provider.RoleAssistant, "No content could be extracted from the page"))
			return nil
		}

		// Store the result in memory
		memoryArgs := map[string]any{
			"documents": []string{result},
			"metadata": map[string]any{
				"url":      url,
				"type":     "web_content",
				"js_query": js,
			},
		}
		task.qdrant.Use(ctx.Identity.Ctx, memoryArgs)

		// Store in graph database for relationships
		cypherQuery := `
			MERGE (p:Page {url: $url})
			CREATE (c:Content {
				text: $content,
				timestamp: datetime()
			})
			CREATE (p)-[:HAS_CONTENT]->(c)
		`
		neo4jArgs := map[string]any{
			"cypher": cypherQuery,
			"params": map[string]any{
				"url":     url,
				"content": result,
			},
		}
		task.neo4j.Use(ctx.Identity.Ctx, neo4jArgs)

		// Format the result nicely
		formattedResult := fmt.Sprintf("Content from %s:\n%s", url, result)
		ctx.AddMessage(provider.NewMessage(provider.RoleAssistant, formattedResult))
	} else {
		errMsg := "JavaScript selector is required for content extraction"
		log.Error(errMsg)
		ctx.AddMessage(provider.NewMessage(provider.RoleAssistant, errMsg))
	}

	return nil
}
