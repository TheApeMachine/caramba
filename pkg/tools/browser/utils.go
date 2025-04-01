package browser

import (
	htmltomarkdown "github.com/JohannesKaufmann/html-to-markdown/v2"
	"github.com/theapemachine/caramba/pkg/errnie"
)

// ConvertToMarkdown converts HTML content to markdown format
func ConvertToMarkdown(htmlContent string) (string, error) {
	markdown, err := htmltomarkdown.ConvertString(htmlContent)
	if err != nil {
		return "", errnie.Error(err)
	}
	return markdown, nil
}
