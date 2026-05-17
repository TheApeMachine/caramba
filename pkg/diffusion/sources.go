package diffusion

import (
	"strings"

	modelweights "github.com/theapemachine/caramba/pkg/model/weights"
	"github.com/theapemachine/caramba/pkg/tokenizer"
)

/*
tokenizerSource translates a diffusion.Source into the tokenizer
package's Source shape, applying the diffusion package's repo_type
default when the manifest does not declare one.
*/
func tokenizerSource(source Source) tokenizer.Source {
	if strings.TrimSpace(source.Source) == "" {
		return tokenizer.Source{}
	}

	return tokenizer.Source{
		Source:   source.Source,
		File:     source.File,
		Cache:    source.Cache,
		Revision: source.Revision,
		RepoType: source.RepoType,
	}
}

/*
weightSource translates a diffusion.Source into the modelweights
package's Source shape.
*/
func weightSource(source Source) modelweights.Source {
	return modelweights.Source{
		Source:   source.Source,
		File:     source.File,
		Cache:    source.Cache,
		Revision: source.Revision,
		RepoType: source.RepoType,
	}
}
