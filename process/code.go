package process

import "github.com/theapemachine/amsh/utils"

type Code struct {
	Language string `json:"language" jsonschema:"Title=language,Description=The language of the code block,required"`
	Files    []File `json:"files" jsonschema:"Title=files,Description=List of files,required"`
}

type File struct {
	Path    string   `json:"path" jsonschema:"Title=path,Description=Path to the file,required"`
	Changes []Change `json:"changes" jsonschema:"Title=changes,Description=Changes to the file,required"`
}

type Change struct {
	Start int    `json:"start" jsonschema:"Title=start,Description=Start position of the change,required"`
	End   int    `json:"end" jsonschema:"Title=end,Description=End position of the change,required"`
	Code  string `json:"code" jsonschema:"Title=code,Description=Code to be changed,required"`
}

/*
SystemPrompt returns the system prompt for the Code process.
*/
func (code *Code) SystemPrompt(key string) string {
	return utils.SystemPrompt(key, "code", utils.GenerateSchema[Code]())
}
