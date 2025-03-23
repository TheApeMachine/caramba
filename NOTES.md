# NOTES

Keep any important information, goals, insights, etc. by writing them down in this file.

It will be attached to each request so you will have access to it.

## Code Examples of Issues

### 2. Command Validation in Runner

```
19:28:51 ERRO <environment/runner.go:64> unexpected end of JSON input
```

Root cause in `pkg/tools/environment/runner.go`:

```go
command := datura.GetMetaValue[string](artifact, "command")
if command == "" {
    return errnie.Error(errors.New("no command"))
}
// Command executed without validation
runner.bufIn.Write([]byte(command))
```

The error "unexpected end of JSON input" is actually coming from elsewhere in the stack (likely in the datura package when processing the artifact), not from the command itself. The command is meant to be a valid bash command, not JSON.

The command validation here should focus on bash command safety and validity:

```go
command := datura.GetMetaValue[string](artifact, "command")
if command == "" {
    return errnie.Error(errors.New("no command"))
}
// Could add bash command validation here if needed
// For example: check for dangerous commands, ensure proper syntax, etc.
runner.bufIn.Write([]byte(command))
```
