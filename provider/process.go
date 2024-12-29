package provider

type Process interface {
	Name() string
	Description() string
	GenerateSchema() interface{}
}

type CompoundProcess struct {
	Processes []Process
	Iteration int
}

func NewCompoundProcess(processes []Process) *CompoundProcess {
	return &CompoundProcess{
		Processes: processes,
		Iteration: 0,
	}
}

func (cp *CompoundProcess) Name() string {
	return cp.Processes[cp.Iteration%len(cp.Processes)].Name()
}

func (cp *CompoundProcess) Description() string {
	return cp.Processes[cp.Iteration%len(cp.Processes)].Description()
}

func (cp *CompoundProcess) GenerateSchema() interface{} {
	out := cp.Processes[cp.Iteration%len(cp.Processes)].GenerateSchema()
	cp.Iteration++
	return out
}
