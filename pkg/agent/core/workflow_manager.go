package core

type WorkflowManager struct {
	workflow *Workflow
}

func NewWorkflowManager() *WorkflowManager {
	return &WorkflowManager{}
}

func (wm *WorkflowManager) SetWorkflow(workflow *Workflow) {
	wm.workflow = workflow
}
