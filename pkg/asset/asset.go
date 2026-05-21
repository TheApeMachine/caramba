package asset

import manifestoasset "github.com/theapemachine/manifesto/asset"

type (
	OperationPort = manifestoasset.OperationPort
	ConfigParam   = manifestoasset.ConfigParam
	TopologyNode  = manifestoasset.TopologyNode
	Topology      = manifestoasset.Topology
	System        = manifestoasset.System
	Schema        = manifestoasset.Schema
)

var (
	ReadFile   = manifestoasset.ReadFile
	TemplateFS = manifestoasset.TemplateFS
	Walk       = manifestoasset.Walk
)
