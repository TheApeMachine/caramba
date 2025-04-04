.PHONY: all

capnp:
	capnp compile -I ../../capnproto/go-capnp/std -ogo pkg/ai/params/params.capnp
	capnp compile -I ../../capnproto/go-capnp/std -ogo pkg/ai/context/context.capnp
	capnp compile -I ../../capnproto/go-capnp/std -ogo pkg/ai/provider/provider.capnp
	capnp compile -I ../../capnproto/go-capnp/std -ogo pkg/tools/tool.capnp
	capnp compile -I ../../capnproto/go-capnp/std -ogo pkg/ai/agent/agent.capnp
