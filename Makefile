.PHONY: all

capnp:
	capnp compile -I ../../capnproto/go-capnp/std -ogo pkg/provider/provider.capnp
	capnp compile -I ../../capnproto/go-capnp/std -ogo pkg/tools/tool.capnp
	capnp compile -I ../../capnproto/go-capnp/std -ogo pkg/ai/agent.capnp
