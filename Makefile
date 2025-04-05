.PHONY: all

capnp:
	capnp compile -I ../../capnproto/go-capnp/std -ogo pkg/radix/radix.capnp
	capnp compile -I ../../capnproto/go-capnp/std -ogo pkg/ai/prompt/prompt.capnp
	capnp compile -I ../../capnproto/go-capnp/std -ogo pkg/ai/toolcall/toolcall.capnp
	capnp compile -I ../../capnproto/go-capnp/std -ogo pkg/ai/message/message.capnp
	capnp compile -I ../../capnproto/go-capnp/std -ogo pkg/ai/params/params.capnp
	capnp compile -I ../../capnproto/go-capnp/std -ogo pkg/ai/context/context.capnp
	capnp compile -I ../../capnproto/go-capnp/std -ogo pkg/ai/provider/provider.capnp
	capnp compile -I ../../capnproto/go-capnp/std -ogo pkg/ai/tool/tool.capnp
	capnp compile -I ../../capnproto/go-capnp/std -ogo pkg/ai/agent/agent.capnp
