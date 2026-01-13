import pytest

from caramba.core.event_codec.payloads import (
    decode_idle_payload,
    decode_impulse_payload,
    decode_memory_answer_payload,
    decode_memory_query_payload,
    decode_memory_write_payload,
    decode_message_payload,
    decode_noise_payload,
    decode_tool_definition_payload,
    decode_tool_registered_payload,
    decode_tool_rejected_payload,
    decode_toolchain_definition_payload,
    decode_toolchain_test_result_payload,
    encode_idle_payload,
    encode_impulse_payload,
    encode_memory_answer_payload,
    encode_memory_query_payload,
    encode_memory_write_payload,
    encode_message_payload,
    encode_noise_payload,
    encode_tool_definition_payload,
    encode_tool_registered_payload,
    encode_tool_rejected_payload,
    encode_toolchain_definition_payload,
    encode_toolchain_test_result_payload,
)


def test_payload_roundtrips() -> None:
    assert decode_message_payload(encode_message_payload(text="hi")) == "hi"

    ts, metrics = decode_idle_payload(encode_idle_payload(ts=1.0, metrics={"a": 1.5}))
    assert ts == 1.0 and metrics == {"a": 1.5}

    assert decode_memory_write_payload(encode_memory_write_payload(key=300, value=400)) == (300, 400)
    assert decode_memory_query_payload(encode_memory_query_payload(key=300)) == 300
    assert decode_memory_answer_payload(encode_memory_answer_payload(value=400)) == 400
    assert decode_noise_payload(encode_noise_payload(tok=12345)) == 12345

    td = encode_tool_definition_payload(name="n", description="d", implementation="i", requirements=["r"])
    assert decode_tool_definition_payload(td) == ("n", "d", "i", ["r"])

    tr = encode_tool_registered_payload(name="n", path="/tmp/x")
    assert decode_tool_registered_payload(tr) == ("n", "/tmp/x")

    tj = encode_tool_rejected_payload(error="boom")
    assert decode_tool_rejected_payload(tj) == "boom"

    impulse = encode_impulse_payload(
        metrics={"a": 1.0},
        signals=[
            {
                "name": "drive",
                "metric": "entropy",
                "value": 0.5,
                "band": {"min": 0.0, "max": 1.0},
                "deviation": 0.0,
                "urgency": 0.1,
            }
        ],
        max_urgency=0.1,
    )
    m2, s2, maxu2 = decode_impulse_payload(impulse)
    assert m2 == {"a": 1.0}
    assert isinstance(s2, list) and s2[0]["name"] == "drive"
    assert maxu2 == pytest.approx(0.1)

    tool_def = encode_toolchain_definition_payload(
        name="t",
        version="v1",
        description="d",
        entrypoint="tool:run",
        code="print('x')",
        tests="assert True",
        capabilities={"network": True},
        requirements=["requests==2.31.0"],
    )
    out = decode_toolchain_definition_payload(tool_def)
    assert out[0] == "t" and out[1] == "v1"

    res = encode_toolchain_test_result_payload(name="t", version="v1", ok=True, output="ok")
    assert decode_toolchain_test_result_payload(res) == ("t", "v1", True, "ok")

