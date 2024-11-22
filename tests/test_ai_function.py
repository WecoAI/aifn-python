import pytest
from aifn import build, AIFunction


def assert_ai_function_response(response, expected_keys, reasoning):
    assert isinstance(response.output, dict)
    assert isinstance(response.metadata, dict)
    assert isinstance(response.metadata["in_tokens"], int)
    assert isinstance(response.metadata["out_tokens"], int)
    assert isinstance(response.metadata["latency_ms"], float)
    if reasoning:
        assert isinstance(response.metadata["reasoning_steps"], list)
        for step in response.metadata["reasoning_steps"]:
            assert isinstance(step, str)
    else:
        assert response.metadata["reasoning_steps"] is None
    for key in expected_keys:
        assert key in response.output


def test_ai_function_sync():
    fn = build(task_description="Give me the french 'translation")
    assert isinstance(fn, AIFunction)
    assert isinstance(fn.fn_name, str)
    assert isinstance(fn.version, (int, str))
    assert isinstance(fn.fn_desc, str)
    assert isinstance(fn.is_async, bool)
    assert not fn.is_async

    response = fn(text_input="hello")
    assert_ai_function_response(response, {"translation"}, reasoning=False)

    responses = fn.batch(batch_inputs=[{"text_input": "hello"}, {"text_input": "world"}, {"text_input": "how are you?"}])
    for response in responses:
        assert_ai_function_response(response, {"translation"}, reasoning=False)


@pytest.mark.asyncio
async def test_ai_function_async():
    fn = build(task_description="translate to french in 'translation' key", is_async=True)
    assert isinstance(fn, AIFunction)
    assert isinstance(fn.fn_name, str)
    assert isinstance(fn.version, (int, str))
    assert isinstance(fn.fn_desc, str)
    assert isinstance(fn.is_async, bool)
    assert fn.is_async

    response = await fn(text_input="hello")
    assert_ai_function_response(response, {"translation"}, reasoning=False)

    responses = await fn.batch(batch_inputs=[{"text_input": "hello"}, {"text_input": "world"}, {"text_input": "how are you?"}])
    for response in responses:
        assert_ai_function_response(response, {"translation"}, reasoning=False)


def test_ai_function_conversion():
    # fn is async
    fn = build(task_description="translate to french in 'translation' key", is_async=True)
    assert fn.is_async is True

    # sync_fn is sync
    sync_fn = fn.make_sync()
    assert sync_fn.is_async is False

    # async_fn is async
    async_fn = sync_fn.make_async()
    assert async_fn.is_async is True
