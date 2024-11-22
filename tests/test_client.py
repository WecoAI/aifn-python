import pytest
from aifn.client import Client


client: Client = Client(timeout=500)


# Fixtures
# Synchronous
@pytest.fixture
def sync_text_evaluator():
    function_name, version, description = client.build(task_description="Evaluate the 'sentiment' of the given text.")
    return function_name, version, description


@pytest.fixture
def sync_image_evaluator():
    function_name, version, description = client.build(
        task_description="Describe the contents of the given images. Provide the 'description' and 'objects'."
    )
    return function_name, version, description


@pytest.fixture
def sync_text_and_image_evaluator():
    function_name, version, description = client.build(
        task_description="Evaluate, solve and arrive at a numerical 'answer' for the image provided. Perform any additional things if instructed."
    )
    return function_name, version, description


# Asynchronous
@pytest.fixture
async def async_text_evaluator():
    fn_name, version_number, fn_desc = await client.abuild(task_description="Evaluate the 'sentiment' of the given text.")
    return fn_name, version_number, fn_desc


@pytest.fixture
async def async_image_evaluator():
    fn_name, version_number, fn_desc = await client.abuild(
        task_description="Describe the contents of the given images. Provide the 'description' and 'objects'."
    )
    return fn_name, version_number, fn_desc


@pytest.fixture
async def async_text_and_image_evaluator():
    fn_name, version_number, fn_desc = await client.abuild(
        task_description="Evaluate, solve and arrive at a numerical 'answer' for the image provided. Perform any additional things if instructed."
    )
    return fn_name, version_number, fn_desc


# Batch
@pytest.fixture
def batch_text_evaluator():
    fn_name, version_number, _ = client.build(
        task_description="I want to evaluate the feasibility of a machine learning task. Provide the 'feasibility', 'justification', and 'suggestions'."
    )
    return fn_name, version_number


@pytest.fixture
def batch_image_evaluator():
    fn_name, version_number, _ = client.build(
        task_description="Describe the contents of the given images. Provide the 'description' and 'objects'."
    )
    return fn_name, version_number


@pytest.fixture
def reasoning_text_evaluator():
    function_name, version, _ = client.build(
        task_description="Evaluate the 'sentiment' of the given text. Think deeply and provide your reasoning behind your answer."
    )
    return function_name, version


@pytest.fixture
def reasoning_text_and_image_evaluator():
    function_name, version, _ = client.build(
        task_description="Evaluate, solve and arrive at a numerical 'answer' for the image provided. Perform any additional things if instructed. Think deeply and provide your reasoning behind your answer."
    )
    return function_name, version


####################################
######### Build Tests ##############
####################################


def assert_build_response(function_name, version, description):
    assert isinstance(function_name, str)
    assert isinstance(version, int)
    assert isinstance(description, str)


# Synchronous
def test_build(sync_text_evaluator, sync_image_evaluator, sync_text_and_image_evaluator):
    evaluators = [sync_text_evaluator, sync_image_evaluator, sync_text_and_image_evaluator]
    for evaluator in evaluators:
        fn_name, version_number, fn_desc = evaluator
        assert_build_response(function_name=fn_name, version=version_number, description=fn_desc)


# Asynchronous
@pytest.mark.asyncio
async def test_abuild(async_text_evaluator, async_image_evaluator, async_text_and_image_evaluator):
    evaluators = [async_text_evaluator, async_image_evaluator, async_text_and_image_evaluator]
    for evaluator in evaluators:
        fn_name, version_number, fn_desc = await evaluator
        assert_build_response(function_name=fn_name, version=version_number, description=fn_desc)


####################################
######### Query Tests ##############
####################################


def assert_query_response(query_response, reasoning):
    assert query_response.__class__.__name__ == "QueryResponse"
    assert isinstance(query_response.output, dict)
    assert isinstance(query_response.metadata["in_tokens"], int)
    assert isinstance(query_response.metadata["out_tokens"], int)
    assert isinstance(query_response.metadata["latency_ms"], float)
    if reasoning:
        # reasoning included in response
        assert isinstance(query_response.metadata["reasoning_steps"], list)
        for step in query_response.metadata["reasoning_steps"]:
            assert isinstance(step, str)
    else:
        assert query_response.metadata["reasoning_steps"] is None


# Synchronous single queries
def test_text_query(sync_text_evaluator):
    function_name, version, _ = sync_text_evaluator
    query_response = client.query(fn_name=function_name, version=version, text_input="I love this product!", strict=True)
    assert_query_response(query_response, reasoning=False)
    assert set(query_response.output.keys()) == {"sentiment"}


def test_image_query(sync_image_evaluator):
    function_name, version, _ = sync_image_evaluator
    query_response = client.query(
        fn_name=function_name,
        version=version,
        images_input=[
            "https://www.integratedtreatmentservices.co.uk/wp-content/uploads/2013/12/Objects-of-Reference.jpg",
            "https://t4.ftcdn.net/jpg/05/70/90/23/360_F_570902339_kNj1reH40GFXakTy98EmfiZHci2xvUCS.jpg",
        ],
        strict=True,
    )
    assert_query_response(query_response, reasoning=False)
    assert set(query_response.output.keys()) == {"description", "objects"}


def test_text_and_image_query(sync_text_and_image_evaluator):
    function_name, version, _ = sync_text_and_image_evaluator
    query_response = client.query(
        fn_name=function_name,
        version=version,
        text_input="Find x and y.",
        images_input=[
            "https://i.ytimg.com/vi/cblHUeq3bkE/hq720.jpg?sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&rs=AOn4CLAKn3piY91QRCBzRgnzAPf7MPrjDQ"
        ],
        strict=True,
    )
    assert_query_response(query_response, reasoning=False)
    assert set(query_response.output.keys()) == {"answer"}


# Asynchronous single queries
@pytest.mark.asyncio
async def test_text_aquery(async_text_evaluator):
    fn_name, version_number, _ = await async_text_evaluator
    query_response = await client.aquery(
        fn_name=fn_name, version=version_number, text_input="I love this product!", strict=True
    )
    assert_query_response(query_response, reasoning=False)
    assert set(query_response.output.keys()) == {"sentiment"}


@pytest.mark.asyncio
async def test_image_aquery(async_image_evaluator):
    fn_name, version_number, _ = await async_image_evaluator
    query_response = await client.aquery(
        fn_name=fn_name,
        version=version_number,
        images_input=[
            "https://www.integratedtreatmentservices.co.uk/wp-content/uploads/2013/12/Objects-of-Reference.jpg",
            "https://t4.ftcdn.net/jpg/05/70/90/23/360_F_570902339_kNj1reH40GFXakTy98EmfiZHci2xvUCS.jpg",
        ],
        strict=True,
    )
    assert_query_response(query_response, reasoning=False)
    assert set(query_response.output.keys()) == {"description", "objects"}


@pytest.mark.asyncio
async def test_text_and_image_aquery(async_text_and_image_evaluator):
    fn_name, version_number, _ = await async_text_and_image_evaluator
    query_response = await client.aquery(
        fn_name=fn_name,
        version=version_number,
        text_input="Find x and y.",
        images_input=[
            "https://i.ytimg.com/vi/cblHUeq3bkE/hq720.jpg?sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&rs=AOn4CLAKn3piY91QRCBzRgnzAPf7MPrjDQ"
        ],
        strict=True,
    )
    assert_query_response(query_response, reasoning=False)
    assert set(query_response.output.keys()) == {"answer"}


# Synchronous batch queries
def test_batch_query_text(batch_text_evaluator):
    fn_name, version_number = batch_text_evaluator
    batch_inputs = [
        {"text_input": "I want to train a model to predict house prices using the Boston Housing dataset hosted on Kaggle."},
        {
            "text_input": "I want to train a model to classify digits using the MNIST dataset hosted on Kaggle using a Google Colab notebook."
        },
    ]
    query_responses = client.batch_query(fn_name=fn_name, version=version_number, batch_inputs=batch_inputs, strict=True)

    assert len(query_responses) == len(batch_inputs)

    for query_response in query_responses:
        assert_query_response(query_response, reasoning=False)
        assert set(query_response.output.keys()) == {"feasibility", "justification", "suggestions"}


def test_batch_query_image(batch_image_evaluator):
    fn_name, version_number = batch_image_evaluator
    batch_inputs = [
        {
            "images_input": [
                "https://www.integratedtreatmentservices.co.uk/wp-content/uploads/2013/12/Objects-of-Reference.jpg"
            ]
        },
        {"images_input": ["https://t4.ftcdn.net/jpg/05/70/90/23/360_F_570902339_kNj1reH40GFXakTy98EmfiZHci2xvUCS.jpg"]},
    ]

    query_responses = client.batch_query(fn_name=fn_name, version=version_number, batch_inputs=batch_inputs, strict=True)

    assert len(query_responses) == len(batch_inputs)

    for query_response in query_responses:
        assert_query_response(query_response, reasoning=False)
        assert set(query_response.output.keys()) == {"description", "objects"}


# Asynchronous batch queries
@pytest.mark.asyncio
async def test_abatch_query_text(batch_text_evaluator):
    fn_name, version_number = batch_text_evaluator
    batch_inputs = [
        {"text_input": "I want to train a model to predict house prices using the Boston Housing dataset hosted on Kaggle."},
        {
            "text_input": "I want to train a model to classify digits using the MNIST dataset hosted on Kaggle using a Google Colab notebook."
        },
    ]
    query_responses = await client.abatch_query(
        fn_name=fn_name, version=version_number, batch_inputs=batch_inputs, strict=True
    )

    assert len(query_responses) == len(batch_inputs)

    for query_response in query_responses:
        assert_query_response(query_response, reasoning=False)
        assert set(query_response.output.keys()) == {"feasibility", "justification", "suggestions"}


@pytest.mark.asyncio
async def test_abatch_query_image(batch_image_evaluator):
    fn_name, version_number = batch_image_evaluator
    batch_inputs = [
        {
            "images_input": [
                "https://www.integratedtreatmentservices.co.uk/wp-content/uploads/2013/12/Objects-of-Reference.jpg"
            ]
        },
        {"images_input": ["https://t4.ftcdn.net/jpg/05/70/90/23/360_F_570902339_kNj1reH40GFXakTy98EmfiZHci2xvUCS.jpg"]},
    ]

    query_responses = await client.abatch_query(
        fn_name=fn_name, version=version_number, batch_inputs=batch_inputs, strict=True
    )

    assert len(query_responses) == len(batch_inputs)

    for query_response in query_responses:
        assert_query_response(query_response, reasoning=False)
        assert set(query_response.output.keys()) == {"description", "objects"}


# Reasoning
def test_reasoning_text_query(reasoning_text_evaluator):
    function_name, version = reasoning_text_evaluator
    query_response = client.query(
        fn_name=function_name, version=version, text_input="I love this product!", return_reasoning=True, strict=True
    )
    assert_query_response(query_response, reasoning=True)
    assert "sentiment" in set(query_response.output.keys())


def test_reasoning_text_and_image_query(reasoning_text_and_image_evaluator):
    function_name, version = reasoning_text_and_image_evaluator
    query_response = client.query(
        fn_name=function_name,
        version=version,
        text_input="Find x and y.",
        images_input=[
            "https://i.ytimg.com/vi/cblHUeq3bkE/hq720.jpg?sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&rs=AOn4CLAKn3piY91QRCBzRgnzAPf7MPrjDQ"
        ],
        return_reasoning=True,
        strict=True,
    )
    assert_query_response(query_response, reasoning=True)
    assert "answer" in set(query_response.output.keys())
