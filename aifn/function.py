from typing import Optional, Union, Callable, Coroutine, Any, List, NamedTuple, Dict
from functools import partial
from copy import deepcopy
from .client import Client
import warnings


def build(task_description: str, is_async: bool = False, api_key: Optional[str] = None) -> "AIFunction":
    """
    Build a specialized AI function for a given task.

    Parameters
    ----------
    task_description : str
        The description of the task for which the function is being built.

    is_async : bool, optional
        Indicates whether the function should be asynchronous. Defaults to False.

    api_key : str, optional
        The API key for the WecoAI service. If not provided, the API key must be set using the environment variable - `WECO_API_KEY`.

    Returns
    -------
    AIFunction
        A specialized AI function for the given task.

    Examples
    --------
    >>> from aifn import build
    >>> country_to_capital = build("Given a country name, return its capital city as 'capital'.")
    >>> country_to_capital("India").output["capital"]
    'New Delhi'
    """
    client = Client(api_key=api_key)
    fn_name, fn_version, fn_desc = client.build(task_description=task_description)
    return AIFunction(fn_name=fn_name, version=fn_version, fn_desc=fn_desc, is_async=is_async, api_key=api_key)


class AIFunction:
    """
    An AI powered function that can be called like any other function to perform a specific task.
    Driven by foundation models at its core, it can be used to perform a wide range of tasks such as text generation, summarization, translation, visual reasoning, etc.
    It can be used in both synchronous and asynchronous modes to suit the needs of the user.
    It also supports multimodal inputs and batch processing for handling multiple inputs at once.
    To take advantage of the full capabilities of the AI function, it is recommended to use the Weco AI platform to unlock features such as model selection, grounding through web search and more.
    """

    def __init__(
        self,
        fn_name: str,
        version: Optional[Union[str, int]] = -1,
        fn_desc: Optional[str] = "",
        is_async: Optional[bool] = False,
        api_key: Optional[str] = None,
    ) -> "AIFunction":
        """
        Retrieve an AI function with its unique name and version.

        Parameters
        ----------
        fn_name : str
            The name of the AI function.

        version : Union[str, int], optional
            The version of the AI function. Defaults to -1, indicating the latest version.

        fn_desc : str, optional
            A description of the AI function. Defaults to an empty string.

        is_async : bool, optional
            Indicates whether the function should be asynchronous. Defaults to False.

        api_key : str, optional
            The API key for accessing the client. If not provided, it must be set using the environment variable - `WECO_API_KEY`.

        Examples
        --------
        >>> from aifn import AIFunction
        >>> idea_evaluator = AIFunction("BusinessIdeaAnalyzer-XYZ123")
        """
        self.fn_name = fn_name
        self.version = version
        self.fn_desc = fn_desc
        self.is_async = is_async
        self._client = Client(api_key=api_key)
        self._fn = self._create_fn(is_batch=False)
        self._batch_fn = self._create_fn(is_batch=True)

    def __call__(
        self,
        text_input: Optional[str] = "",
        images_input: Optional[List[str]] = [],
        return_reasoning: Optional[bool] = False,
        strict: Optional[bool] = False,
    ) -> NamedTuple:
        """
        Call the AI function with the provided inputs.

        Parameters
        ----------
        text_input : str, optional
            The text input to the function. Defaults to an empty string.

        images_input : List[str], optional
            A list of image URLs or base64 encoded images to be used as input to the function. Defaults to an empty list.

        return_reasoning : bool, optional
            If True, includes reasoning in the response. Defaults to False.

        strict : bool, optional
            A flag to indicate if the function should be queried in strict mode. Defaults to False.

        Returns
        -------
        NamedTuple
            A NamedTuple containing the output and metadata of the response.

        Examples
        --------

        **Build and call a function**

        >>> from aifn import build
        >>> country_to_capital = build("Given a country name, return its capital city as 'capital'.")
        >>> response = country_to_capital("France")
        >>> response.output["capital"]
        'Paris'

        **Retrieve and call an existing function**

        >>> from aifn import AIFunction
        >>> idea_evaluator = AIFunction("BusinessIdeaAnalyzer-XYZ123")
        >>> response = idea_evaluator("A platform to connect pet owners with pet sitters.")
        >>> response.output["score"]
        0.85

        **Call an existing function with image inputs**

        >>> from aifn import AIFunction
        >>> image_classifier = AIFunction("ImageClassifier-ABC123")
        >>> response = image_classifier(images_input=["https://example.com/cat.jpg"])
        >>> response.output["label"]
        'cat'
        """
        return self._fn(
            fn_name=self.fn_name,
            version=self.version,
            text_input=text_input,
            images_input=images_input,
            return_reasoning=return_reasoning,
            strict=strict,
        )

    def batch(
        self,
        batch_inputs: Optional[List[Dict[str, Any]]] = [],
        return_reasoning: Optional[bool] = False,
        strict: Optional[bool] = False,
    ) -> List[NamedTuple]:
        """
        Call the AI function in batch mode with the provided inputs.

        Parameters
        ----------
        batch_inputs : List[Dict[str, Any]], optional
            A list of input dictionaries for the function. Each dictionary can include:
            - "text_input": A string for text input.
            - "images_input": A list of image URLs or base64 encoded images.

        return_reasoning : bool, optional
            If True, includes reasoning in the response. Defaults to False.

        strict : bool, optional
            A flag to indicate if the function should be queried in strict mode. Defaults to False.

        Returns
        -------
        List[NamedTuple]
            A list of NamedTuples, each containing the output and metadata of the response.

        Examples
        -------
        >>> from aifn import build
        >>> country_to_capital = build("Given a country name, return its capital city as 'capital'.")
        >>> batch_inputs = [{"text_input": "India"}, {"text_input": "USA"}, {"text_input": "UK"}]
        >>> responses = country_to_capital.batch(batch_inputs)
        >>> outputs = [response.output["capital"] for response in responses]
        >>> outputs
        ['New Delhi', 'Washington, D.C.', 'London']
        """
        return self._batch_fn(
            fn_name=self.fn_name,
            version=self.version,
            batch_inputs=batch_inputs,
            return_reasoning=return_reasoning,
            strict=strict,
        )

    def __repr__(self) -> str:
        version = f"{self.version}" if isinstance(self.version, int) and self.version > -1 else f"({self.version})"
        return f"{self.fn_name}/v{version}"

    def __str__(self) -> str:
        return self.__repr__()

    def _create_fn(self, is_batch: bool) -> Union[Callable[..., Any], Coroutine[Any, Any, Any]]:
        """
        Create a function based on the specified attributes.

        Parameters
        ----------
        is_batch : bool
            Indicates whether the function should be created in batch mode.

        Returns
        -------
        Union[Callable[..., Any], Coroutine[Any, Any, Any]]
            A callable or coroutine function with the specified attributes.
        """
        if not self.is_async and not is_batch:
            core_fn = self._client.query
        elif self.is_async and not is_batch:
            core_fn = self._client.aquery
        elif not self.is_async and is_batch:
            core_fn = self._client.batch_query
        elif self.is_async and is_batch:
            core_fn = self._client.abatch_query

        annotations = deepcopy(core_fn.__annotations__)
        annotations.pop("fn_name", None)
        annotations.pop("version", None)

        partial_fn = partial(core_fn, fn_name=self.fn_name, version=self.version)

        if self.is_async:

            async def async_wrapper(*args, **kwargs):
                return await partial_fn(*args, **kwargs)

            async_wrapper.__doc__ = self.fn_desc
            async_wrapper.__annotations__ = annotations
            return async_wrapper
        else:
            partial_fn.__doc__ = self.fn_desc
            partial_fn.__annotations__ = annotations
            return partial_fn

    def make_sync(self) -> "AIFunction":
        """
        Convert an asynchronous AI function to a synchronous AI Function.

        Returns
        -------
        AIFunction
            A new AIFunction instance in synchronous mode.

        Examples
        -------
        >>> from aifn import build
        >>> country_to_capital = build("Given a country name, return its capital city as 'capital'.", is_async=True)
        >>> sync_country_to_capital = country_to_capital.make_sync()
        >>> response = sync_country_to_capital("USA")
        >>> response.output["capital"]
        'Washington, D.C.'
        """
        if not self.is_async:
            warnings.warn(f"{self} is already a synchronous AI Function...Returning the same object.")
            return self
        return AIFunction(
            fn_name=self.fn_name, version=self.version, fn_desc=self.fn_desc, is_async=False, api_key=self._client.api_key
        )

    def make_async(self) -> "AIFunction":
        """
        Convert a synchronous AI function to an asynchronous AI Function.

        Returns
        -------
        AIFunction
            A new AIFunction instance in asynchronous mode.

        Examples
        -------
        >>> from aifn import build
        >>> country_to_capital = build("Given a country name, return its capital city as 'capital'.", is_async=False)
        >>> async_country_to_capital = country_to_capital.make_async()
        >>> response = await async_country_to_capital("USA")
        >>> response.output["capital"]
        'Washington, D.C.'
        """
        if self.is_async:
            warnings.warn(f"{self} is already an asynchronous AI Function...Returning the same object.")
            return self
        return AIFunction(
            fn_name=self.fn_name, version=self.version, fn_desc=self.fn_desc, is_async=True, api_key=self._client.api_key
        )
