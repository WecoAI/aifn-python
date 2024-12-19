<figure markdown="span">
    ![Image title](assets/ai_function_light.png#only-dark)
    ![Image title](assets/ai_function_dark.png#only-light)
</figure>

#

![Python](https://img.shields.io/badge/Python-3.10.14-blue)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
<a href="https://colab.research.google.com/github/WecoAI/aifn-python/blob/main/examples/cookbook.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" width=110 height=20/></a>
<a target="_blank" href="https://lightning.ai/new?repo_url=https%3A%2F%2Fgithub.com%2FWecoAI%2Faifn-python%2Fblob%2Fmain%2Fexamples%2Fcookbook.ipynb"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open in Studio" width=100 height=20/></a>

A client facing API for interacting with the [Weco AI](https://www.weco.ai/)'s AI function [platform](https://www.aifunction.com). It empowers you to go from zero to AI in just a few seconds!

Use this API to build *complex* AI features *fast*. We lower the barrier of entry to AI features by providing an interface to prototype solutions quickly, in just a few lines of code and in natural language.

## What We Offer

- Structured Output (outputs are Python dictionaries that **always** follow your AI functions JSON schema)
- Multimodal (language & vision)
- Grounding (Access to the web)
- Interpretable (observe reasoning behind outputs)
- Batched Inputs (have inputs be processed in concurrently)
- Sync-Async Duality (functions can be both synchronous & asynchronous)

## Getting Started

Install the `aifn` package:
```bash
pip install aifn
```

When using the Weco API, you will need to set the API key: You can find/create your API key [here](https://www.aifunction.com/account/api-keys). Once you have your API key, you can pass it directly to our core functions and classes using the `api_key` argument or set it as an environment variable as shown:
```bash
export WECO_API_KEY=<YOUR_WECO_API_KEY>
```

### Example

We created a function on our [platform](https://www.aifunction.com) for the following task:
> "Analyze a business idea and provide a well reasoned evaluation. Return 'viability_score' (0-100), 'strengths' (list), 'weaknesses' (list), and 'next_steps' (list)."

Here's how you can use this function anywhere in your code!
```python
from aifn import AIFunction
idea_evaluator = AIFunction("BusinessIdeaAnalyzer-XYZ123") # Replace with your actual function name
response = idea_evaluator("A subscription service for personalized, AI-generated bedtime stories for children.").output
```

To learn how to get the most your of **your** AI functions, check out our [cookbook](cookbook/cookbook.md), our [API reference](api/api.md) and these end-to-end [examples](cookbook/examples/maze_runner.md).

Happy building!
