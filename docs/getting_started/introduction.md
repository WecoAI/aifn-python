# Introduction

Lets jump right in!

## Export API Key

When using the Weco API, you will need to set the API key. You can find/setup your API key [here](https://www.aifunction.com/account/api-keys). Here's what it looks like.

![alt text](../assets/api_keys.png)

Once you have your API key, pass it directly to our `build` function or `AIFunction` class (don't worry, we'll cover these shortly) using the `api_key` argument or set it as an environment variable as shown below:
```bash
export WECO_API_KEY=<YOUR_WECO_API_KEY>
```

## Build & Deploy

We can create a function on the Weco AI [platform](https://www.aifunction.com) for the following task:
> "Analyze a business idea and provide a well reasoned evaluation. Return 'viability_score' (0-100), 'strengths' (list), 'weaknesses' (list), and 'next_steps' (list)."

If you use our online [platform](https://www.aifunction.com) to do this, then you'll need to retrieve your AI function:
```python
from aifn import AIFunction
idea_evaluator = AIFunction("BusinessIdeaAnalyzer-XYZ123"),  # Replace with your actual function name
print(f"{idea_evaluator.fn_name}/{idea_evaluator.version}")
```

Or, we can just stick to using Python to create our AI function:
```python
from aifn import build
idea_evaluator = build(
    task_description="Analyze a business idea and provide a well reasoned evaluation. Return 'viability_score' (0-100), 'strengths' (list), 'weaknesses' (list), and 'next_steps' (list).",
)
print(f"{idea_evaluator.fn_name}/{idea_evaluator.version}")
```

Now you can call your AI function, just like any other function, anywhere in your code!
```python
response = idea_evaluator("A subscription service for personalized, AI-generated bedtime stories for children.")
```

To learn how to get the most your of **your** AI functions, check out our [cookbook](../cookbook/cookbook.md), our [API reference](../api/api.md) and these end-to-end [examples](../cookbook/examples/maze_runner.md).

Happy building!
