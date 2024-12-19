## Getting Started

A client facing API for interacting with the [Weco AI](https://www.weco.ai/)'s AI function [platform](https://www.aifunction.com). It empowers you to go from zero to AI in just a few seconds!

Here are a few features our users often ask about. Feel free to follow along:

<a href="https://colab.research.google.com/github/WecoAI/aifn-python/blob/main/examples/cookbook.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" width=110 height=20/></a>
<a target="_blank" href="https://lightning.ai/new?repo_url=https%3A%2F%2Fgithub.com%2FWecoAI%2Faifn-python%2Fblob%2Fmain%2Fexamples%2Fcookbook.ipynb"><img src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/app-2/studio-badge.svg" alt="Open in Studio" width=100 height=20/></a>


```python
# Install the package
%pip install aifn
```

You can find/setup your API key [here](https://www.aifunction.com/account/api-keys).


```python
import os
os.environ["WECO_API_KEY"] = "YOUR_WECO_API_KEY"
```

You can build powerful AI functions for complex tasks quickly and without friction. For example, you can create an AI function on our [platform](https://www.aifunction.com/function/new) with a simple description as shown below:

> "Analyze a business idea and provide a well reasoned evaluation. Return 'viability_score' (0-100), 'strengths' (list), 'weaknesses' (list), and 'next_steps' (list)."

Once the function has been built, you can call, test and deploy it anywhere in your code with just three lines of code:


```python
from aifn import AIFunction

idea_evaluator = AIFunction("BusinessIdeaAnalyzer-XYZ123") # Replace with your actual function name

response = idea_evaluator("A subscription service for personalized, AI-generated bedtime stories for children.").output
print(response)
```

## Structured Output

There is a 100% guarantee of a structured output on an function call! This means that you can pass any unstructured input in and get back a valid structured output that follows the JSON schema optimally determined by the AI function for your task. You can also edit, write and replace the JSON schema for an AI function on our [platform](https://www.aifunction.com/).

Here's an example where we extract relevant information about the different object in this image:

![Objects on Table](https://us.images.westend61.de/0001348304i/directly-above-shot-of-various-objects-on-table-EYF01650.jpg)


```python
from aifn import build

information_extractor = build("Describe the unique set of the objects present in the image. Provide the 'object', a 'description' and 'count' of how many times that particular object appeared in the image.")

# You may be wondering what the metadata here is. We'll get to that in a second.
object_dict, metadata = information_extractor(images_input=["https://us.images.westend61.de/0001348304i/directly-above-shot-of-various-objects-on-table-EYF01650.jpg"])
for object in object_dict['objects']:
    print(f"{object['object']}: {object['description']} (appeared {object['count']} times)")
```

## AI Function and Function Response Metadata

We do recommend building AI functions on our [platform](https://www.aifunction.com/) to prototype faster and achieve the right balance between speed, intelligence and cost. Here's how you can find the AI function we built in the previous example on the [platform](https://www.aifunction.com/). Simply extract the function name, and use it to find the corresponding AI function on the [platform](https://www.aifunction.com/). If you need to know the exact version, you can extract that from the `AIFunction` class as well to find the exact version being used.


```python
fn_name = information_extractor.fn_name
print(f"Function name: {fn_name}")
version = information_extractor.version
print(f"Version: {version}")
```

If you're a developer and you want to use AI powered functions, you probably care about things like the input and output token counts and the latency of the AI function. Its easily accessible! Let's look at how we can get this information for the previous function call.


```python
n_input_tokens = metadata["in_tokens"]
n_output_tokens = metadata["out_tokens"]
latency_milliseconds = metadata["latency_ms"]
print(f"Input tokens: {n_input_tokens}, Output tokens: {n_output_tokens}, Latency: {latency_milliseconds}ms")
```

## Sync-Async Duality

AI Functions can be made to be both synchronous or asynchronous functions, allowing for more flexible use.


```python
from aifn import build

# Build an synchronous function
translator = build("Return the 'translation' of english to french.")
translation = translator("Hello, how are you?").output['translation']
print(translation)

# Make it asynchronous
async_translator = translator.make_async()
output, metadata = await async_translator("Hello, how are you?")
print(output['translation'])

# Build an asynchronous function
translator = build("Return the 'translation' of english to french.", is_async=True)
output, metadata = await translator("Hello, how are you?")
print(output['translation'])

# Make it synchronous
sync_translator = translator.make_sync()
translation = sync_translator("Hello, how are you?").output['translation']
print(translation)
```

## Batching

In the previous examples, we've shown you how to call an AI function with just one input. We understand that sometimes you want to submit a large batch of inputs to be processed in concurrently. Every AI function, whether synchronous or asynchronous, can perform batch processing.


```python
task_evaluator = build(task_description="I want to know if AI can solve a problem for me, how easy it is to arrive at a solution and whether any helpful tips for me along the way. Help me understand this through - 'feasibility', 'justification', and 'suggestions'.")

task1 = {
    "text_input": "I want to train a model to predict house prices using the Boston Housing dataset hosted on Kaggle."
}
task2 = {
    "text_input": "I want to train a model to classify digits using the MNIST dataset hosted on Kaggle using a Google Colab notebook. Attached is an example of what some of the digits would look like.",
}
responses = task_evaluator.batch([task1, task2])
for response in responses:
    print("=" * 50)
    print(response.output)
    print("=" * 50)
```

## Multimodality

Our AI functions can interpret complex visual information, follow instructions in natural language and provide practical insights. We accept a variety of different forms of image input:
1. Base64 encoding
2. Publically available URL
3. Local image path

Let's explore how we can have an AI function manage a part of our household. By running this once a month, I am able to find ways to cut down my energy consumption and ultimately save me money!


```python
import base64

task_description = """
You are a smart home energy analyzer that can process images of smart meters, home exteriors, 
and indoor spaces to provide energy efficiency insights. The analyzer should:
    1. Interpret smart meter readings
    2. Assess home features relevant to energy consumption
    3. Analyze thermostat settings
    4. Provide energy-saving recommendations
    5. Evaluate renewable energy potential

The output should include:
    - 'energy_consumption': current usage and comparison to average
    - 'home_analysis': visible energy features and potential issues
    - 'thermostat_settings': current settings and recommendations
    - 'energy_saving_recommendations': actionable suggestions with estimated savings
    - 'renewable_energy_potential': assessment of current and potential renewable energy use
    - 'estimated_carbon_footprint': current footprint and potential reduction
"""

energy_analyzer = build(task_description=task_description)

request = """
Analyze these images of my home and smart meter to provide energy efficiency insights 
and recommendations for reducing my electricity consumption.
"""

# Base64 encoded image
with open("/path/to/home_exterior.jpeg", "rb") as img_file:
    my_home_exterior = base64.b64encode(img_file.read()).decode('utf-8')

analysis = energy_analyzer(
    text_input=request,
    images_input=[
        "https://example.com/my_smart_meter_reading.png",  # Public URL
        f"data:image/jpeg;base64,{my_home_exterior}",      # Base64 encoding
        "/path/to/living_room_thermostat.jpg"              # Local image path
    ]
).output

for key, value in analysis.items(): print(f"{key}: {value}")
```

## Grounding via Access to the Web

Some of our models even have access to the web. Here's a complete list of models that do:


| Model Name                        | Provider    |
|-----------------------------------|-------------|
| gemini-1.5-pro-002-online         | Google      |
| gemini-1.5-flash-002-online       | Google      |
| llama-3.1-sonar-huge-128k-online  | Perplexity  |
| llama-3.1-sonar-large-128k-online | Perplexity  |
| llama-3.1-sonar-small-128k-online | Perplexity  |

To enable this, you need to the **Settings** menu of the specific function and select one of the above as the model to use. After, that simply change the `version` to the latest version or pass $-1$ when retrieving the `AIFunction` and deploy your code to production! An even easier way when deploying your code to production is to pass a particular alias that you set for the function version on our [platform](https://www.aifunction.com). This way, whenever you want to switch to a particular function version, you just need to set the alias on our [platform](https://www.aifunction.com) and it will automatically be deployed to your production code!

If your AI function has access to the web, it opens up a whole new way to ground your model outputs in every day events such as the stock market, current news and more!

## Interpretability

You can now understand why a model generated an output. For this, you'll need to enable Chain of Thought (CoT) for the function version on the [platform](https://www.aifunction.com). You can find this under the **Settings** for a particular function version. Then, to view the model's reasoning behind an output, simply use `return_reasoning=True` when you call the function on an input! This can be done for both synchronous and asynchronous and can be be done with the batched inputs as well. The reasoning behind the output can be found in the `.metadata["reasoning_steps"]` attribute of the function response.


```python
task_evaluator = build(task_description="I want to know if AI can solve a problem for me, how easy it is to arrive at a solution and whether any helpful tips for me along the way. Help me understand this through - 'feasibility', 'justification', and 'suggestions'.")

output, metadata = task_evaluator("I want to train a model to predict house prices using the Boston Housing dataset hosted on Kaggle.", return_reasoning=True)
for key, value in output.items(): print(f"{key}: {value}")
for i, step in enumerate(metadata["reasoning_steps"]): print(f"Step {i+1}: {step}")
```

## End-to-End Examples

For more on what AI functions can do, check out these [examples](examples/maze_runner.md).
