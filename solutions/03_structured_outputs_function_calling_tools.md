# Part 3: Structured Outputs, Function Calling & Native Tools

This section covers three powerful capabilities of the Gemini API: structured outputs for extracting information into defined schemas, function calling for connecting to external tools and APIs, and native tools like Google Search for enhanced capabilities.

```python
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import List, Optional
import sys
import os

IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    from google.colab import userdata
    GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
else:
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', None)

# Create client with api key
MODEL_ID = "gemini-2.5-flash-preview-05-20"
client = genai.Client(api_key=GOOGLE_API_KEY)
```

## 1. Structured Outputs

Structured outputs allow you to constrain Gemini to respond with JSON in a specific format instead of unstructured text. This is essential for:
- **Data extraction**: Converting unstructured text into structured data
- **API integration**: Getting consistent formats for downstream processing  
- **Database insertion**: Ensuring data matches your schema requirements
- **Quality control**: Validating that responses contain required fields

```python
class Recipe(BaseModel):
    recipe_name: str
    ingredients: List[str]
    prep_time_minutes: int
    difficulty: str  # "easy", "medium", "hard"
    servings: int

# Using Pydantic models for structured output
response = client.models.generate_content(
    model=MODEL_ID,
    contents="Give me 2 popular cookie recipes with ingredients and prep details.",
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=List[Recipe],
    ),
)

# Get structured data directly
recipes: List[Recipe] = response.parsed
print(recipes)
```

## !! Exercise: PDF to Structured Data !!

Extract structured information from a PDF invoice or document using the Files API and structured outputs.

1. **Upload a PDF** using the Files API
2. **Define a Pydantic schema** for invoice/document data
3. **Extract structured data** from the PDF content
4. **Print the results** in both structured and JSON format

```python
class InvoiceItem(BaseModel):
    description: str
    quantity: int
    unit_price: float
    total: float

class InvoiceData(BaseModel):
    invoice_number: str
    date: str
    vendor_name: str
    vendor_address: str
    total_amount: float
    items: List[InvoiceItem]

# Upload a PDF file (replace with your PDF path)
pdf_file_path = "../assets/data/rewe_invoice.pdf"

# Upload the file
file_id = client.files.upload(path=pdf_file_path)

# Extract structured data from PDF
response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        "Extract all invoice information from this PDF including items, vendor details, and totals.",
        file_id
    ],
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=InvoiceData,
    ),
)

# Get structured invoice data
invoice: InvoiceData = response.parsed
print(f"Invoice #{invoice.invoice_number}")
print(f"Vendor: {invoice.vendor_name}")
print(f"Date: {invoice.date}")
print(f"Total: ${invoice.total_amount}")
print(f"Items ({len(invoice.items)}):")
for item in invoice.items:
    print(f"  - {item.description}: {item.quantity} x ${item.unit_price} = ${item.total}")
```

## 2. Function Calling

Function calling allows Gemini to intelligently decide when to call specific functions you define. This enables:
- **External API integration**: Connect to weather, stocks, databases
- **Dynamic calculations**: Perform real-time computations
- **System interaction**: Execute commands or retrieve system information
- **Multi-step workflows**: Chain function calls for complex tasks


```python

def get_weather(location: str) -> dict:
    """Gets current weather for a location.
    
    Args:
        location: The city name, e.g. "San Francisco"
        
    Returns:
        Weather information dictionary
    """
    # Mock weather data - in real use, you'd call a weather API
    weather_data = {
        "temperature": 22,
        "condition": "sunny", 
        "humidity": 60,
        "location": location,
        "feels_like": 24
    }
    print(f"🌤️ FUNCTION CALLED: get_weather(location='{location}')")
    return weather_data

# Define function declarations for the model
weather_function = {
    "name": "get_weather",
    "description": "Gets current weather for a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city name"
            }
        },
        "required": ["location"]
    }
}
tools = [types.Tool(function_declarations=[weather_function])]

# Send request with function declarations
response = client.models.generate_content(
    model=MODEL_ID,
    contents="What's the weather like in London?",
    config=types.GenerateContentConfig(tools=tools)
)

# Check for function calls
function_call = response.candidates[0].content.parts[0].function_call
print(f"Model wants to call: {function_call.name}")
print(f"With arguments: {dict(function_call.args)}")

# Execute the function
if function_call.name == "get_weather":
    result = get_weather(**function_call.args)
else:
    result = {"error": "Unknown function"}

print(f"Function result: {result}")

# Send function result back to model
function_response = types.Part.from_function_response(
    name=function_call.name,
    response={"result": result}
)

# Get final response
final_response = client.models.generate_content(
    model=MODEL_ID,
    contents=[
        response.candidates[0].content.parts[0],  # Original function call
        function_response  # Function result
    ],
    config=types.GenerateContentConfig(tools=tools)
)

print(f"\nFinal response: {final_response.text}")
```

### Automatic Function Calling (Python Only)

The Python SDK can automatically handle function execution for you:

```python
def calculate_area(length: float, width: float) -> dict:
    """Calculate the area of a rectangle.
    
    Args:
        length: Length of the rectangle
        width: Width of the rectangle

    Returns:
        Price calculations
    """
    area = length * width
    print(f"CALC: {length} × {width} = {area}")
    return {"operation": "area", "result": area}

# Using automatic function calling - much simpler!
config = types.GenerateContentConfig(
    tools=[get_weather, calculate_area]  # Pass functions directly
)

response = client.models.generate_content(
    model=MODEL_ID,
    contents="What's the weather in Tokyo and what's the area of a 5x3 meter room?",
    config=config
)

print(response.text)  # SDK handles function calls automatically
```

## !! Exercise: Calculator Tool !!

Create a calculator with multiple mathematical operations using function calling.

1. **Define calculator functions** (add, subtract, multiply, divide)
2. **Test single operations** with manual function calling
3. **Test complex expressions** with automatic function calling
4. **Handle error cases** (division by zero)

```python
def add(a: float, b: float) -> dict:
    """Add two numbers.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Sum of the numbers
    """
    result = a + b
    print(f"CALC: {a} + {b} = {result}")
    return {"operation": "addition", "result": result}

def subtract(a: float, b: float) -> dict:
    """Subtract two numbers.
    
    Args:
        a: First number  
        b: Second number
        
    Returns:
        Difference of the numbers
    """
    result = a - b
    print(f"CALC: {a} - {b} = {result}")
    return {"operation": "subtraction", "result": result}

def multiply(a: float, b: float) -> dict:
    """Multiply two numbers.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        Product of the numbers
    """
    result = a * b
    print(f"CALC: {a} × {b} = {result}")
    return {"operation": "multiplication", "result": result}

def divide(a: float, b: float) -> dict:
    """Divide two numbers.
    
    Args:
        a: Dividend
        b: Divisor
        
    Returns:
        Quotient of the numbers
    """
    if b == 0:
        print(f"CALC: Error - Division by zero")
        return {"operation": "division", "error": "Division by zero"}
    
    result = a / b
    print(f"CALC: {a} ÷ {b} = {result}")
    return {"operation": "division", "result": result}

# Test the calculator with automatic function calling
calculator_tools = [add, subtract, multiply, divide]

# Single operation
response = client.models.generate_content(
    model=MODEL_ID,
    contents="What is 15 multiplied by 7?",
    config=types.GenerateContentConfig(tools=calculator_tools)
)
print("Single operation:")
print(response.text)

# Complex calculation
response = client.models.generate_content(
    model=MODEL_ID,
    contents="Calculate (25 + 15) × 3 - 10. Do this step by step.",
    config=types.GenerateContentConfig(tools=calculator_tools)
)
print("\nComplex calculation:")
print(response.text)
```

## 3. Native Tools

Gemini provides native tools for enhanced capabilities like searching the web and analyzing URL content.

### Google Search Integration

**Use cases:**
- Current events and news
- Real-time data lookup
- Fact verification
- Research assistance

```python
# Define Google Search tool
google_search_tool = types.Tool(google_search=types.GoogleSearch())

# Current events query
response = client.models.generate_content(
    model=MODEL_ID,
    contents="What are the latest developments in renewable energy technology in 2025?",
    config=types.GenerateContentConfig(
        tools=[google_search_tool],
    )
)

print("🔍 Current Renewable Energy News:")
print(response.text)
```

### URL Context Tool

**Use cases:**
- Website content analysis
- Documentation summarization
- Competitive research
- Content extraction

```python
# URL context for analyzing specific web pages
url_context_tool = types.Tool(url_context=types.UrlContext())

response = client.models.generate_content(
    model=MODEL_ID,
    contents="Summarize the key features and benefits mentioned on https://www.python.org/about/ in 3 bullet points.",
    config=types.GenerateContentConfig(
        tools=[url_context_tool],
    )
)

print("🌐 Python.org Summary:")
print(response.text)
```

### Code Execution Tool

Gemini can execute Python code to perform calculations, create visualizations, and process data.

```python
# Code execution tool
code_execution_tool = types.Tool(code_execution={})

response = client.models.generate_content(
    model=MODEL_ID,
    contents="Create a bar chart showing the population of the 5 largest cities in the world. Use matplotlib.",
    config=types.GenerateContentConfig(
        tools=[code_execution_tool],
    )
)

print("Code execution result:")
print(response.text)
```


## !! Exercise: Data Analysis with Code Execution !!

Task:
- Combine the code execution tool with the google search tool
- Search for information online, e.g. the population of the 5 largest cities in the world
- Create a chart/analysis of the data

```python
prompt = """
Search for the population of the 5 largest cities in the world and create a bar chart.
"""

code_execution_tool = types.Tool(code_execution={})
google_search_tool = types.Tool(google_search=types.GoogleSearch())

response = client.models.generate_content(
    model=MODEL_ID,
    contents=prompt,
    config=types.GenerateContentConfig(
        tools=[code_execution_tool, google_search_tool],
    )
)

print("🏥 Healthcare AI Research:")
print(response.text)
```

## Recap & Next Steps

**What You've Learned:**
- Structured outputs using Pydantic models for reliable data extraction and validation
- Function calling to integrate external APIs, databases, and custom business logic
- Native tools including Google Search, URL context analysis, and code execution
- Combining multiple tools for comprehensive workflows and complex problem-solving

**Key Takeaways:**
- Structured outputs ensure consistent data formats for downstream applications
- Function calling enables seamless integration with external systems and real-time data
- Native tools provide powerful capabilities without additional setup or infrastructure
- Tool combinations unlock sophisticated workflows and multi-step problem solving
- Proper validation and error handling are crucial for reliable tool interactions

**Next Steps:** Continue with [Part 4: Model Context Protocol (MCP)](https://github.com/philschmid/gemini-2.5-ai-engineering-workshop/blob/main/solutions/04_model_context_protocol_mcp.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philschmid/gemini-2.5-ai-engineering-workshop/blob/main/solutions/04_model_context_protocol_mcp.ipynb)

**More Resources:**
- [Structured Output Documentation](https://ai.google.dev/gemini-api/docs/structured-output?lang=python)
- [Function Calling Documentation](https://ai.google.dev/gemini-api/docs/function-calling?lang=python)
- [Grounding with Google Search](https://ai.google.dev/gemini-api/docs/grounding)
- [URL Context Tool](https://ai.google.dev/gemini-api/docs/url-context)
- [Code Execution Documentation](https://ai.google.dev/gemini-api/docs/code-execution)
