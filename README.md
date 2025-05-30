- final exercise MCP Remote Agent with chat interface 
- Readme 
- convert to notewook, 
- add python snippets to the solutions


# AI Engineering with the Google Gemini 2.5 Model Family

**Track:** Anything Else (or MCP if focusing on protocol-level integration)

**Format:** Workshop

As AI continues to evolve with the arrival of models like Google's Gemini 2.5 signals a shift beyond simple chatbots towards effortable, dynamic, agentic AI systems capable of complex reasoning and tool use.

Many teams struggle to bridge the gap between potential of LLMs and building reliable, multi-step AI applications that solve real-world problems. This hands-on workshop that dives deep into the practical engineering aspects of working with the Gemini 2.5 family.

You will learn how to:

*   Build and Prototype with Google AI Studio & API
*   Tackle multi-modal input/output challenges.
*   Effectively use of function calling and tool integration.
*   Tune and cost and performance for Gemini 2.5.
*   Integrate MCP with Gemini 2.5 and sovling challenges like state management, transport, or authentication.

Whether you're aiming to build internal copilots, autonomous task agents, or knowledge-driven automation systems, mastering the engineering of Gemini 2.5 is key. Join us to learn the practical skills needed to take your AI applications beyond simple generation and into true agentic behavior.

## Workshop Outline

### Section 1: Introduction to Gemini API & Google AI Studio

*   **1.a: Google AI Studio and API Key Setup**
    *   Overview of Google AI Studio
    *   Obtaining your API Key
    *   Exploring available models and documentation
        *   *Existing Example: Model context windows (modified to show where to find model names)*
*   **1.b: Your First Gemini API Request (using Python/Colab)**
    *   Basic text generation
        *   *Existing Example: Simple text generation*
    *   Streaming responses
        *   *Existing Example: Streaming text*
    *   OpenAI API compatibility (mention/link if available in docs, placeholder for example)
    *   Model configurations (temperature, top_k, top_p, etc.)
        *   *Partially covered in existing examples like "System prompt" and "Structured output". Needs a dedicated example or section.*
    *   System Prompts
        *   *Existing Example: System prompt*

### Section 2: Multi-modal Capabilities

*   **2.a: Text & Code Generation with Long Context**
    *   Working with large text inputs (conceptual, show model context limits)
        *   *Existing Example: Model context windows*
    *   Code generation (beyond simple execution, e.g., explaining code, translating code)
        *   *(Placeholder: Need example beyond code execution)*
*   **2.b: Image Understanding**
    *   Image question answering (single and multiple images)
        *   *Existing Example: Image question answering*
    *   Object detection / Bounding boxes
        *   *Existing Example: Bounding boxes*
    *   Image segmentation
        *   *Existing Example: Image segmentation*
*   **2.c: Document Understanding**
    *   PDF data analysis and summarization
        *   *Existing Example: PDF and CSV data analysis and summarization (PDF part)*
    *   Extracting structured data from PDFs
        *   *Existing Example: Extract structured data from a PDF*
    *   Translating documents
        *   *Existing Example: Translate documents*
*   **2.d: Video Understanding**
    *   Video question answering
        *   *Existing Example: Video question answering*
    *   Video summarization
        *   *Existing Example: Video summarization*
    *   Video transcription
        *   *Existing Example: Video transcription*
    *   YouTube video summarization
        *   *Existing Example: YouTube video summarization*
*   **2.e: Audio Understanding & Generation**
    *   Audio question answering
        *   *Existing Example: Audio question answering*
    *   Audio transcription
        *   *Existing Example: Audio transcription*
    *   Audio summarization
        *   *Existing Example: Audio summarization*
    *   Text-to-Speech (TTS)
        *   *(Placeholder: New example based on `gemini-2.5-flash-preview-tts` or `gemini-2.5-pro-preview-tts`)*
*   **2.f: Image Generation**
    *   Image generation with Gemini models
        *   *Existing Example: Image generation (Gemini and Imagen) - Gemini part*
    *   Image generation with Imagen
        *   *Existing Example: Image generation (Gemini and Imagen) - Imagen part*
    *   Story telling with images (multi-turn image generation, or text-to-image series)
        *   *(Placeholder: Needs a dedicated example, could adapt existing image generation)*
    *   Image editing
        *   *Existing Example: Edit an image*
*   **2.g: Video Generation (Conceptual/Future)**
    *   Veo example (mention capabilities, link to announcements if available)
        *   *(Placeholder: Describe Veo, link to resources. No direct API example expected via `google-genai` at this time.)*

### Section 3: Agentic AI with Gemini - Function Calling and Tools

*   **3.a: Structured Outputs**
    *   Generating JSON responses
        *   *Existing Example: Structured output (Cats example)*
        *   *Existing Example: Extract structured data from a PDF (Invoice example)*
*   **3.b: Function Calling**
    *   Basic function calling example
        *   *Existing Example: Function calling & tool use (Weather example - basic part)*
    *   Parallel function calling
        *   *Existing Example: Function calling & tool use (Weather and Appointments - parallel part)*
    *   Automatic function calling (model decides when to call functions)
        *   *(Covered implicitly in existing examples, but could be highlighted)*
*   **3.c: Native Tools**
    *   Code Execution
        *   *Existing Example: Code execution (Sum of primes)*
    *   URL Context (providing URLs for the model to process)
        *   *Implicitly in: YouTube video summarization, PDF/CSV analysis from URL, Translate documents from URL. Explicit example could be good.*
        *   *(Placeholder: A simple "summarize this webpage" example)*
    *   Google Search (Grounded Responses)
        *   *Existing Example: Grounded responses with search tool*
    *   URL Context + Search
        *   *(Placeholder: Example combining URL input with search for broader context)*

### Section 4: Performance, Cost, and Tuning

*   **4.a: Token Counting**
    *   Counting text tokens (prompt and response)
        *   *Existing Example: Counting chat tokens*
    *   Counting multimodal input tokens (images, audio, video)
        *   *Existing Example: Calculating multimodal input tokens*
*   **4.b: Understanding "Thinking" & Reasoning Models**
    *   Explanation of how reasoning models work (e.g., `gemini-2.5-pro-exp-03-25` allow the model to show its reasoning steps)
        *   *Existing Example: Reasoning models*
    *   How reasoning steps are accounted for: While models may go through internal "thinking" or reasoning steps (visible in AI Studio for some experimental models), billing is based on the standard input and output tokens. The complexity of reasoning will influence the processing required and potentially the number of output tokens generated to articulate the reasoning, which is then billed.
*   **4.c: Managing Complex Requests (Conceptual "Thinking Budget")**
    *   There isn't a direct "thinking budget" to configure. Managing complex or potentially long-running requests involves:
        *   Clear and concise prompting.
        *   Breaking down very complex tasks into smaller, chained prompts if necessary.
        *   Setting appropriate output token limits (`max_output_tokens`).
        *   Using more capable models (e.g., Pro vs. Flash) for more complex reasoning tasks, understanding the associated cost implications per token.
*   **4.d: Using Different Models for Cost/Performance**
    *   Overview of model families (Flash, Pro) and when to use them
        *   *Briefly in: Model context windows. Needs expansion.*
    *   Strategies for choosing the right model
*   **4.e: Context Caching**
    *   Concept and benefits
    *   Example of using context caching
        *   *Existing Example: Context caching*
*   **4.f: Other Performance Considerations**
    *   Rate limits and retries
        *   *Existing Example: Rate limits and retries*
    *   Concurrent requests
        *   *Existing Example: Concurrent requests and generation*

### Section 5: Advanced Integration - MCP and Agentic Systems

*   **5.a: Model Context Protocol (MCP)**
    *   Overview of MCP
    *   MCP server example for function calling (e.g., the provided weather example)
        *   *Existing Example: Model Context Protocol (Weather MCP)*
*   **5.b: Building an Agentic Loop with MCP (e.g., Deep Wiki Agent)**
    *   State management in agentic loops
    *   Transport considerations
    *   Authentication with MCP services
    *   *(Placeholder: Design and implement a conceptual Deep Wiki agent example script using MCP. This will be a more involved example.)*
*   **5.c: Miscellaneous Useful Features**
    *   Embeddings
        *   *Existing Example: Embeddings generation*
    *   Safety Settings
        *   *Existing Example: Safety settings and filters*
    *   Using with LiteLLM
        *   *Existing Example: LiteLLM*

## Existing Examples Mapping

*   Text
    *   Simple text generation -> Section 1.b
    *   Streaming text -> Section 1.b
    *   System prompt -> Section 1.b
    *   Reasoning models -> Section 4.b
    *   Structured output -> Section 3.a
*   Images
    *   Image question answering -> Section 2.b
    *   Image generation (Gemini and Imagen) -> Section 2.f
    *   Edit an image -> Section 2.f
    *   Bounding boxes -> Section 2.b
    *   Image segmentation -> Section 2.b
*   Audio
    *   Audio question answering -> Section 2.e
    *   Audio transcription -> Section 2.e
    *   Audio summarization -> Section 2.e
*   Video
    *   Video question answering -> Section 2.d
    *   Video summarization -> Section 2.d
    *   Video transcription -> Section 2.d
    *   YouTube video summarization -> Section 2.d & 3.c (URL Context part)
*   PDFs and other data types
    *   PDF and CSV data analysis and summarization -> Section 2.c (PDF), (CSV part needs a place or separate example)
    *   Translate documents -> Section 2.c & 3.c (URL Context part)
    *   Extract structured data from a PDF -> Section 2.c & 3.a
*   Agentic behaviour
    *   Function calling & tool use -> Section 3.b
    *   Code execution -> Section 3.c
    *   Model Context Protocol -> Section 5.a
    *   Grounded responses with search tool -> Section 3.c
*   Token counting & context windows
    *   Model context windows -> Section 1.a & 2.a & 4.d
    *   Counting chat tokens -> Section 4.a
    *   Calculating multimodal input tokens -> Section 4.a
    *   Context caching -> Section 4.e
*   Miscellaneous
    *   Rate limits and retries -> Section 4.f
    *   Concurrent requests and generation -> Section 4.f
    *   Embeddings generation -> Section 5.c
    *   Safety settings and filters -> Section 5.c
    *   LiteLLM -> Section 5.c

## TODO / Placeholders:

*   **Section 1.b:** Dedicated example for model configurations (temperature, top_k, etc.).
*   **Section 1.b:** OpenAI API compatibility example/documentation.
*   **Section 2.a:** Example for advanced code generation (explaining, translating code).
*   **Section 2.c:** Decide where the CSV analysis part of "PDF and CSV data analysis" fits best or if it needs a separate small example.
*   **Section 2.e:** New example for Text-to-Speech (TTS).
*   **Section 2.f:** Dedicated example for story telling with images.
*   **Section 2.g:** Describe Veo, link to resources for video generation.
*   **Section 3.c:** Explicit example for URL Context (e.g., "summarize this webpage").
*   **Section 3.c:** Example combining URL context + search.
*   **Section 4.b:** Clarify "thought tokens" counting/billing.
*   **Section 4.c:** Explain "thinking budget" concept.
*   **Section 4.d:** Expand on choosing different models for cost/performance.
*   **Section 5.b:** Design and implement a conceptual Deep Wiki agent example script using MCP.

---
*This README.md provides a structured outline for the workshop. The next steps would involve filling in the placeholder content, creating new examples where needed, and potentially reorganizing the `gemini-by-example.md` content into separate files per section or example for better manageability if the workshop becomes very large.*
