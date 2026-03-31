"""Mock LLM backend with quality-differentiated simulated responses."""

from __future__ import annotations

import asyncio
import random
import time
from typing import Literal

from src.models.base import ModelBackend, GenerationParams, GenerationResult
from src.utils.tokenizer import estimate_token_count


# Response templates organized by category and quality tier
RESPONSE_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "greeting": {
        "small": [
            "Hello! How can I help you today?",
            "Hi there! What would you like to know?",
        ],
        "large": [
            "Hello! I'm here to help you with any questions or tasks you might have. "
            "Whether you need information, analysis, creative writing, or problem-solving, "
            "feel free to ask me anything.",
            "Hi there! I'm ready to assist you. I can help with a wide range of topics "
            "including research, coding, writing, math, and more. What's on your mind?",
        ],
    },
    "factual": {
        "small": [
            "Based on general knowledge, {topic} is a well-known concept. "
            "It refers to the fundamental principles that govern this area.",
            "The answer relates to {topic}. This is a common topic that involves "
            "several key aspects and has been studied extensively.",
        ],
        "large": [
            "Let me provide a comprehensive answer about {topic}.\n\n"
            "**Overview:** This is a multifaceted subject that encompasses several key areas. "
            "At its core, it involves understanding the fundamental principles and their "
            "practical applications.\n\n"
            "**Key Points:**\n"
            "1. The foundational concepts establish the framework for understanding\n"
            "2. Historical development has shaped current practices\n"
            "3. Modern applications continue to evolve\n"
            "4. There are ongoing debates and research in this field\n\n"
            "**Conclusion:** Understanding {topic} requires considering both theoretical "
            "foundations and practical implications.",
        ],
    },
    "code": {
        "small": [
            "Here's a basic implementation:\n\n```python\ndef solution():\n    "
            "# Basic implementation\n    result = []\n    "
            "for item in data:\n        result.append(process(item))\n    "
            "return result\n```\n\nThis should work for simple cases.",
        ],
        "large": [
            "Here's a well-structured implementation with error handling and documentation:\n\n"
            "```python\nfrom typing import List, Optional\n\n\n"
            "class Solution:\n"
            '    """Implements an efficient solution with proper error handling."""\n\n'
            "    def __init__(self, config: Optional[dict] = None):\n"
            "        self.config = config or {}\n"
            "        self._cache = {}\n\n"
            "    def solve(self, data: List[any]) -> List[any]:\n"
            '        """Process data with caching and validation.\n\n'
            "        Args:\n"
            "            data: Input data to process.\n\n"
            "        Returns:\n"
            "            Processed results.\n\n"
            "        Raises:\n"
            "            ValueError: If data is invalid.\n"
            '        """\n'
            "        if not data:\n"
            '            raise ValueError("Data cannot be empty")\n\n'
            "        results = []\n"
            "        for item in data:\n"
            "            if item in self._cache:\n"
            "                results.append(self._cache[item])\n"
            "            else:\n"
            "                result = self._process(item)\n"
            "                self._cache[item] = result\n"
            "                results.append(result)\n"
            "        return results\n\n"
            "    def _process(self, item: any) -> any:\n"
            '        """Process a single item."""\n'
            "        # Implementation details here\n"
            "        return item\n```\n\n"
            "**Key design decisions:**\n"
            "- Uses caching to avoid redundant computation\n"
            "- Input validation prevents silent failures\n"
            "- Type hints improve code clarity\n"
            "- Docstrings follow Google style for consistency",
        ],
    },
    "math": {
        "small": [
            "The answer can be computed as follows:\n"
            "Using the basic formula, we get the result. "
            "The calculation involves standard mathematical operations.",
        ],
        "large": [
            "Let me solve this step by step.\n\n"
            "**Step 1: Identify the problem type**\n"
            "This is a mathematical problem that requires careful analysis.\n\n"
            "**Step 2: Set up the equation**\n"
            "We can express this as a formal mathematical relationship.\n\n"
            "**Step 3: Apply the solution method**\n"
            "Using the appropriate technique, we systematically work through "
            "the computation.\n\n"
            "**Step 4: Verify the result**\n"
            "We can verify our answer by substituting back into the original "
            "equation to confirm correctness.\n\n"
            "**Final Answer:** The result follows from the systematic application "
            "of the method described above.",
        ],
    },
    "reasoning": {
        "small": [
            "This is an interesting question. The main points to consider are "
            "the primary factors involved and their relationships. "
            "Overall, it depends on the specific context and requirements.",
        ],
        "large": [
            "This is an excellent question that requires careful analysis. "
            "Let me break it down systematically.\n\n"
            "**Analysis Framework:**\n\n"
            "1. **Core Premise:** We need to examine the fundamental assumptions "
            "and establish what we know to be true.\n\n"
            "2. **Contributing Factors:**\n"
            "   - Factor A: This plays a significant role because...\n"
            "   - Factor B: This interacts with Factor A to create...\n"
            "   - Factor C: Often overlooked, but critical for...\n\n"
            "3. **Counterarguments:**\n"
            "   It's important to consider opposing viewpoints. Some argue that "
            "the relationship is more nuanced than it appears.\n\n"
            "4. **Synthesis:**\n"
            "   When we combine all these factors, the most supported conclusion is "
            "that the outcome depends on the interplay between multiple variables.\n\n"
            "**Key Takeaway:** The answer is not binary but exists on a spectrum, "
            "and the optimal approach depends on weighing the relevant trade-offs "
            "in context.",
        ],
    },
    "general": {
        "small": [
            "That's a good question. Here's a brief answer based on the available "
            "information. The key point is that this topic has several important aspects "
            "to consider.",
        ],
        "large": [
            "Thank you for this thoughtful question. Let me provide a comprehensive response.\n\n"
            "**Background:**\n"
            "This topic has a rich history and multiple dimensions worth exploring. "
            "Understanding it requires considering both the theoretical framework "
            "and practical implications.\n\n"
            "**Detailed Analysis:**\n"
            "There are several key aspects to consider:\n\n"
            "1. **Historical Context:** The development of this area has been shaped "
            "by significant events and contributions.\n\n"
            "2. **Current State:** Today, we see active research and development "
            "continuing to push boundaries.\n\n"
            "3. **Future Outlook:** Emerging trends suggest continued evolution "
            "with potential for significant breakthroughs.\n\n"
            "**Practical Implications:**\n"
            "For those working in this field, the key considerations are maintaining "
            "awareness of best practices while adapting to new developments.\n\n"
            "I hope this provides a helpful overview. Feel free to ask if you'd like "
            "me to dive deeper into any specific aspect.",
        ],
    },
}


def _classify_prompt_category(prompt: str) -> str:
    """Classify a prompt into a response category."""
    prompt_lower = prompt.lower()

    greetings = ["hello", "hi ", "hey", "good morning", "good evening", "howdy"]
    if any(prompt_lower.startswith(g) for g in greetings):
        return "greeting"

    code_markers = ["code", "function", "implement", "program", "debug", "class", "def ", "```"]
    if any(m in prompt_lower for m in code_markers):
        return "code"

    math_markers = ["calculate", "compute", "solve", "equation", "integral", "derivative", "sum of"]
    if any(m in prompt_lower for m in math_markers):
        return "math"

    reasoning_markers = ["explain", "why", "analyze", "compare", "evaluate", "reason", "argue"]
    if any(m in prompt_lower for m in reasoning_markers):
        return "reasoning"

    factual_markers = ["what is", "who is", "when", "where", "define", "describe"]
    if any(m in prompt_lower for m in factual_markers):
        return "factual"

    return "general"


class MockBackend(ModelBackend):
    """Simulated LLM backend with quality-differentiated responses.

    Produces realistic, differentiated outputs to demonstrate routing value:
    - Small model: shorter, simpler responses
    - Large model: detailed, structured, comprehensive responses
    """

    def __init__(
        self,
        model_name: str,
        tier: Literal["small", "large"],
        avg_latency_ms: float = 50,
        latency_jitter: float = 0.2,
    ) -> None:
        self._model_name = model_name
        self._tier = tier
        self._avg_latency_ms = avg_latency_ms
        self._latency_jitter = latency_jitter

    def model_name(self) -> str:
        return self._model_name

    def model_tier(self) -> str:
        return self._tier

    async def generate(self, prompt: str, params: GenerationParams | None = None) -> GenerationResult:
        """Generate a simulated response with realistic latency."""
        params = params or GenerationParams()
        start = time.perf_counter()

        # Simulate processing time
        jitter = random.uniform(1 - self._latency_jitter, 1 + self._latency_jitter)
        delay_s = (self._avg_latency_ms * jitter) / 1000
        await asyncio.sleep(delay_s)

        # Generate response based on category and tier
        category = _classify_prompt_category(prompt)
        templates = RESPONSE_TEMPLATES.get(category, RESPONSE_TEMPLATES["general"])
        tier_templates = templates.get(self._tier, templates["small"])
        response_text = random.choice(tier_templates)

        # Substitute topic placeholder if present
        topic_words = prompt.split()[:5]
        topic = " ".join(topic_words)
        response_text = response_text.replace("{topic}", topic)

        # Truncate to max_tokens if needed
        words = response_text.split()
        if len(words) > params.max_tokens:
            response_text = " ".join(words[:params.max_tokens])

        elapsed_ms = (time.perf_counter() - start) * 1000
        prompt_tokens = estimate_token_count(prompt)
        completion_tokens = estimate_token_count(response_text)

        return GenerationResult(
            text=response_text,
            model_name=self._model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=elapsed_ms,
            metadata={
                "category": category,
                "tier": self._tier,
                "simulated": True,
            },
        )

    async def health_check(self) -> bool:
        """Mock backend is always healthy."""
        return True
