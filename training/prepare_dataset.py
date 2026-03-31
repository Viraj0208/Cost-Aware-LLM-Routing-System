"""Prepare a complexity-labeled dataset for training the router classifier.

Generates a dataset of prompts labeled as 'simple' (0) or 'complex' (1)
using a combination of public datasets and heuristic labeling.

Usage:
    python -m training.prepare_dataset --output training/data/router_dataset.csv
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

# Simple prompts — factual Q&A, definitions, short tasks
SIMPLE_TEMPLATES = [
    "What is {topic}?",
    "Define {topic}.",
    "Who invented {topic}?",
    "When was {topic} created?",
    "What does {abbrev} stand for?",
    "Name three types of {topic}.",
    "Is {statement} true or false?",
    "What is the capital of {country}?",
    "Translate '{phrase}' to {language}.",
    "What color is {object}?",
    "How many {unit} are in a {larger_unit}?",
    "List five {category}.",
    "Who wrote {book}?",
    "What year did {event} happen?",
    "Summarize {topic} briefly.",
]

SIMPLE_FILLS = {
    "topic": [
        "photosynthesis", "gravity", "democracy", "the internet", "DNA",
        "machine learning", "blockchain", "an API", "a database", "HTTP",
        "Python", "JavaScript", "an operating system", "cloud computing",
        "artificial intelligence", "the water cycle", "evolution", "atoms",
        "electricity", "magnetism", "the solar system", "bacteria",
        "a programming language", "an algorithm", "the stock market",
        "inflation", "supply and demand", "a neural network", "a compiler",
        "a REST API", "version control", "Git", "a linked list",
    ],
    "abbrev": ["CPU", "RAM", "HTTP", "SQL", "API", "HTML", "CSS", "JSON", "URL", "DNS"],
    "statement": [
        "the Earth is round", "water boils at 100C", "Python is compiled",
        "HTML is a programming language", "the Sun orbits Earth",
    ],
    "country": ["France", "Japan", "Brazil", "Egypt", "Canada", "India", "Germany"],
    "phrase": ["hello", "thank you", "goodbye", "good morning", "how are you"],
    "language": ["Spanish", "French", "Japanese", "German", "Italian"],
    "object": ["the sky", "grass", "the sun", "blood", "snow"],
    "unit": ["centimeters", "minutes", "bytes", "grams", "milliliters"],
    "larger_unit": ["meter", "hour", "kilobyte", "kilogram", "liter"],
    "category": ["fruits", "programming languages", "planets", "countries", "colors"],
    "book": ["Romeo and Juliet", "1984", "The Great Gatsby", "Hamlet"],
    "event": ["World War II end", "the moon landing", "the internet invented"],
}

# Complex prompts — reasoning, code, multi-step analysis
COMPLEX_TEMPLATES = [
    "Explain {complex_topic} step by step, including the mathematical foundations and practical applications.",
    "Write a Python function that implements {algorithm}. Include type hints, error handling, and analyze the time complexity.",
    "Compare and contrast {concept_a} and {concept_b}. Analyze their strengths, weaknesses, and appropriate use cases with examples.",
    "Implement {data_structure} in Python with {operations}. Include comprehensive unit tests.",
    "Analyze the implications of {topic} on {domain}. Consider multiple perspectives and provide evidence-based arguments.",
    "Design a {system_type} architecture for {use_case}. Include component diagrams, API specifications, and scalability considerations.",
    "Derive the {formula} from first principles. Show each step of the proof and explain the assumptions.",
    "Debug the following code and explain the root cause:\n```python\n{buggy_code}\n```\nProvide a corrected version with explanation.",
    "Evaluate the trade-offs between {option_a} and {option_b} for {scenario}. Consider performance, maintainability, and cost.",
    "Write a comprehensive implementation of {pattern} design pattern in Python. Demonstrate with a real-world example and discuss when to use it.",
    "1. What is {topic_a}?\n2. How does it relate to {topic_b}?\n3. What are the key differences?\n4. When should you use each?\n5. Provide code examples.",
    "Implement a {ml_task} model from scratch without using ML frameworks. Include data preprocessing, training loop, and evaluation metrics.",
    "Solve the following mathematical problem step by step: {math_problem}. Verify your answer.",
    "Explain why {phenomenon} occurs. Discuss the underlying mechanisms, provide analogies, and address common misconceptions.",
]

COMPLEX_FILLS = {
    "complex_topic": [
        "transformer attention mechanisms", "gradient descent optimization",
        "distributed consensus algorithms", "quantum computing fundamentals",
        "compiler design phases", "TCP/IP networking stack",
        "database indexing strategies", "garbage collection algorithms",
        "public key cryptography", "neural network backpropagation",
    ],
    "algorithm": [
        "a balanced binary search tree", "Dijkstra's shortest path",
        "A* pathfinding with multiple heuristics", "a merge sort with visualization",
        "consistent hashing", "the Raft consensus algorithm",
        "a bloom filter", "a skip list", "an LRU cache",
    ],
    "concept_a": [
        "microservices", "SQL databases", "REST APIs", "synchronous programming",
        "object-oriented programming", "Kubernetes", "batch processing",
    ],
    "concept_b": [
        "monolithic architecture", "NoSQL databases", "GraphQL", "asynchronous programming",
        "functional programming", "Docker Swarm", "stream processing",
    ],
    "data_structure": [
        "a red-black tree", "a trie (prefix tree)", "a graph with adjacency list",
        "a priority queue using a heap", "a concurrent hash map",
    ],
    "operations": [
        "insertion, deletion, search, and traversal",
        "push, pop, peek, and iteration",
        "BFS, DFS, shortest path, and cycle detection",
    ],
    "domain": [
        "cybersecurity", "healthcare", "financial markets", "education",
        "autonomous vehicles", "natural language processing",
    ],
    "system_type": ["microservices", "event-driven", "serverless", "real-time"],
    "use_case": [
        "an e-commerce platform", "a real-time chat application",
        "a video streaming service", "an IoT monitoring dashboard",
    ],
    "formula": [
        "Black-Scholes option pricing", "Bayes' theorem",
        "the gradient of cross-entropy loss", "eigenvalue decomposition",
    ],
    "buggy_code": [
        "def factorial(n): return n * factorial(n-1)",
        "def find_max(lst): return max(lst) if lst else 0",
        "async def fetch(url): return requests.get(url)",
    ],
    "option_a": ["PostgreSQL", "REST", "monolith", "AWS Lambda"],
    "option_b": ["MongoDB", "gRPC", "microservices", "EC2 containers"],
    "scenario": [
        "a high-traffic web application", "a data analytics pipeline",
        "a real-time trading system", "a social media platform",
    ],
    "pattern": ["Observer", "Strategy", "Factory Method", "Command", "Decorator"],
    "topic_a": ["Docker", "TCP", "supervised learning", "SQL indexes"],
    "topic_b": ["Kubernetes", "UDP", "unsupervised learning", "NoSQL"],
    "ml_task": [
        "logistic regression", "decision tree classifier",
        "k-means clustering", "a simple neural network",
    ],
    "math_problem": [
        "Find the eigenvalues of [[2,1],[1,3]]",
        "Prove that the sum of first n odd numbers is n^2",
        "Solve the recurrence T(n) = 2T(n/2) + n",
    ],
    "phenomenon": [
        "transformers outperform RNNs for long-range dependencies",
        "gradient vanishing occurs in deep networks",
        "hash collisions affect performance",
        "cache invalidation is considered hard in computer science",
    ],
}


def _fill_template(template: str, fills: dict[str, list[str]]) -> str:
    """Fill a template with random values from the fills dict."""
    result = template
    for key, values in fills.items():
        placeholder = "{" + key + "}"
        if placeholder in result:
            result = result.replace(placeholder, random.choice(values), 1)
    return result


def generate_dataset(
    n_simple: int = 2500,
    n_complex: int = 2500,
    seed: int = 42,
) -> list[dict[str, str | int]]:
    """Generate a labeled dataset of simple and complex prompts."""
    random.seed(seed)
    dataset = []

    for _ in range(n_simple):
        template = random.choice(SIMPLE_TEMPLATES)
        prompt = _fill_template(template, SIMPLE_FILLS)
        dataset.append({"prompt": prompt, "label": 0, "category": "simple"})

    for _ in range(n_complex):
        template = random.choice(COMPLEX_TEMPLATES)
        prompt = _fill_template(template, COMPLEX_FILLS)
        dataset.append({"prompt": prompt, "label": 1, "category": "complex"})

    random.shuffle(dataset)
    return dataset


def save_dataset(dataset: list[dict], output_path: Path) -> None:
    """Save dataset to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "label", "category"])
        writer.writeheader()
        writer.writerows(dataset)
    print(f"Saved {len(dataset)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate router training dataset")
    parser.add_argument(
        "--output", type=str,
        default="training/data/router_dataset.csv",
        help="Output CSV path",
    )
    parser.add_argument("--n-simple", type=int, default=2500)
    parser.add_argument("--n-complex", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = generate_dataset(args.n_simple, args.n_complex, args.seed)
    save_dataset(dataset, Path(args.output))

    # Print stats
    simple = sum(1 for d in dataset if d["label"] == 0)
    complex_ = sum(1 for d in dataset if d["label"] == 1)
    print(f"Simple: {simple}, Complex: {complex_}")


if __name__ == "__main__":
    main()
