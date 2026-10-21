from dotenv import load_dotenv
import ollama
from langsmith import traceable

load_dotenv()

MAX_ITERATIONS = 10
MODEL = "granite4:3b"
TEMPERATURE = 0

PRODUCT_PRICES = {
    "laptop": 1299.99,
    "headphones": 149.95,
    "keyboard": 89.50,
}
DISCOUNT_PERCENTAGES = {
    "bronze": 5,
    "silver": 12,
    "gold": 23,
}


@traceable(run_type="tool", name="get_product_price")
def get_product_price(product: str) -> float:
    """Look up a catalog product's USD price; reject unknown products."""
    product = product.strip().lower()
    print(f"  >> Executing get_product_price(product={product!r})")
    if product not in PRODUCT_PRICES:
        raise ValueError(f"Unknown product: {product!r}")
    return PRODUCT_PRICES[product]


@traceable(run_type="tool", name="apply_discount")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a bronze, silver, or gold discount to a nonnegative price."""
    discount_tier = discount_tier.strip().lower()
    print(f"  >> Executing apply_discount(price={price}, discount_tier={discount_tier!r})")
    if price < 0:
        raise ValueError("Price must be nonnegative")
    if discount_tier not in DISCOUNT_PERCENTAGES:
        raise ValueError(f"Unknown discount tier: {discount_tier!r}")
    percentage = DISCOUNT_PERCENTAGES[discount_tier]
    return round(price * (1 - percentage / 100), 2)


TOOLS_FOR_LLM = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the actual price in USD of a catalog product. Call this before applying a discount.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "Product name: laptop, headphones, or keyboard.",
                        "enum": list(PRODUCT_PRICES),
                    }
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Calculate the final price after applying a discount to a previously retrieved catalog price.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {
                        "type": "number",
                        "description": "Exact price previously returned by get_product_price.",
                    },
                    "discount_tier": {
                        "type": "string",
                        "description": "Discount tier: bronze (5%), silver (12%), or gold (23%).",
                        "enum": list(DISCOUNT_PERCENTAGES),
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    """Send conversation messages and tool schemas to Ollama."""
    return ollama.chat(
        model=MODEL,
        messages=messages,
        tools=TOOLS_FOR_LLM,
        options={"temperature": TEMPERATURE},
    )


@traceable(name="Native Ollama Agent Loop", run_type="chain")
def run_agent(question: str) -> str | None:
    """Answer a shopping question using a manual Ollama tool-calling loop."""
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    prompt = (
        "You are a helpful shopping assistant with access to a product catalog "
        "and a discount calculator.\n"
        "Rules:\n"
        "1. Never invent product prices. Always call get_product_price to look them up.\n"
        "2. Only call apply_discount after get_product_price has returned a price. "
        "Use that exact price, not an estimated price.\n"
        "3. Never calculate discounts yourself; use apply_discount.\n"
        "4. If a discount is requested without a tier, ask which tier to use.\n"
        "5. If a tool reports an error, explain it rather than inventing a result."
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": question},
    ]

    print(f">> Executing run_agent(question={question!r})")
    print("=" * 60)

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        ai_message = ollama_chat_traced(messages=messages).message
        tool_calls = ai_message.tool_calls or []
        if not tool_calls:
            answer = ai_message.content or ""
            print(f"\nFinal Answer: {answer}")
            return answer

        messages.append(ai_message)
        tool_name = tool_calls[0].function.name
        tool_args = tool_calls[0].function.arguments
        print(f"  [Tool Selected] {tool_name}: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        
        if tool_to_use is None:
          observation = f"Error: Unknown tool {tool_name!r}"
        else:
          observation = tool_to_use(**tool_args)

        print(f"  [Tool Result] {observation}")
        messages.append(
            {
                "role": "tool",
                "tool_name": tool_name,
                "content": str(observation),
            }
        )          

    print("Error: Maximum iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello Native Ollama Agent (tool calling)")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")