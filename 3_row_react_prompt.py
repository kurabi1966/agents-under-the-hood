import re
import inspect

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
    if float(price) < 0:
        raise ValueError("Price must be nonnegative")
    if discount_tier not in DISCOUNT_PERCENTAGES:
        raise ValueError(f"Unknown discount tier: {discount_tier!r}")
    percentage = DISCOUNT_PERCENTAGES[discount_tier]
    return round(float(price) * (1 - percentage / 100), 2)

tools = {
  "get_product_price": get_product_price,
  "apply_discount": apply_discount,  
}

def get_tool_descriptions(tools_dict):
  descriptions = []
  for tool_name, tool_function in tools_dict.items():
    original_function = getattr(tool_function, "__wrapped__", tool_function)
    signature = inspect.signature(original_function)
    docstring = inspect.getdoc(tool_function) or ""
    descriptions.append(f"{tool_name}{signature} - {docstring}")
  
  #  Sample return
  # get_product_price(product: str) -> float - Look up a catalog product's USD price; reject unknown products.
  # apply_discount(price: float, discount_tier: str) -> float - Apply a bronze, silver, or gold discount to a nonnegative price.
  return "\n".join(descriptions)
    
tool_descriptions = get_tool_descriptions(tools)

tool_names = ", ".join(tools.keys())    

react_prompt = f"""
STRICT RULES - you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real proce.
2. Only call apply_discount AFTER you have received a proce from get_product_price. Pass the exact price returned by get_product_price - do NOT assume one.
3. NEVER calculate discount yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use - do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action input: the input to the action as comma seperated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:
"""


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(model, messages, options):
  return ollama.chat(model=model, messages=messages, options=options)


@traceable(name="Native Ollama Agent Loop", run_type="chain")
def run_agent(question: str) -> str | None:
    print(f">> Executing run_agent(question={question!r})")
    print("=" * 60)

    prompt = react_prompt.format(question=question)
    scratchpad = ""
    
    for iteration in range(1, MAX_ITERATIONS + 1):
      print(f"\n--- Iteration {iteration} ---")
      full_prompt = prompt + scratchpad
      print(f"\nfull_prompt:\n{full_prompt}")
      output = ollama_chat_traced(
        model=MODEL, 
        messages=[{"role": "user", "content": full_prompt}],
        options={"stop": ["\nObservation"], "temperature": TEMPERATURE}
      ).message.content
      
      print(f"  [Parsing] Looking for final Answer in LLM output...")
      final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
      if final_answer_match:
        final_answer = final_answer_match.group(1).strip()
        print(f"Final Answer: {final_answer}")
        return final_answer
        
      # Not yet... we do not have a final answer. Let us see what tool to use
      print(f"  [Parsing] Loking for Action and Action Input in LLM output....")
      action_match =  re.search(r"Action:\s*(.+)", output)
      action_input_match =  re.search(r"Action input:\s*(.+)", output)
      
      if not action_match or not action_input_match:
        print(" [Parsing] ERROR: Could not parse Action/Action Input from LLM output")
        break
      
      tool_name = action_match.group(1).strip()
      tool_input_raw = action_input_match.group(1).strip()
      
      # do we have to call the tools manually?
      raw_args = [x.strip() for x in tool_input_raw.split(",")]
      args = [x.split("=",1)[-1].strip().strip("'\"") for x in raw_args]
      print(f"  [Tool Selected] {tool_name} with args: {args}")
      #  We have the tool name and the tool args... How to call it?

      
      if tool_name not in tools:
        observation = f"Error: Tool '{tool_name}' not found. Available tools: {list[str](tools.key())}"
      else:
        observation = str(tools[tool_name](*args)) # << Calling the tool     
         
      print(f"  [Tool Result] {observation}")
      
      scratchpad += f"{output}\nObservation: {observation}\nThought:"
      
    print("Error: Maximum iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello Native Ollama Agent (tool calling)")
    print()
    # print(get_tool_descriptions(tools_dict=tools))
    result = run_agent("What is the price of a laptop after applying a gold discount?")
    # result = run_agent("ما هو سعر الحاسوب المحمول بعد تطبيق خصم ذهبي؟") # it works (Amazing)
    # result = run_agent("¿Cuál es el precio de un portátil tras aplicar un descuento oro?") # it works (Amazing)