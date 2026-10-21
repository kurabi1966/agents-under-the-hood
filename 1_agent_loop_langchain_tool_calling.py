from dotenv import load_dotenv

load_dotenv()

#  ================

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATION = 10

MODEL = "qwen3:1.7b"
PROVIDER = "ollama"
# MODEL = "gpt-5"
# PROVIDER = "openai"
TEMPERATURE=0

#  ---- Tools (Langchain @toll decorator) -----


@tool
def get_product_price(product: str) -> float:
    """
    Look up the price of a product in the product catalog.

    Use this tool when a user asks about the price of a
    specific product or when the product price is needed
    for further calculations, such as applying discounts.

    Args:
      product (str):
        The name of the product to look up.
        Supported products:
        - "laptop"
        - "headphones"
        - "keyboard"

        Product names are case-insensitive.

    Returns:
      float:
        The product price in USD.
        Returns 0.0 if the product is not found
        in the catalog.
    """
    print(f"  >> Executing get_product_price(product='{product}')")
    # mock data
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Calculate the final price after applying a discount
    based on the specified discount tier.

    Use this tool when a user requests a discounted price
    or wants to calculate the cost after a discount.

    Args:
      price (float):
          The original price before applying the discount.
          Must be a non-negative number.

      discount_tier (str):
          The discount category to apply.
          Supported tiers:
          - "bronze": 5% discount
          - "silver": 12% discount
          - "gold": 23% discount

    Returns:
      float:
          The final price after applying the discount.
    """
    print(
        f"  >> Executing apply_discount(price={price}, discount_tire='{discount_tier}')"
    )
    discount_percentage = {"bronze": 5, "silver": 12, "gold": 23}
    return round(price - (price * discount_percentage.get(discount_tier, 0) / 100), 2)

@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    """
    Execute the AI agent to process a user question.

    Use this function to submit a natural language request
    to the AI agent. The agent will analyze the question,
    determine whether any registered tools are required,
    execute the necessary tools, and generate a response.

    Args:
        question (str):
            The user's natural language question or request.
            Examples:
            - "What is the price of a laptop?"
            - "Apply a Gold discount to a keyboard."
            - "What is the final price of headphones
                with a Platinum discount?"

    Returns:
        The agent's execution result, including its
        generated response and message history.
        The exact structure depends on the agent implementation.
    """
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}
    llm = init_chat_model(f"{PROVIDER}:{MODEL}", temperature=TEMPERATURE)
    llm_with_tolls = llm.bind_tools(tools)
    print(f">> Executing run_agent(question='{question}')")
    print("=" * 60)
    messages = [
      SystemMessage(
        content=(
          "You are a helpful shoping assistant. "
          "You have access to a product catalog tool "
          "and a discount tool.\n\n"
          "STRICT RULES - you must follow these exactly\n"
          "1. NEVER guess or assume any product price. "
          "You MUST call get_product_price first to get the real price.\n"
          "2. Only call apply discount AFTER you have recieved "
          "a price from get_product_price. Pass the exact price "
          "returned by get_product_price - do NOT pass a made-up number.\n "
          "3. NEVER calculate discounts yourself using math. "
          "Always use the apply_discount toll.\n"
          "4. If the user does not specify a discount tier, "
          "ask them which tier to use - do NOT assume one."
        )
      ),
      HumanMessage(
        content=(
          question
        )
      ),
    ]
    for iteration in range(1,MAX_ITERATION + 1):
      print(f"\n--- Iteration {iteration} ---")
      
      ai_message = llm_with_tolls.invoke(messages)
      
      messages.append(ai_message)
      
      
      tool_calls = ai_message.tool_calls
      
      # if no toll calls, then this is the final answer
      if not tool_calls:
        print(f"\nFinal Answer: {ai_message.content}")
        return ai_message.content
      
      for tool_call in tool_calls:
        tool_name = tool_call["name"]  
        tool_args = tool_call["args"]  
        tool_call_id = tool_call["id"]  

      tool_to_use = tools_dict.get(tool_name)
      if tool_to_use is None:
        raise ValueError(f"Tool '{tool_name} not found")
      
      print(f"  [Tool Selected] {tool_name} with args: {tool_args}")
      
      # Call the tool
      observation = tool_to_use.invoke(tool_args)
      print(f"  [Tool Result] {observation}")
      # Do we need to combine the observation to the existing messages? Yes
      messages.append(
        ToolMessage(content=str(observation), tool_call_id=tool_call_id) # ??? Why we are passing the tool_call_id? For tracing
      )
      
    # exiting the loop with no answer
    print("Error: Max iteration reached without a final answer")
    return None
      
      
      
      
    


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")
