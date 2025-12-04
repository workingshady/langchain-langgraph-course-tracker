import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from logger import log_error, log_header, log_success, log_warning
from nodes import app
from langsmith import traceable 
load_dotenv()

@traceable(name="interactive_react_session")
def main():
    log_header("🧠 Interactive ReAct Agent")
    log_success("Type 'quit' to exit. Agent ready!")

    while True:
        try:
            user_input = input("\n👤 You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                log_success("👋 Goodbye!")
                break

            if not user_input:
                continue

            # Run agent
            result = app.invoke({"messages": [HumanMessage(content=user_input)]})
            final_answer = result["messages"][-1].content

            print(f"\n🤖 Agent: {final_answer}")
            log_success("Ready for next question!")

        except KeyboardInterrupt:
            log_success("\n👋 Goodbye!")
            break
        except Exception as e:
            log_error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
