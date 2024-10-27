import subprocess
import os
from typing import Any, Dict, Optional
from langchain_core.tools import BaseTool, Tool
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.callbacks import CallbackManager
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st
from langchain_community.callbacks import StreamlitCallbackHandler

# Initialize Claude
os.environ["ANTHROPIC_API_KEY"] = " "
llm = ChatAnthropic(
    model="claude-2.1",
    temperature=0,
    verbose=True
)

import re

# Replace duplication of "/" of api url
def replace_duplicate_pattern(string: str, pattern: str) -> str:
    regex = re.compile(re.escape(pattern) + r'(?:/[^/]*)?' + re.escape(pattern))
    replaced_string = re.sub(regex, pattern, string)
    return replaced_string

class BashTool(BaseTool):
    name: str = "bashTool"
    description: str = "Run commands in Windows PowerShell or Bash and return the result"
    
    def _run(self, command: str) -> str:
        # Clean up the command
        command = command.replace('"', '')
        command = command.replace('`', '')
        
        try:
            # For Windows PowerShell
            if os.name == 'nt':
                # Using PowerShell explicitly
                powershell_command = ['powershell', '-Command', command]
                result = subprocess.run(
                    powershell_command,
                    capture_output=True,
                    text=True,
                    shell=True
                )
                # Combine stdout and stderr
                output = result.stdout + result.stderr
                return output if output else "Command executed successfully (no output)"
            
            # For Unix-like systems
            else:
                result = subprocess.run(
                    command,
                    shell=True,
                    check=False,  # Don't raise exception on non-zero exit
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                # Combine stdout and stderr
                output = result.stdout + result.stderr
                return output if output else "Command executed successfully (no output)"
                
        except Exception as e:
            return f"Error executing command: {str(e)}"
    
    async def _arun(self, command: str) -> str:
        raise NotImplementedError("BashTool does not support async")

class NetboxTool(BaseTool):
    name: str = "netboxTool"
    description: str = """Execute Netbox REST API queries to get information about network infrastructure.
    
    Common API endpoints:
    - /api/dcim/devices/ - List all devices
    - /api/dcim/devices/{id}/ - Get specific device details
    - /api/ipam/ip-addresses/ - List all IP addresses
    - /api/ipam/prefixes/ - List all prefix ranges
    - /api/virtualization/virtual-machines/ - List all VMs
    - /api/dcim/interfaces/ - List all interfaces
    - /api/dcim/sites/ - List all sites
    - /api/dcim/racks/ - List all racks
    
    Filtering examples:
    - /api/dcim/devices/?name=server1 - Filter devices by name
    - /api/dcim/devices/?status=active - Filter by status
    - /api/ipam/ip-addresses/?address=192.168.1.1 - Filter IPs
    - /api/dcim/devices/?site=datacenter1 - Filter by site
    """
    
    def _run(self, input: str) -> str:
        import requests
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': 'Token  '
        }
        input = input.replace('\n', '')
        if input.startswith('/'):
            input = replace_duplicate_pattern(input, '/api')
        else:
            input = '/api/' + input.lstrip('/')
            
        url = "https://demo.netbox.dev" + input
        print(f'Requesting URL: {url}')
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return f"Error: {str(e)}\nStatus code: {response.status_code if 'response' in locals() else 'N/A'}"
    
    async def _arun(self, input: str) -> str:
        raise NotImplementedError("NetboxTool does not support async")

# Initialize tools
tools = [
    BashTool(),
    NetboxTool()
]

# ReAct Prompt Template modified for Claude with memory
template = '''
You are Claude, a helpful AI assistant created by Anthropic. You aim to be direct, helpful, and honest in your interactions.

You are designed to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on various topics. You can generate coherent and relevant responses based on the input you receive, allowing you to engage in natural conversations.

You have access to tools that can help you complete tasks. Use these tools when necessary to provide accurate and helpful responses.

Previous conversation history:
{chat_history}

TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes

Action: the action to take, should be one of [{tool_names}]

Action Input: the input to the action

Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No

Final Answer: [your response here]
```

Begin!

New input: {input}

{agent_scratchpad}
'''

def init_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def format_chat_history(messages):
    """Format chat history for the prompt"""
    formatted_history = []
    for msg in messages:
        if msg["role"] == "user":
            formatted_history.append(f"Human: {msg['content']}")
        else:
            formatted_history.append(f"Assistant: {msg['content']}")
    return "\n".join(formatted_history)

def main():
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT9yb5GUIMG2TCVoFuxEiU3nKix5CEwz8OYdYGihojWpj3RhwJymOA_ZYmSBhPK1OhSfg&usqp=CAU")
    st.title("Netbox Assistant ")
    
    # Initialize session state
    init_session_state()
    
    # Create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create PromptTemplate
    prompt = PromptTemplate.from_template(template)
    
    # Create React Agent with memory
    agent = create_react_agent(llm, tools, prompt)
    
    # Create AgentExecutor with memory
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What can I help you with?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append(f"Human: {prompt}")
        
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.chat_message("assistant"):
            streamlit_callback = StreamlitCallbackHandler(st.container())
            
            # Include chat history in the input
            response = agent_executor.invoke(
                {
                    "input": prompt,
                    "chat_history": format_chat_history(st.session_state.messages[-5:])  # Last 5 messages
                },
                callbacks=[streamlit_callback]
            )
            
            assistant_response = response["output"]
            st.markdown(assistant_response)
            
            # Update chat history
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            st.session_state.chat_history.append(f"Assistant: {assistant_response}")

            # Keep only last 10 messages in memory
            if len(st.session_state.messages) > 10:
                st.session_state.messages = st.session_state.messages[-10:]
                st.session_state.chat_history = st.session_state.chat_history[-10:]

if __name__ == "__main__":
    main()