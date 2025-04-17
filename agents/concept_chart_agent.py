import os
import matplotlib.pyplot as plt
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import tempfile

# Load environment variables
load_dotenv()

# Prompt for Concept Mapping & Data Agent
concept_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""
You are a concept mapping and data assistant.

Your job is to create a structured, multi-level outline of the topic: "{topic}"

Provide the following in markdown format:

1. *High-level overview* (2-3 sentences)
2. *Mind map style bullet points* with subpoints (nested bullets)
3. *Key facts or data points* (dates, numbers, definitions)
4. *Applications or real-world use cases*
5. *Related concepts or fields*
6. *Recent trends or news (if relevant)*

Respond clearly using markdown formatting.
"""
)

# Prompt for chart instruction parsing
chart_prompt = PromptTemplate(
    input_variables=["query"],
    template="""
You are a visualization planner.

Extract or assume a small synthetic dataset (2-5 data points max) to visualize the user's request.

Respond with:
- The chart type (e.g., bar, line, pie)
- The chart title
- A dictionary of data (labels as keys, values as numbers)
- Optional x-axis and y-axis labels

Query: {query}
Response format:
Chart Type: ...
Title: ...
X Label: ...
Y Label: ...
Data: {{"Label1": number, "Label2": number, ...}}
"""
)

def initialize_agent():
    llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.4)
    concept_chain = LLMChain(llm=llm, prompt=concept_prompt)
    chart_chain = LLMChain(llm=llm, prompt=chart_prompt)
    return concept_chain, chart_chain

def generate_plot(chart_type, title, x_label, y_label, data_dict):
    fig, ax = plt.subplots()

    labels = list(data_dict.keys())
    values = list(data_dict.values())

    if chart_type.lower() == "bar":
        ax.bar(labels, values)
    elif chart_type.lower() == "line":
        ax.plot(labels, values, marker='o')
    elif chart_type.lower() == "pie":
        ax.pie(values, labels=labels, autopct='%1.1f%%')
        plt.axis('equal')

    ax.set_title(title)
    if chart_type.lower() != "pie":
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    # Save temp plot
    tmp_path = tempfile.mktemp(suffix=".png")
    plt.tight_layout()
    plt.savefig(tmp_path)
    plt.close()
    return tmp_path

def process_query(query):
    concept_chain, chart_chain = initialize_agent()
    query = query.lower()

    if any(word in query for word in ["plot", "chart", "visualize", "compare", "graph"]):
        try:
            plan = chart_chain.run(query)
            
            # Parse the chart generation instruction
            chart_type = plan.split("Chart Type:")[1].split("\n")[0].strip()
            title = plan.split("Title:")[1].split("\n")[0].strip()
            x_label = plan.split("X Label:")[1].split("\n")[0].strip()
            y_label = plan.split("Y Label:")[1].split("\n")[0].strip()
            data_line = plan.split("Data:")[1].strip()
            
            data_dict = eval(data_line)
            
            img_path = generate_plot(chart_type, title, x_label, y_label, data_dict)
            
            return {
                "type": "chart",
                "title": title,
                "image_path": img_path,
                "message": f"{title}\n\n_(Generated chart based on your query)_"
            }
            
        except Exception as e:
            # Fallback to concept mapping if chart generation fails
            result = concept_chain.run(query)
            return {
                "type": "text",
                "message": f"⚠ Error generating plot: {str(e)}\n\nFallbacking to concept mapping:\n\n{result}"
            }
    else:
        # Default: run concept/data agent
        result = concept_chain.run(query)
        return {
            "type": "text",
            "message": result
        } 