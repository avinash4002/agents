import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
import google.generativeai as genai
from langchain.tools import tool
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
genai.configure(api_key=os.getenv("Gemini_api_key"))
from crewai import LLM

llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.7,
)
# Ask user for topic
topic = input("Enter the topic: ")

# Create Agents
researcher = Agent(
    role="Researcher",
    goal=f"Find detailed information about the topic: {topic}",
    backstory=(
        "You are a researcher. You will be given a topic and need to find factual and detailed information. "
        "You must ensure the legitimacy and usefulness of the information so the writer can create a well-reasoned and diplomatic essay."
    ),
    llm=llm,
    verbose=True
)

writer = Agent(
    role="Writer",
    goal=f"Write a detailed UPSC-style essay on {topic} based on the researcher's input.",
    backstory=(
        "You are a professional UPSC CSE essay writer. You critically analyze the research provided and write "
        "a well-structured, diplomatic essay with reasoning, facts, and a clear format (introduction, body, conclusion)."
    ),
    llm=llm,
    verbose=True
)
editor = Agent(
    role="Editor",
    goal=f"Edit the essay written by the writer on {topic}.",
    backstory=(
        "You are an editor. You will review the essay for clarity, coherence, and correctness. "
        "Make sure the essay is well-structured and free of errors."
        "Maintain the eassy in the upse format."
    ),
    llm=llm,
    verbose=True
)

# Create Tasks
researcher_task = Task(
    description=f"Research and gather detailed information on the topic: {topic}.",
    expected_output=f"A detailed summary and bullet points covering all aspects of the topic: {topic}.",
    agent=researcher
)

writing_task = Task(
    description=f"Using the researcher's findings, write a critical, factual, and diplomatic UPSC-style essay on: {topic}.",
    expected_output="A 500-700 word essay with proper structure, argumentation, and examples.",
    agent=writer,
    depends_on=[researcher_task]
)
editing_task = Task(
    description=f"Edit the essay written by the writer on: {topic}.",
    expected_output="A polished and refined version of the essay, ready for submission.",
    agent=editor,
    depends_on=[writing_task]
)
# Create and run Crew
crew = Crew(
    agents=[researcher, writer,editor],
    tasks=[researcher_task, writing_task,editing_task],
    verbose=True
)

# Run the workflow
result = crew.kickoff()
print("\n\n✅ FINAL ESSAY OUTPUT:\n")
print(result)