import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openai")
from crewai import Agent,Task,Crew
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
topic=input("Enter the topic for the essay: ")
load_dotenv()
llm=ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7)

researcher=Agent(
    role = "Researcher",
    goal="find detailed information about the {topic}",
    backstory="1) You are a researcher. You will be given a topic and you need to find detailed information about it."
    "2)you will find the legit information which will help the writer to critically and diplomatically write about it.",
    llm=llm,
    verbose=True
)

writer=Agent(
    role = "writer",
    goal="write a detailed eassy on the {topic} based on the information provided by the researcher",
    backstory="1)You are a professional essay writer for the upse cse exams"
    "2)you will analysis the information gathered by the researcher and critically analyze it"
    "3)you will write an eassy which will be diplomatic and have proper resoning and the eassy format",
    llm=llm,    
    verbose=True
)
researcher_task=Task(
    description="find detailed information about the {topic}",
    expected_output="detailed information about the {topic}",
    agent=researcher,
    
    
)
writing_task=Task(
    description="write a detailed eassy on the {topic} based on the information provided by the researcher",
    expected_output="detailed eassy on the {topic}"
    "2)you will analysis the information gathered by the researcher and critically analyze it"
     "3) you will then write a critical and diplomatic eassy ",
    agent=writer,
    depends_on=[researcher_task])

crew=Crew(
    agents=[researcher,writer],
    tasks=[researcher_task,writing_task],
    verbose=True    
)

result=crew.kickoff()
print(result)