import warnings
warnings.filterwarnings("ignore")

from crewai import Agent,Task,Crew,LLM
from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai
import os
genai.configure(api_key=os.getenv("Gemini_api_key"))
llm=LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.7,
)

plan_agent=Agent(
    role="Insurance recommender Representative ",
    goal="1)Be polite and friendly."
         "2)You are a customer service representative for an insurance company.",
    backstory=(
        "You are a customer service representative for an insurance company."
        "You are given a customer query:{query} and need to provide the best insurance plan recommendation. "
),
    allow_delegation=False,
    verbose=True,
    llm=llm
    
)
final_agent=Agent(
    role = "Insurance sales Representative",
    goal = "Be polite and friendly."
           "You are a customer service representative for an insurance company. "
           "your task is to sell the insurance plan recommended by the plan agent.",
    backstory=("you are the sales representative for the insurance company"
               "you will be very polite to the customer"
               "you will be given the insurance plan recommended by the plan agent"
                "and you will try to sell the insurance plan to the customer"
                "you should try to convince the customer to buy the insurance plan"
                "sell the plan which is most profitable for the company"
    ),
    llm=llm,
    verbose=True)

from crewai_tools import ScrapeWebsiteTool

scrape_rate=ScrapeWebsiteTool(
    website_url="https://www.forbes.com/advisor/health-insurance/how-much-does-health-insurance-cost/"
)

plan_task=Task(
    description=("{customer} reached to you with a insurance related query"
    "he needs a insurance plan on {query}"
    "you need to recommend him the best insurance plan"
    "you can use the scrape tool to get the latest rates of the insurance plans"
),
    expected_output="the best insurance plan for the customer"
                    "answer him in the simplest way possible",
    agent=plan_agent,
    tools=[scrape_rate]
)
final_task=Task(
    description="you are the sales representative for the insurance company"
                "you will be very polite to the customer"
                "you will be given the insurance plan recommended by the plan agent"
                "and you will try to sell the insurance plan to the customer"
                "you should try to convince the customer to buy the insurance plan"
                "sell the plan which is most profitable for the company",
    expected_output="the best insurance plan for the customer"
                "the plan should be profitable for the company"
                "try to manipulate the customer to buy the plan"
                "play with the words to make the customer buy the plan"
                "try to manipulate the customer by using his emotions and future fears",
    agent=final_agent,
)
crew=Crew(
    agents=[plan_agent,final_agent],
    tasks=[plan_task,final_task],   
    verbose=True,
    memory=True)
inputs={
    "customer":"chaman",
    "query":"health insurance"
}
results=crew.kickoff(inputs=inputs)
print("Plan Agent Result:") 
print(results)