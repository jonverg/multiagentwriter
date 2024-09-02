import os
from crewai import Agent, Task, Crew
from langchain_community.llms import OpenAI  
from langchain_community.tools import DuckDuckGoSearchRun
from crewai_tools import PDFSearchTool

# Initialize tools
search_tool = DuckDuckGoSearchRun()

def create_researcher_agent(search_tool, pdf_search_tool):
    researcher = Agent(
        role='Website and Document Researcher',
        goal='Scrape and gather relevant information from the {company} (in {location}) website and associated documents to advertise their products effectively.',
        backstory="""You are responsible for extracting detailed
        information from {company} website, including product descriptions,
        company history, and any customer testimonials. You will also use
        the PDFSearchTool to retrieve relevant content from documents associated
        with the company. This information will be used to support advertising content
        and will assist the Content Planner agent. Begin the review by stating your role.""",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool, pdf_search_tool]
    )
    return researcher


def create_planner_agent():
    planner = Agent(
        role="Content Planner",
        goal="Plan accurate and engaging content on the topic for a blog article advertising the company's products and services.",
        backstory="""You're tasked with creating a comprehensive content plan based on the
        information gathered by the researcher. Your plan will guide the content writer in
        crafting a compelling blog post that highlights the company's offerings.
        You also plan a blog article based on the topic: {topic}. You collect information
        that helps the audience learn something and make informed decisions.
        Your work is the basis for the Content Writer to write an article for this topic.
        Begin the review by stating your role.""",
        allow_delegation=False,
        verbose=True
    )
    return planner 
    
def create_writer_agent():
    writer = Agent(
        role="Content Writer",
        goal="Write an insightful and factually accurate opinion piece about the topic: {topic}, while advertising the company's products and services.",
        backstory="""You're working on writing a new blog post about the topic: {topic}. You base your writing on the work of the Content Planner, who provides an outline and relevant context about the topic. You follow the main objectives and direction of the outline, as provided by the Content Planner. You also provide objective and impartial insights and back them up with information provided by the Content Planner. You acknowledge in your opinion piece when your statements are opinions as opposed to objective statements. Begin the review by stating your role.""",
        allow_delegation=False,
        verbose=True
    )
    return writer

def create_legal_reviewer_agent():
    legal_reviewer = Agent(
        role="Legal Reviewer",
        goal="Review the content to ensure it adheres to legal standards.",
        backstory="""You are a legal reviewer, known for your ability to ensure that content is legally compliant and free from any potential legal issues. Make sure your suggestion is concise (within 3 bullet points), concrete and to the point. Begin the review by stating your role.""",
        allow_delegation=False,
        verbose=True
    )
    return legal_reviewer


def create_ethics_reviewer_agent():
    ethics_reviewer = Agent(
        role="Ethics Reviewer",
        goal="Review the content to ensure it adheres to ethical standards.",
        backstory="""You are an ethics reviewer, known for your ability to ensure that content is ethically sound and free from any potential ethical issues. Make sure your suggestion is concise (within 3 bullet points), concrete and to the point. Begin the review by stating your role.""",
        allow_delegation=False,
        verbose=True
    )
    return ethics_reviewer

def create_geo_optimizer_agent(pdf_search_tool):
        geo_optimizer = Agent(
            role="Generative Engine Optimization Specialist",
            goal="Optimize the content for keywords and ensure it maximizes reach and visibility within Generative AI Engines such as ChatGPT, Claude, SGE, Gemini, and Perplexity.",
            backstory="""Your role is to enhance the content's discoverability by optimizing it for relevant keywords and ensuring it performs well in search queries within Generative AI engines. Ensure the blogpost is still detailed. Overall, you focus on improving visibility when people inquire about the company’s products, services, and expertise. You achieve enhancing the content's discoverability by: incorporating unique words, keyword stuffing, technical terms, and using the PDFSearchTool to retrieve relevant statistics or quotations from documents related to the company. Begin the review by stating your role.""",
            allow_delegation=False,
            verbose=True,
            tools=[pdf_search_tool]
        )
        return geo_optimizer

# Define Tasks
def define_tasks(optimize_for, search_tool, pdf_search_tool):
    # Initialize the researcher agent with both search and PDF tools
    researcher = create_researcher_agent(search_tool, pdf_search_tool)
    planner = create_planner_agent()
    writer = create_writer_agent()
    legal_reviewer = create_legal_reviewer_agent()
    ethics_reviewer = create_ethics_reviewer_agent()
    
    # Task 1: Research about the company and products
    task1 = Task(
        description="""Perform research about the company theDevMasters in Irvine, California. Gather
        relevant information about their products, services, company's history,
        and any customer testimonials from both the company website and associated documents.
        Focus on extracting details that can be used to effectively advertise the company's offerings.
        Additionally, provide statistics about {topic}.""",
        expected_output="Detailed report of key information gathered from the company's website and documents, organized in bullet points.",
        agent=researcher
    )

    # Task 2: Plan content for the blog article
    task2 = Task(
        description="""Using the research provided, develop a comprehensive
        content plan for a blog article that advertises theDevMasters’ products
        and services while covering the topic: {topic}. Your plan should include
        a clear structure, key points to be discussed, and any relevant examples
        or data that should be included.""",
        expected_output="A detailed content plan outlining the blog structure, key points, and suggested content for each section.",
        agent=planner
    )

    # Task 3: Write the blog post
    task3 = Task(
        description="""Write a persuasive and factually accurate blog post based
        on the content plan provided by the Content Planner. The blog post should
        highlight the company’s products and services while thoroughly discussing
        the topic: {topic}. Ensure the tone is engaging, avoid complex language,
        and make the content accessible to a wide audience.""",
        expected_output="A full blog post of at least 4 paragraphs, adhering to the content plan and effectively advertising the company.",
        agent=writer
    )

    # Task 4: Review content for legal compliance
    task4 = Task(
        description="""Review the blog post to ensure it adheres to legal standards
        and is free from any potential legal issues. Focus on avoiding any false claims,
        ensuring copyright compliance, and making sure the content aligns with advertising standards.""",
        expected_output="A concise review of the blog post in 3 bullet points, outlining any legal concerns or confirming compliance.",
        agent=legal_reviewer
    )

    # Task 5: Review content for ethical standards
    task5 = Task(
        description="""Review the blog post to ensure it adheres to ethical standards.
        This includes checking for any misleading information, ensuring the content
        respects all cultural and social sensitivities, and confirming that the content
        does not exploit or harm any individuals or groups.""",
        expected_output="A concise review of the blog post in 3 bullet points, outlining any ethical concerns or confirming ethical soundness.",
        agent=ethics_reviewer
    )

    # Task 6: Conditional task based on user choice for SEO or GEO optimization
    if optimize_for == "GEO":
        geo_optimizer = create_geo_optimizer_agent(pdf_search_tool)
        task6 = Task(
            description="""Optimize the blog post for keywords and ensure that it
            maximizes reach and visibility within Generative AI Engines such as
            ChatGPT, Claude, SGE, Gemini, and Perplexity. Focus on enhancing the content's
            discoverability by incorporating unique words, keyword stuffing, ensuring it is easy to understand,
            has technical terms, and add quotations or statistics from the document.
            Overall, Rewrite the blog post with the optimized keywords and improvements.""",
            expected_output="A detailed and optimized blog post with at least 4 paragraphs designed to perform well in generative engine rankings, including keyword stuffing, statistics/quotations, and readability improvements.",
            agent=geo_optimizer
        )
    else:
        seo_optimizer = Agent(
            role="Search Engine Optimization Specialist",
            goal="Optimize the content for search engines like Google...",
            backstory="Your role is to enhance the content's discoverability in search engines...",
            verbose=True,
            allow_delegation=False
        )
        task6 = Task(
            description="""Optimize the blog post for search engines (SEO). Ensure that it includes relevant keywords, meta descriptions,
            alt texts for images, and is structured with headings and subheadings that align with SEO best practices. Improve readability
            and ensure that the content is highly engaging and accessible.""",
            expected_output="A detailed SEO-optimized blog post with at least 4 paragraphs that follow SEO guidelines for better search engine visibility.",
            agent=seo_optimizer
        )
    
    return [task1, task2, task3, task4, task5, task6]

def create_crew(optimize_for, search_tool, pdf_search_tool):
    tasks = define_tasks(optimize_for, search_tool, pdf_search_tool)
    
    # Initialize agents
    planner = create_planner_agent()
    writer = create_writer_agent()
    legal_reviewer = create_legal_reviewer_agent()
    ethics_reviewer = create_ethics_reviewer_agent()
    
    # Create the crew with the initialized agents
    crew = Crew(
        agents=[planner, writer, legal_reviewer, ethics_reviewer],
        tasks=tasks,
        verbose=True
    )
    return crew

def generate_blog_post(company, location, topic, optimize_for, search_tool, pdf_search_tool):
    """
    Function to generate a blog post using the multi-agent system.
    """
    crew = create_crew(optimize_for, search_tool, pdf_search_tool)
    return crew.kickoff(inputs={"company": company, "location": location, "topic": topic})