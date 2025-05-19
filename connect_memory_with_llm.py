import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import random
from colorama import init, Fore, Style

# Initialize colorama for colored console output
init()

# Step 1: Setup LLM (Mistral with HuggingFace)
# Check if token exists
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print(f"{Fore.RED}Error: HF_TOKEN environment variable not set{Style.RESET_ALL}")
    exit(1)

HUGGINGFACE_REPO_ID = "google/flan-t5-base"

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        token=HF_TOKEN,
        task="text-generation",
        temperature=0.5,
        max_length=512
    )
    return llm

# Step 2: Connect LLM with FAISS and Create chain
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
Don't provide anything outside the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk, please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt

# Step 3: Load Database
try:
    DB_FAISS_PATH = "vectorstore/db_faiss"
    print(f"{Fore.CYAN}Loading medical knowledge database...{Style.RESET_ALL}")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    print(f"{Fore.GREEN}✓ Medical knowledge database loaded successfully{Style.RESET_ALL}")
except Exception as e:
    print(f"{Fore.RED}Error loading database: {e}{Style.RESET_ALL}")
    exit(1)

# Step 4: Create QA chain
try:
    print(f"{Fore.CYAN}Initializing medical consultation system...{Style.RESET_ALL}")
    model = load_llm(HUGGINGFACE_REPO_ID)
    qa_chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
    )
    print(f"{Fore.GREEN}✓ Medical AI assistant ready{Style.RESET_ALL}")
except Exception as e:
    print(f"{Fore.RED}Error creating QA chain: {e}{Style.RESET_ALL}")
    exit(1)

# Medical-themed greeting messages
medical_greetings = [
    "Hello, I'm your virtual medical assistant. What symptoms are you experiencing today?",
    "Welcome to the medical consultation system. Please describe your health concern.",
    "I'm here to help with your medical questions. What would you like to know?",
    "Good day! I'm your medical AI assistant. How can I help with your health concerns?",
    "Medical assistant online. Please describe your symptoms or health question."
]

def print_welcome_message():
    print(f"""
{Fore.CYAN}╔═══════════════════════════════════════════════════════════╗
║                 {Fore.WHITE}Medical AI Assistant{Fore.CYAN}                      ║
╚═══════════════════════════════════════════════════════════╝{Style.RESET_ALL}

This assistant can answer your medical questions based on medical literature.
Type 'exit' or 'quit' at any time to end the consultation.
""")

def main():
    print_welcome_message()
    
    while True:
        # Choose a random greeting from our list
        greeting = random.choice(medical_greetings)
        user_query = input(f"{Fore.CYAN}{greeting}{Style.RESET_ALL}\n> ")
        
        # Check if user wants to exit
        if user_query.lower() in ['exit', 'quit', 'bye', 'goodbye']:
            print(f"\n{Fore.CYAN}Thank you for using the Medical AI Assistant. Take care of your health!{Style.RESET_ALL}")
            break
            
        # Process the query
        try:
            print(f"{Fore.YELLOW}Analyzing your health concern...{Style.RESET_ALL}")
            response = qa_chain.invoke({'query': user_query})
            
            print(f"\n{Fore.GREEN}Medical Recommendation:{Style.RESET_ALL}")
            print(f"{response['result']}")
            
            # Optionally uncomment to show sources
            # print(f"\n{Fore.BLUE}Sources:{Style.RESET_ALL}")
            # for i, doc in enumerate(response["source_documents"][:2]):
            #     print(f"{Fore.BLUE}Document {i+1}:{Style.RESET_ALL}\n{doc.page_content[:200]}...\n")
            
            print("\n" + "-" * 80 + "\n")
            
        except Exception as e:
            print(f"{Fore.RED}Error processing your query: {e}{Style.RESET_ALL}")
            print("Please try rephrasing your question or ask something else.")

if __name__ == "__main__":
    main()