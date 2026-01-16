from setuptools import setup, find_packages

setup(
    name='medical_chatbot',
    version='0.1.0',    
    author='Vidhya Hariharan',
    packages=find_packages(),
    install_requires=[
        'flask',
        'langchain',
        'langchain-pinecone',
        'langchain-openai',
        'langchain-community',
        'sentence-transformers',
        'pypdf',
        'python-dotenv'
    ]
)