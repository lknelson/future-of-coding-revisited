import ollama

coding_memo_1_dir_txt = '../data/coding_memo_main.txt'

with open(coding_memo_1_dir_txt, 'r') as file:
    memo_1 = file.read()

print(len(memo_1)) #32,766 character 

prompt = """ parse the given document and create bullet points of what definition of inequality articles is
"""

output = ollama.chat(
    model="llama3.1:70b",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        
        {
            "role": "user",
            "content": "Read the following document: " + memo_1
        },
        {
            "role": "user",
            "content": prompt
        },

    ],
      options={
        "seed": 101,
        "temperature": 0,
        "num_ctx": 128200,
        "num_predict": 1000,
        #"num_threads": 10
      }

)

irrelevant_definition = output["message"]["content"]

irrelevant_definition_path = '../data/inequality_llama3_1.txt'

with open(irrelevant_definition_path, 'w') as file:
    file.write(irrelevant_definition)
    
coding_memo_2_dir_txt = '../data/coding_memo_extra.txt'

with open(coding_memo_2_dir_txt, 'r') as file:
    memo_2 = file.read()

print(len(memo_2)) #32,766 character 

output_2 = ollama.chat(
    model="llama3.1:70b",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        
        {
            "role": "user",
            "content": "Read the following document: " + memo_2
        },
        {
            "role": "user",
            "content": prompt
        },

    ],
    
    options={
        "seed": 101,
        "temperature": 0,
        "num_ctx": 128200,
        "num_predict": 1000,
        #"num_threads": 10
    }
)

relevant_definition_path = '../data/inequality-2_llama3_1.txt'

with open(relevant_definition_path, 'w') as file:
    file.write(output_2["message"]["content"])
