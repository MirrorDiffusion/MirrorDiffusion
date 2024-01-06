import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

auto_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl/")
model = T5ForConditionalGeneration.from_pretrained(
    "google/flan-t5-xl/", device_map="auto", torch_dtype=torch.float16
)
model = model
def generate_captions(input_prompt):
    input_ids = auto_tokenizer(input_prompt, return_tensors="pt").input_ids.to("cuda")

    outputs = model.generate(
        input_ids,
        temperature=0.8,
        num_return_sequences=200,
        do_sample=True, max_new_tokens=128, 
        top_k=10
    )
    return auto_tokenizer.batch_decode(outputs, skip_special_tokens=True)

 
source_concept =  "A picture of cat."#@param {type:"string"}
target_concept = "A picture of dog. ." #@param {type:"string"}
source_text = f"Provide a caption for images containing a {source_concept}. "
"The captions should be in English and should be no longer than 150 characters."

target_text = source_text.replace(source_concept, target_concept)



from diffusers import StableDiffusionPipeline 


pipeline = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
)
pipeline = pipeline.to("cuda")

def embed_captions(l_sentences, tokenizer, text_encoder, device="cuda"):
    with torch.no_grad():
        l_embeddings = []
        for sent in l_sentences:
            text_inputs = tokenizer(
                sent,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask=None)[0]
            l_embeddings.append(prompt_embeds)
    return torch.concatenate(l_embeddings, dim=0).mean(dim=0).unsqueeze(0)

tokenizer = pipeline.tokenizer
text_encoder = pipeline.text_encoder
with torch.no_grad():
  a =  None
  b = None
  for i in range(5):
    print(i)
    source_captions = generate_captions(source_text)
    target_captions = generate_captions(target_text)
    print("Source caption examples: \n")
    print(len(source_captions))
    print(source_captions[:5])
    
    print("\nTarget caption examples: \n")
    print(target_captions[:5])
    source_embeddings = embed_captions(source_captions, tokenizer, text_encoder).detach()
    target_embeddings = embed_captions(target_captions, tokenizer, text_encoder).detach()
    del source_captions,target_captions
    if a is None:
        a =source_embeddings.clone().detach() 
    else:
        a +=source_embeddings.clone().detach()
    if b is None:
        b = target_embeddings.clone().detach() 
    else:
        b += target_embeddings.clone().detach()
a /=5
b /=5   
torch.save(a,'./assets/embeddings_sd_1.4/cat.pt')
torch.save(b,'./assets/embeddings_sd_1.4/dog.pt')