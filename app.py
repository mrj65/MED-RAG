import faiss
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
import json
import gradio as gr
import matplotlib.pyplot as plt
import tempfile
import os
import subprocess

class MedicalRAG:
    def __init__(self, embed_path, pmids_path, content_path):
        self.download_files()  
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load data
        self.embeddings = np.load(embed_path)
        self.index = self._create_faiss_index(self.embeddings)
        self.pmids, self.content = self._load_json_files(pmids_path, content_path)
        # Setup models
        self.encoder, self.tokenizer = self._setup_encoder()
        self.generator = self._setup_generator()
    def download_files(self):
        urls = [
            "https://ftp.ncbi.nlm.nih.gov/pub/lu/MedCPT/pubmed_embeddings/embeds_chunk_36.npy",
            "https://ftp.ncbi.nlm.nih.gov/pub/lu/MedCPT/pubmed_embeddings/pmids_chunk_36.json",
            "https://ftp.ncbi.nlm.nih.gov/pub/lu/MedCPT/pubmed_embeddings/pubmed_chunk_36.json"
        ]
        for url in urls:
            file_name = url.split('/')[-1]
            if not os.path.exists(file_name):
                print(f"Downloading {file_name}...")
                subprocess.run(["wget", url], check=True)
            else:
                print(f"{file_name} already exists. Skipping download.")

    def _create_faiss_index(self, embeddings):
        index = faiss.IndexFlatIP(768)  # 768 is embedding dimension
        index.add(embeddings)
        return index

    def _load_json_files(self, pmids_path, content_path):
        with open(pmids_path) as f:
            pmids = json.load(f)
        with open(content_path) as f:
            content = json.load(f)
        return pmids, content

    def _setup_encoder(self):
        model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(self.device)
        tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
        return model, tokenizer

    def _setup_generator(self):
        return pipeline(
            "text-generation",
            #model="HuggingFaceTB/SmolLM2-1.7B-Instruct",
            model = "HuggingFaceTB/SmolLM2-360M-Instruct",
            device=self.device,
            torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
        )

    def encode_query(self, query):
        with torch.no_grad():
            inputs = self.tokenizer([query], truncation=True, padding=True, 
                                  return_tensors='pt', max_length=64).to(self.device)
            embeddings = self.encoder(**inputs).last_hidden_state[:, 0, :]
            return embeddings.cpu().numpy()

    def search_documents(self, query_embedding, k=8):
        scores, indices = self.index.search(query_embedding, k=k)
        return [(self.pmids[idx], float(score)) for idx, score in zip(indices[0], scores[0])], indices[0]

    def get_document_content(self, pmid):
        doc = self.content.get(pmid, {})
        return {
            'title': doc.get('t', '').strip(),
            'date': doc.get('d', '').strip(),
            'abstract': doc.get('a', '').strip()
        }

    def visualize_embeddings(self, query_embed, relevant_indices, labels):
        plt.figure(figsize=(20, len(relevant_indices) + 1))
        
        # Prepare embeddings for visualization
        embeddings = np.vstack([query_embed[0], self.embeddings[relevant_indices]])
        normalized_embeddings = embeddings / np.max(np.abs(embeddings))
        # plt
        for idx, (embedding, label) in enumerate(zip(normalized_embeddings, labels)):
            y_pos = len(labels) - 1 - idx
            plt.imshow(embedding.reshape(1, -1), aspect='auto', extent=[0, 768, y_pos, y_pos+0.8],
                      cmap='inferno')
        
        # Add labels and styling
        plt.yticks(range(len(labels)), labels)
        plt.xlabel('Embedding Dimensions')
        plt.colorbar(label='Normalized Value')
        plt.title('Query and Retrieved Document Embeddings')
        
        # Save plot
        temp_path = os.path.join(tempfile.gettempdir(), f'embeddings_{hash(str(embeddings))}.png')
        plt.savefig(temp_path, bbox_inches='tight', dpi=150)
        plt.close()
        return temp_path

    def generate_answer(self, query, contexts):
        prompt = (
            "<|im_start|>system\n"
            "You are a helpful medical assistant. Answer questions based on the provided literature."
            "<|im_end|>\n<|im_start|>user\n"
            f"Based on these medical articles, answer this question:\n\n"
            f"Question: {query}\n\n"
            f"Relevant Literature:\n{contexts}\n"
            "<|im_end|>\n<|im_start|>assistant"
        )
        
        response = self.generator(
            prompt,
            max_new_tokens=200,
            temperature=0.3,
            top_p=0.95,
            do_sample=True
        )
        return response[0]['generated_text'].split("<|im_start|>assistant")[-1].strip()

    def process_query(self, query):
        try:
            # Encode and search
            query_embed = self.encode_query(query)
            doc_matches, indices = self.search_documents(query_embed)
            
            # Prepare documents and labels
            documents = []
            sources = []
            labels = ["Query"]
            
            for pmid, score in doc_matches:
                doc = self.get_document_content(pmid)
                if doc['abstract']:
                    documents.append(f"Title: {doc['title']}\nAbstract: {doc['abstract']}")
                    sources.append(f"PMID: {pmid}, Score: {score:.3f}, Link: https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
                    labels.append(f"Doc {len(labels)}: {doc['title'][:30]}...")

            
            # Generate outputs
            visualization = self.visualize_embeddings(query_embed, indices, labels)
            answer = self.generate_answer(query, "\n\n".join(documents[:3]))
            sources_text = "\n".join(sources)
            context = "\n\n".join(documents)
            
            return answer, sources_text, context, visualization
            
        except Exception as e:
            print(f"Error: {str(e)}")
            return str(e), "Error retrieving sources", "", None
def create_interface():
    rag = MedicalRAG(
        embed_path="embeds_chunk_36.npy",
        pmids_path="pmids_chunk_36.json",
        content_path="pubmed_chunk_36.json"
    )
    
    with gr.Blocks(title="Medical Literature QA") as interface:
        gr.Markdown("# Medical Literature Question Answering")
        with gr.Row():
            with gr.Column():
                query = gr.Textbox(lines=2, placeholder="Enter your medical question...", label="Question")
                submit = gr.Button("Submit", variant="primary")
                sources = gr.Textbox(label="Sources", lines=3)
                plot = gr.Image(label="Embedding Visualization")
            with gr.Column():
                answer = gr.Textbox(label="Answer", lines=5)
                context = gr.Textbox(label="Context", lines=6)      
        with gr.Row():
            gr.Examples(
                examples=[
                    ["What are the latest treatments for diabetes?"],
                    ["How effective are COVID-19 vaccines?"],
                    ["What are common symptoms of the flu?"],
                    ["How can I maintain good heart health?"]
                ],
                inputs=query
            )
        
        submit.click(
            fn=rag.process_query,
            inputs=query,
            outputs=[answer, sources, context, plot]
        )
    
    return interface

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)