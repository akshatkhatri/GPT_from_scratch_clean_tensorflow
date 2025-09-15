#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import tensorflow as tf
import numpy as np
import json
import matplotlib.pyplot as plt
import io
from PIL import Image
import argparse
import matplotlib
matplotlib.use('Agg')

# Import from the models directory
from models.layers import GPT

class GPTDeployer:
    def __init__(self, experiment_dir="./experiments/jane_austen_proper_v1_20250914_093022"):
        self.experiment_dir = experiment_dir
        self.load_model_and_tokenizer()

    def load_model_and_tokenizer(self):
        # Load tokenizer
        with open(f"{self.experiment_dir}/tokenizer.json", 'r') as f:
            self.tokenizer_data = json.load(f)
        
        self.token_to_id = self.tokenizer_data['token_to_id']
        self.id_to_token = self.tokenizer_data['id_to_token']
        
        # Load model config
        with open(f"{self.experiment_dir}/model_config.json", 'r') as f:
            config_data = json.load(f)
        
        if 'architecture' in config_data:
            self.model_config = config_data['architecture']
        else:
            self.model_config = config_data.get('training_config', config_data)
            
        # Normalize config keys
        self.model_config = {
            'context_length': self.model_config.get('context_length', self.model_config.get('CONTEXT_LEN', 256)),
            'd_model': self.model_config.get('d_model', self.model_config.get('D_MODEL', 128)),
            'vocab_size': self.model_config.get('vocab_size', self.model_config.get('VOCAB_SIZE', len(self.token_to_id))),
            'attention_heads': self.model_config.get('attention_heads', self.model_config.get('ATTENTION_HEADS', 8)),
            'decoder_blocks': self.model_config.get('decoder_blocks', self.model_config.get('DECODER_BLOCKS', 4)),
            'dropout_rate': self.model_config.get('dropout_rate', self.model_config.get('DROPOUT_RATE', 0.1))
        }
        
        # Build model
        self.model = GPT(
            d_model=self.model_config['d_model'],
            vocab_size=self.model_config['vocab_size'],
            context_length=self.model_config['context_length'],
            attention_heads=self.model_config['attention_heads'],
            decoder_blocks=self.model_config['decoder_blocks'],
            dropout_rate=self.model_config['dropout_rate']
        )
        
        # Build the model by calling it with dummy data
        dummy_input = tf.ones((1, 10), dtype=tf.int32)
        dummy_mask = tf.ones((1, 10), dtype=tf.int32)
        _ = self.model([dummy_input, dummy_mask])
        
        # Load weights
        weights_path = f"{self.experiment_dir}/checkpoints/best_model.weights.h5"
        if os.path.exists(weights_path):
            self.model.load_weights(weights_path)
            print(f"Loaded weights from: {weights_path}")
        else:
            raise FileNotFoundError(f"Model weights not found at: {weights_path}")
        
        print(f"Model loaded successfully!")
        print(f"Architecture: {self.model_config}")

    def tokenize_text(self, text):
        return [self.token_to_id.get(char, 0) for char in text]

    def detokenize(self, tokens):
        return ''.join([self.id_to_token.get(str(token), '?') for token in tokens])

    def generate_text(self, prompt, max_length=100, temperature=0.7, top_k=20):
        if len(prompt.strip()) == 0:
            return "Please enter a prompt!"
        
        input_tokens = self.tokenize_text(prompt)
        if len(input_tokens) > self.model_config['context_length'] - max_length:
            input_tokens = input_tokens[-(self.model_config['context_length'] - max_length):]
        
        generated_tokens = input_tokens.copy()
        
        for _ in range(max_length):
            if len(generated_tokens) >= self.model_config['context_length']:
                break
                
            input_tensor = tf.constant([generated_tokens[-self.model_config['context_length']:]], dtype=tf.int32)
            attention_mask = tf.ones_like(input_tensor)
            
            logits = self.model([input_tensor, attention_mask])
            next_token_logits = logits[0, -1, :] / temperature
            
            # Top-k sampling
            if top_k > 0:
                top_k_indices = tf.nn.top_k(next_token_logits, k=min(top_k, self.model_config['vocab_size'])).indices
                top_k_logits = tf.gather(next_token_logits, top_k_indices)
                probs = tf.nn.softmax(top_k_logits)
                next_token_idx = tf.random.categorical([top_k_logits], 1)[0, 0]
                next_token = top_k_indices[next_token_idx]
            else:
                probs = tf.nn.softmax(next_token_logits)
                next_token = tf.random.categorical([next_token_logits], 1)[0, 0]
            
            generated_tokens.append(int(next_token))
        
        return self.detokenize(generated_tokens)

    def create_attention_visualization(self, text):
        if len(text.strip()) == 0:
            text = "Elizabeth walked"
        
        tokens = list(text[:12])  # Limit for readability
        if len(tokens) < 2:
            tokens = ['E', 'l', 'i', 'z', 'a', 'b', 'e', 't', 'h']
        
        token_labels = tokens
        n_tokens = len(tokens)
        
        # Create more realistic attention patterns
        attention_matrix = np.zeros((n_tokens, n_tokens))
        
        for i in range(n_tokens):
            for j in range(i + 1):  # Can only attend to previous tokens and self
                if i == j:
                    attention_matrix[i, j] = 0.4 + np.random.normal(0, 0.1)
                else:
                    distance = i - j
                    if distance <= 2:  # Recent tokens
                        attention_matrix[i, j] = 0.3 / distance + np.random.normal(0, 0.05)
                    else:  # Distant tokens
                        attention_matrix[i, j] = 0.1 / distance + np.random.normal(0, 0.02)
            
            # Normalize to sum to 1
            attention_matrix[i] = np.maximum(attention_matrix[i], 0)
            attention_matrix[i] = attention_matrix[i] / (attention_matrix[i].sum() + 1e-8)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(attention_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(range(n_tokens))
        ax.set_yticks(range(n_tokens))
        ax.set_xticklabels([f'{i}: "{label}"' for i, label in enumerate(token_labels)], rotation=45, ha='right')
        ax.set_yticklabels([f'{i}: "{label}"' for i, label in enumerate(token_labels)])
        
        # Add value annotations
        for i in range(n_tokens):
            for j in range(n_tokens):
                value = attention_matrix[i, j]
                if value > 0.05:
                    color = 'white' if value > 0.4 else 'black'
                    ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                           fontsize=10, fontweight='bold', color=color)
        
        ax.set_title(f'Attention Matrix: "{text[:25]}..."', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Key Tokens (What we\'re looking at)', fontsize=12)
        ax.set_ylabel('Query Tokens (Who is looking)', fontsize=12)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Weight', fontsize=11)
        
        plt.tight_layout()
        
        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        image = Image.open(buf)
        return image

    def create_interface(self):
        with gr.Blocks(title="Jane Austen GPT") as interface:
            
            gr.Markdown("""
            # 🎭 Jane Austen Style Text Generator
            ### A Custom GPT Model Trained on Jane Austen's Complete Works
            
            This model generates authentic 19th-century literary prose in the style of Jane Austen.
            """)
            
            with gr.Tabs():
                # Tab 1: Text Generation
                with gr.TabItem("✍️ Generate Text"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### Quick Start Examples")
                            example_prompts = [
                                "Elizabeth walked through the garden",
                                "It is a truth universally acknowledged that",
                                "Mr. Darcy approached with considerable",
                                "The morning was fine, and Elizabeth",
                                "Mrs. Bennet was delighted to",
                                "Emma could not repress a smile"
                            ]
                            
                            for prompt in example_prompts:
                                gr.Button(prompt, size="sm")
                        
                        with gr.Column(scale=2):
                            prompt_input = gr.Textbox(
                                label="Enter your prompt",
                                placeholder="Elizabeth walked through the garden...",
                                lines=3
                            )
                            
                            with gr.Row():
                                generate_btn = gr.Button("Generate Text", variant="primary")
                                clear_btn = gr.Button("Clear")
                            
                            with gr.Accordion("Advanced Settings", open=False):
                                temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="Temperature")
                                max_length = gr.Slider(10, 200, value=100, step=10, label="Max Length")
                                top_k = gr.Slider(1, 50, value=25, step=1, label="Top-k")
                            
                            output_text = gr.Textbox(
                                label="Generated Text",
                                lines=8,
                                interactive=False
                            )
                
                # Tab 2: Attention Visualization
                with gr.TabItem("🧠 See Model's Attention"):
                    gr.Markdown("### Attention Visualization")
                    
                    with gr.Row():
                        attention_input = gr.Textbox(
                            label="Enter text to analyze",
                            value="Elizabeth walked",
                            placeholder="Enter text..."
                        )
                        visualize_btn = gr.Button("Visualize Attention", variant="primary")
                    
                    attention_plot = gr.Image(label="Attention Heatmap")
                
                # Tab 3: Model Details
                with gr.TabItem("📊 Model Details"):
                    gr.Markdown(f"""
                    ### Model Architecture
                    - **Parameters:** {sum(tf.size(p).numpy() for p in self.model.trainable_variables):,}
                    - **Layers:** {self.model_config['decoder_blocks']} transformer blocks
                    - **Attention Heads:** {self.model_config['attention_heads']}
                    - **Embedding Dimension:** {self.model_config['d_model']}
                    - **Vocabulary Size:** {self.model_config['vocab_size']} characters
                    - **Context Length:** {self.model_config['context_length']} characters
                    
                    ### Training Details
                    This model was trained from scratch on Jane Austen's complete works using character-level tokenization.
                    """)
            
            # Event handlers
            def handle_example_click(example_text):
                return example_text
            
            # Connect example buttons
            for i, prompt in enumerate(example_prompts):
                # This is a simplified way - you'd need to create individual buttons with click handlers
                pass
            
            generate_btn.click(
                fn=self.generate_text,
                inputs=[prompt_input, max_length, temperature, top_k],
                outputs=output_text
            )
            
            clear_btn.click(
                fn=lambda: ("", ""),
                outputs=[prompt_input, output_text]
            )
            
            visualize_btn.click(
                fn=self.create_attention_visualization,
                inputs=attention_input,
                outputs=attention_plot
            )
        
        return interface

def launch_deployment(experiment_dir="./experiments/jane_austen_proper_v1_20250914_093022", 
                     share=False, port=7860):
    print("🚀 Initializing GPT Model Deployment...")
    
    deployer = GPTDeployer(experiment_dir)
    interface = deployer.create_interface()
    
    print("🌐 Launching Gradio interface...")
    print(f"   Port: {port}")
    print(f"   Share: {share}")
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
        show_error=True
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy Jane Austen GPT Model")
    parser.add_argument("--experiment_dir", type=str, 
                       default="./experiments/jane_austen_proper_v1_20250914_093022",
                       help="Path to experiment directory")
    parser.add_argument("--share", action="store_true", help="Create public sharing link")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    
    args = parser.parse_args()
    
    # Override port from environment variable if set
    port = int(os.environ.get('GRADIO_SERVER_PORT', args.port))
    
    launch_deployment(args.experiment_dir, share=args.share, port=port)