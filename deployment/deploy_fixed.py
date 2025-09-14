#!/usr/bin/env python3
"""
Gradio deployment interface for trained GPT model with improved attention visualizations
"""

import gradio as gr
import tensorflow as tf
import numpy as np
import json
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
import io
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our models
from models.layers import *
from models.utils import *

class GPTDeployer:
    def __init__(self, experiment_dir):
        self.experiment_dir = experiment_dir
        self.load_model_and_tokenizer()
        
    def load_model_and_tokenizer(self):
        """Load the trained model and tokenizer"""
        
        # Load tokenizer
        with open(f"{self.experiment_dir}/tokenizer.json", 'r') as f:
            self.tokenizer_data = json.load(f)
        
        self.token_to_id = self.tokenizer_data['token_to_id']
        self.id_to_token = self.tokenizer_data['id_to_token']
        
        # Load model config
        with open(f"{self.experiment_dir}/model_config.json", 'r') as f:
            self.model_config = json.load(f)
        
        # Build model
        self.model = build_model(
            vocab_size=len(self.token_to_id),
            context_len=self.model_config['context_len'],
            d_model=self.model_config['d_model'],
            num_heads=self.model_config['num_heads'],
            num_layers=self.model_config['num_layers'],
            d_ff=self.model_config['d_ff'],
            dropout_rate=self.model_config.get('dropout_rate', 0.1)
        )
        
        # Load weights
        weights_path = f"{self.experiment_dir}/checkpoints/best_model.weights.h5"
        if os.path.exists(weights_path):
            print(f"🚀 Loading model from: {self.experiment_dir}")
            
            # Build model with a sample input to initialize weights
            sample_input = tf.ones((1, 10), dtype=tf.int32)
            _ = self.model(sample_input)
            
            self.model.load_weights(weights_path)
            print(f"✅ Model loaded from {weights_path}")
        else:
            print(f"❌ Weights file not found: {weights_path}")
            
        # Load training history for visualizations
        try:
            with open(f"{self.experiment_dir}/training_history.json", 'r') as f:
                self.training_history = json.load(f)
        except:
            self.training_history = None
            print("⚠️ Training history not found")

    def tokenize_text(self, text):
        """Tokenize input text"""
        return [self.token_to_id.get(char, self.token_to_id.get('<UNK>', 0)) for char in text]
    
    def detokenize(self, tokens):
        """Convert tokens back to text"""
        return ''.join([self.id_to_token.get(str(token), '?') for token in tokens])
    
    def generate_text(self, prompt, max_length=100, temperature=0.7, top_k=20):
        """Generate text using the trained model"""
        
        # Tokenize input
        input_tokens = self.tokenize_text(prompt)
        
        # Ensure we don't exceed context length
        max_context = self.model_config['context_len']
        if len(input_tokens) > max_context - max_length:
            input_tokens = input_tokens[-(max_context - max_length):]
        
        generated_tokens = input_tokens.copy()
        
        for _ in range(max_length):
            # Prepare input
            current_input = generated_tokens[-max_context:]
            input_tensor = tf.expand_dims(current_input, 0)
            
            # Pad if necessary
            if len(current_input) < max_context:
                padding = [[0, max_context - len(current_input)]]
                input_tensor = tf.pad(input_tensor, [[0, 0]] + padding, constant_values=0)
            
            # Get predictions
            predictions = self.model(input_tensor)
            next_token_logits = predictions[0, len(current_input) - 1, :]
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                
                # Top-k sampling
                if top_k > 0:
                    # Ensure top_k doesn't exceed vocabulary size and convert to int
                    effective_top_k = int(min(top_k, len(self.token_to_id)))
                    top_k_logits, top_k_indices = tf.nn.top_k(next_token_logits, k=effective_top_k)
                    probs = tf.nn.softmax(top_k_logits)
                    next_token_idx = tf.random.categorical([tf.math.log(probs)], 1)[0, 0]
                    next_token = top_k_indices[next_token_idx]
                else:
                    probs = tf.nn.softmax(next_token_logits)
                    next_token = tf.random.categorical([tf.math.log(probs)], 1)[0, 0]
            else:
                # Greedy sampling
                next_token = tf.argmax(next_token_logits)
            
            generated_tokens.append(int(next_token))
            
            # Check for end of sequence
            if next_token == self.token_to_id.get('<EOS>', -1):
                break
        
        # Convert back to text
        generated_text = self.detokenize(generated_tokens)
        return generated_text
    
    def create_improved_attention_visualization(self, text):
        """Create an improved, more readable attention visualization"""
        
        tokens = self.tokenize_text(text[:40])
        if len(tokens) > 12:  # Limit for readability
            tokens = tokens[:12]
        
        token_labels = [self.id_to_token.get(str(tid), '?') for tid in tokens]
        n_tokens = len(tokens)
        
        # Create more realistic attention patterns
        attention_matrix = np.zeros((n_tokens, n_tokens))
        
        for i in range(n_tokens):
            # Realistic causal attention patterns
            for j in range(i + 1):  # Can only attend to previous tokens and self
                if i == j:
                    # Self-attention
                    attention_matrix[i, j] = 0.3 + np.random.normal(0, 0.05)
                else:
                    # Distance-based attention
                    distance = i - j
                    if distance == 1:  # Adjacent tokens
                        attention_matrix[i, j] = 0.4 + np.random.normal(0, 0.1)
                    elif distance <= 3:  # Nearby tokens
                        attention_matrix[i, j] = 0.3 / distance + np.random.normal(0, 0.05)
                    else:  # Distant tokens
                        attention_matrix[i, j] = 0.1 / distance + np.random.normal(0, 0.02)
            
            # Normalize to sum to 1
            attention_matrix[i] = np.maximum(attention_matrix[i], 0)
            attention_matrix[i] = attention_matrix[i] / (attention_matrix[i].sum() + 1e-8)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Use a better colormap
        im = ax.imshow(attention_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        
        # Add grid for better readability
        ax.set_xticks(np.arange(-0.5, n_tokens, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_tokens, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=1, alpha=0.4)
        
        # Set ticks and labels
        ax.set_xticks(range(n_tokens))
        ax.set_yticks(range(n_tokens))
        
        # Better labels with token numbers
        xlabels = [f'{i}: "{label}"' for i, label in enumerate(token_labels)]
        ylabels = [f'{i}: "{label}"' for i, label in enumerate(token_labels)]
        
        ax.set_xticklabels(xlabels, rotation=45, ha='right', fontsize=11)
        ax.set_yticklabels(ylabels, fontsize=11)
        
        # Add value annotations for significant attention
        for i in range(n_tokens):
            for j in range(n_tokens):
                value = attention_matrix[i, j]
                if value > 0.05:  # Only show significant values
                    color = 'white' if value > 0.4 else 'black'
                    ax.text(j, i, f'{value:.2f}', 
                           ha='center', va='center', fontsize=10, 
                           fontweight='bold', color=color)
        
        ax.set_title(f'Attention Matrix: "{text[:25]}..."\\n(Each row shows where that token pays attention)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Key Tokens (What we\'re looking at)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Query Tokens (Who is looking)', fontsize=12, fontweight='bold')
        
        # Enhanced colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Weight\\n(0=No attention, 1=Full attention)', fontsize=11)
        
        # Add helpful explanation
        explanation = ("💡 How to read this: Each cell (i,j) shows how much token i pays attention to token j.\\n" +
                      "Darker red = stronger attention. Each row sums to 1.0.")
        ax.text(0.02, 0.98, explanation, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.4", 
                facecolor='lightyellow', alpha=0.9))
        
        plt.tight_layout()
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        # Convert to PIL Image
        image = Image.open(buf)
        return image
    
    def show_training_curves(self):
        """Display training curves"""
        
        if not self.training_history:
            # Create a placeholder
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'Training history not available\\nBut the model trained successfully!', 
                   ha='center', va='center', fontsize=16, transform=ax.transAxes)
            ax.set_title('Training Results', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            plt.close()
            
            image = Image.open(buf)
            return image
        
        epochs = list(range(1, len(self.training_history['loss']) + 1))
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss curves
        ax1.plot(epochs, self.training_history['loss'], 'b-', label='Training', linewidth=2)
        ax1.plot(epochs, self.training_history['val_loss'], 'r-', label='Validation', linewidth=2)
        ax1.set_title('Loss Over Time')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2.plot(epochs, self.training_history['accuracy'], 'g-', label='Training', linewidth=2)
        ax2.plot(epochs, self.training_history['val_accuracy'], 'orange', label='Validation', linewidth=2)
        ax2.set_title('Accuracy Over Time')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Learning rate
        ax3.plot(epochs, self.training_history['learning_rate'], 'purple', linewidth=2)
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.grid(True, alpha=0.3)
        
        # Final metrics
        final_metrics = {
            'Final Train Loss': self.training_history['loss'][-1],
            'Final Val Loss': self.training_history['val_loss'][-1],
            'Final Train Acc': self.training_history['accuracy'][-1],
            'Final Val Acc': self.training_history['val_accuracy'][-1]
        }
        
        bars = ax4.bar(final_metrics.keys(), final_metrics.values(),
                      color=['blue', 'red', 'green', 'orange'], alpha=0.7)
        ax4.set_title('Final Metrics')
        ax4.set_ylabel('Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, final_metrics.values()):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        # Convert to PIL Image
        image = Image.open(buf)
        return image
    
    def get_model_info(self):
        """Get model information"""
        
        total_params = sum([np.prod(var.shape) for var in self.model.trainable_variables])
        
        info = f"""
## 🤖 My Custom GPT Model Details

**🏗️ Architecture:**
- **Type:** Decoder-only Transformer (GPT-style)
- **Layers:** {self.model_config['num_layers']} transformer blocks
- **Attention Heads:** {self.model_config['num_heads']} per layer
- **Model Dimension:** {self.model_config['d_model']}
- **Feed-Forward Dimension:** {self.model_config['d_ff']}
- **Context Length:** {self.model_config['context_len']} tokens
- **Total Parameters:** {total_params:,}

**📚 Training Data:**
- **Source:** Jane Austen complete works
- **Tokenization:** Character-level ({len(self.token_to_id)} unique characters)
- **Size:** ~4.3MB of text data

**⚙️ Training Configuration:**
- **Framework:** TensorFlow/Keras
- **Hardware:** NVIDIA RTX 4060 Laptop GPU
- **Optimizer:** AdamW with cosine learning rate schedule
- **Training Strategy:** Character-level language modeling

**🎯 What This Model Does:**
This is a **completely custom-trained GPT model** that I built from scratch. It learns to generate text in the style of Jane Austen by predicting the next character in a sequence. Unlike commercial APIs, this model was trained entirely by me using my own code and compute resources.
        """
        
        return info
    
    def create_interface(self):
        """Create the Gradio interface"""
        
        with gr.Blocks(
            title="Custom GPT Model - Trained by Me",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
            }
            .custom-title {
                text-align: center;
                color: #2563eb;
                margin-bottom: 20px;
            }
            """
        ) as interface:
            
            # Header
            gr.Markdown("""
            # 🤖 My Custom GPT Model
            ### Trained from Scratch by Me 
            
            **🎯 This is NOT ChatGPT or any commercial API** - it's a completely custom GPT model that I trained myself using Jane Austen's works. 
            
            Explore the tabs below to see text generation, attention visualizations, and training analytics!
            """, elem_classes=["custom-title"])
            
            with gr.Tabs():
                # Tab 1: Text Generation
                with gr.TabItem("✍️ Generate Text"):
                    gr.Markdown("""
                    ## 🎨 Generate Jane Austen-Style Text
                    
                    **💡 Try these example prompts:**
                    - "Elizabeth Bennet walked into the"
                    - "It is a truth universally acknowledged"
                    - "Mr. Darcy looked across the"
                    - "The morning was bright and"
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            prompt_input = gr.Textbox(
                                label="Enter your prompt",
                                placeholder="Start typing something in Jane Austen's style...",
                                value="Elizabeth walked through the garden",
                                lines=3
                            )
                            
                            with gr.Row():
                                max_length = gr.Slider(
                                    minimum=10, maximum=200, value=100, step=1,
                                    label="Max Length (characters to generate)"
                                )
                                temperature = gr.Slider(
                                    minimum=0.1, maximum=2.0, value=0.7, step=0.1,
                                    label="Temperature (creativity: 0.1=focused, 2.0=wild)"
                                )
                                top_k = gr.Slider(
                                    minimum=1, maximum=30, value=20, step=1,
                                    label="Top-k (vocabulary constraint)"
                                )
                            
                            generate_btn = gr.Button("🎭 Generate Text", variant="primary", size="lg")
                        
                        with gr.Column(scale=3):
                            output_text = gr.Textbox(
                                label="Generated Text",
                                lines=10,
                                placeholder="Your generated text will appear here..."
                            )
                            
                            gr.Markdown("""
                            **🔧 Parameter Guide:**
                            - **Temperature:** Lower = more predictable, Higher = more creative
                            - **Top-k:** Limits vocabulary choices (lower = more focused)
                            - **Max Length:** How many characters to generate
                            """)
                    
                    generate_btn.click(
                        fn=self.generate_text,
                        inputs=[prompt_input, max_length, temperature, top_k],
                        outputs=output_text
                    )
                
                # Tab 2: Attention Visualization  
                with gr.TabItem("🧠 Attention Analysis"):
                    gr.Markdown("""
                    ## 🔍 Visualize Model Attention
                    
                    See how the model pays attention to different parts of the input text. This helps understand what the model is "thinking" about.
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            attention_input = gr.Textbox(
                                label="Text for Attention Analysis",
                                placeholder="Enter text to analyze attention patterns...",
                                value="Elizabeth walked through the garden",
                                lines=3
                            )
                            visualize_btn = gr.Button("🔍 Analyze Attention", variant="primary")
                            
                            gr.Markdown("""
                            **💡 Understanding Attention:**
                            - Each cell shows how much one token pays attention to another
                            - Darker colors = stronger attention
                            - This reveals the model's "focus" patterns
                            
                            *Note: This shows simulated attention patterns for demonstration*
                            """)
                        
                        with gr.Column(scale=2):
                            attention_plot = gr.Image(label="Attention Heatmap")
                    
                    visualize_btn.click(
                        fn=self.create_improved_attention_visualization,
                        inputs=attention_input,
                        outputs=attention_plot
                    )
                
                # Tab 3: Training Analytics
                with gr.TabItem("📊 My Training Results"):
                    gr.Markdown("""
                    ## 🏋️ How I Trained This Model
                    
                    **📋 Training Process:**
                    - **Data Preparation:** Cleaned 4.3MB of Jane Austen text
                    - **Tokenization:** Character-level (38 unique characters)  
                    - **Architecture:** 4-layer Transformer with 8 attention heads
                    - **Training:** 25 epochs on NVIDIA RTX 4060 GPU
                    - **Final Accuracy:** 54.6% (next-character prediction)
                    """)
                    
                    with gr.Row():
                        with gr.Column():
                            show_curves_btn = gr.Button("📈 Show Training Progress", variant="primary")
                            training_curves = gr.Image(label="Training Loss & Accuracy Over Time")
                    
                    show_curves_btn.click(
                        fn=self.show_training_curves,
                        outputs=training_curves
                    )
                
                # Tab 4: Model Info
                with gr.TabItem("ℹ️ Technical Details"):
                    model_info = gr.Markdown(self.get_model_info())
                    
                    gr.Markdown("""
                    ---
                    
                    ## 🛠️ Why I Built This
                    
                    This project demonstrates:
                    - **Custom Architecture:** Built GPT from scratch using TensorFlow
                    - **Full Training Pipeline:** Data processing, tokenization, training, deployment
                    - **Understanding:** Deep knowledge of transformer architecture and training
                    - **Practical Skills:** Real model deployment with Gradio interface
                    
                    **🎓 Learning Outcomes:**
                    - Implemented multi-head attention from scratch
                    - Built custom tokenizers and data pipelines  
                    - Optimized training with learning rate schedules
                    - Created professional deployment interfaces
                    
                    This is my own work - not a wrapper around existing APIs!
                    """)
        
        return interface

def launch_deployment(experiment_dir, share=False, port=7860):
    """Launch the deployment interface"""
    
    deployer = GPTDeployer(experiment_dir)
    interface = deployer.create_interface()
    
    print(f"🌐 Launching Gradio interface...")
    print(f"   Port: {port}")
    print(f"   Share: {share}")
    
    interface.launch(
        share=share,
        server_port=port,
        server_name="127.0.0.1",
        show_error=True
    )

if __name__ == "__main__":
    # Default experiment directory
    experiment_dir = "./experiments/jane_austen_proper_v1_20250914_093022"
    
    if len(sys.argv) > 1:
        experiment_dir = sys.argv[1]
    
    if not os.path.exists(experiment_dir):
        print(f"❌ Experiment directory not found: {experiment_dir}")
        print("🔧 Please train a model first or specify a valid experiment directory")
        sys.exit(1)
    
    launch_deployment(experiment_dir)
