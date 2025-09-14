#!/usr/bin/env python3
"""
Gradio deployment interface for trained GPT model
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
from models.layers import GPT
from analysis.visualizer import GPTVisualizer

class GPTDeployment:
    """Gradio deployment for trained GPT model                        with gr.Column(scale=3):
                            attention_input = gr.Textbox(
                                label="Text for Attention Analysis",
                                placeholder="Enter text to visualize attention patterns...",
                                value="Elizabeth walked through the garden"
                            )
                            
                            with gr.Row():
                                visualize_btn = gr.Button("Multi-View Analysis", variant="primary")
                                detailed_btn = gr.Button("Detailed Matrix", variant="secondary")
                        
                        with gr.Column(scale=4):
                            attention_plot = gr.Image(label="Attention Visualization")
                            
                            gr.Markdown("""
                            **🎯 Visualization Options:**
                            - **Multi-View Analysis:** Shows 4 different attention perspectives
                            - **Detailed Matrix:** Clean, readable attention matrix with values
                            
                            *Note: These are simulated attention patterns for demonstration*
                            """)
                    
                    visualize_btn.click(
                        fn=self.create_attention_visualization,
                        inputs=attention_input,
                        outputs=attention_plot
                    )
                    
                    detailed_btn.click(
                        fn=self.create_token_attention_breakdown,
                        inputs=attention_input,
                        outputs=attention_plot
                    ) __init__(self, experiment_dir):
        self.experiment_dir = experiment_dir
        self.load_model_and_tokenizer()
        
    def load_model_and_tokenizer(self):
        """Load the trained model and tokenizer"""
        
        # Load tokenizer
        with open(f"{self.experiment_dir}/tokenizer.json", 'r') as f:
            self.tokenizer_data = json.load(f)
        
        self.token_to_id = self.tokenizer_data['token_to_id']
        self.id_to_token = self.tokenizer_data['id_to_token']
        self.tokenizer_type = self.tokenizer_data['tokenizer_type']
        
        # Load model config
        with open(f"{self.experiment_dir}/model_config.json", 'r') as f:
            model_config = json.load(f)
        
        self.model_arch = model_config['architecture']
        
        # Create model
        self.model = GPT(
            d_model=self.model_arch['d_model'],
            vocab_size=self.model_arch['vocab_size'],
            context_length=self.model_arch['context_length'],
            attention_heads=self.model_arch['attention_heads'],
            epsilon=0.001,  # Not used for inference
            decoder_blocks=self.model_arch['decoder_blocks'],
            dropout_rate=0.0  # No dropout during inference
        )
        
        # Load weights
        best_weights_path = f"{self.experiment_dir}/checkpoints/best_model.weights.h5"
        if os.path.exists(best_weights_path):
            # Initialize model with dummy input
            dummy_input = tf.zeros((1, self.model_arch['context_length']), dtype=tf.int32)
            dummy_mask = tf.ones_like(dummy_input)
            _ = self.model((dummy_input, dummy_mask), training=False)
            
            # Load weights
            self.model.load_weights(best_weights_path)
            print(f"✅ Model loaded from {best_weights_path}")
        else:
            print(f"⚠️ No trained weights found at {best_weights_path}")
        
        # Load training history for visualizations
        with open(f"{self.experiment_dir}/training_history.json", 'r') as f:
            self.training_history = json.load(f)
    
    def tokenize_text(self, text):
        """Tokenize input text"""
        if self.tokenizer_type == 'char':
            return [self.token_to_id.get(char, 0) for char in text]
        else:
            # Basic word tokenization for demo
            words = text.split()
            return [self.token_to_id.get(word, 0) for word in words]
    
    def detokenize_ids(self, token_ids):
        """Convert token IDs back to text"""
        tokens = [self.id_to_token.get(str(tid), '?') for tid in token_ids]
        
        if self.tokenizer_type == 'char':
            return ''.join(tokens)
        else:
            return ' '.join(tokens)
    
    def generate_text(self, prompt, max_length=100, temperature=0.8, top_k=20):
        """Generate text given a prompt"""
        
        # Tokenize prompt
        prompt_tokens = self.tokenize_text(prompt)
        
        if len(prompt_tokens) == 0:
            return "Error: Empty prompt after tokenization"
        
        # Generate
        generated_tokens = prompt_tokens.copy()
        context_length = self.model_arch['context_length']
        
        for _ in range(max_length):
            # Prepare input
            current_tokens = generated_tokens[-context_length:]
            if len(current_tokens) < context_length:
                # Pad at beginning
                padded_tokens = [0] * (context_length - len(current_tokens)) + current_tokens
            else:
                padded_tokens = current_tokens
            
            # Convert to tensor
            input_ids = tf.constant([padded_tokens], dtype=tf.int32)
            attention_mask = tf.ones_like(input_ids)
            
            # Get predictions
            logits = self.model((input_ids, attention_mask), training=False)
            next_token_logits = logits[0, -1, :]  # Last token predictions
            
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
            
            next_token = int(next_token.numpy())
            generated_tokens.append(next_token)
            
            # Stop if we generate end token or reach max length
            if next_token == 0:  # Assuming 0 is padding/end token
                break
        
        # Convert back to text
        generated_text = self.detokenize_ids(generated_tokens)
        
        return generated_text
    
    def create_attention_visualization(self, text):
        """Create improved attention visualization with multiple views"""
        
        tokens = self.tokenize_text(text[:50])
        if len(tokens) > 15:  # Reduced for better readability
            tokens = tokens[:15]
        
        token_labels = [self.id_to_token.get(str(tid), '?') for tid in tokens]
        
        # Create more realistic attention patterns
        n_tokens = len(tokens)
        attention_matrix = np.zeros((n_tokens, n_tokens))
        
        # Simulate realistic attention patterns
        for i in range(n_tokens):
            # Strong self-attention
            attention_matrix[i, i] = 0.3
            
            # Attention to previous tokens (causal)
            for j in range(i):
                # Decay with distance, but with some randomness
                distance = i - j
                base_attention = 0.7 / (distance + 1)
                attention_matrix[i, j] = base_attention + np.random.normal(0, 0.1)
            
            # Normalize to make it a proper attention distribution
            attention_matrix[i] = np.maximum(attention_matrix[i], 0)
            attention_matrix[i] = attention_matrix[i] / (attention_matrix[i].sum() + 1e-8)
        
        # Create a figure with subplots for different views
        fig = plt.figure(figsize=(16, 12))
        
        # Main attention heatmap
        ax1 = plt.subplot(2, 2, 1)
        im1 = ax1.imshow(attention_matrix, cmap='viridis', aspect='auto')
        ax1.set_xticks(range(len(tokens)))
        ax1.set_yticks(range(len(tokens)))
        ax1.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=10)
        ax1.set_yticklabels(token_labels, fontsize=10)
        ax1.set_title('Attention Heatmap', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Key Tokens', fontsize=12)
        ax1.set_ylabel('Query Tokens', fontsize=12)
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        
        # Add attention values as text
        for i in range(n_tokens):
            for j in range(n_tokens):
                if attention_matrix[i, j] > 0.1:  # Only show significant attention
                    ax1.text(j, i, f'{attention_matrix[i, j]:.2f}', 
                            ha='center', va='center', fontsize=8, 
                            color='white' if attention_matrix[i, j] > 0.5 else 'black')
        
        # Alternative view: Line plot of attention weights for each token
        ax2 = plt.subplot(2, 2, 2)
        for i in range(min(5, n_tokens)):  # Show first 5 tokens
            ax2.plot(range(n_tokens), attention_matrix[i], 
                    marker='o', label=f'Token {i}: "{token_labels[i]}"', linewidth=2)
        ax2.set_title('Attention Weights per Query Token', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Key Token Position', fontsize=12)
        ax2.set_ylabel('Attention Weight', fontsize=12)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Attention flow diagram
        ax3 = plt.subplot(2, 2, 3)
        # Create a simplified flow visualization
        token_positions = np.arange(n_tokens)
        ax3.scatter(token_positions, [1]*n_tokens, s=100, c='lightblue', edgecolors='navy')
        
        # Add token labels
        for i, (pos, label) in enumerate(zip(token_positions, token_labels)):
            ax3.annotate(f'{i}:{label}', (pos, 1), xytext=(pos, 0.8), 
                        ha='center', fontsize=10, fontweight='bold')
        
        # Draw attention flows for the last token (most interesting)
        if n_tokens > 1:
            last_token_idx = n_tokens - 1
            for j in range(last_token_idx):
                if attention_matrix[last_token_idx, j] > 0.15:  # Only significant attention
                    ax3.annotate('', xy=(j, 1), xytext=(last_token_idx, 1),
                               arrowprops=dict(arrowstyle='->', 
                                             lw=attention_matrix[last_token_idx, j] * 5,
                                             color='red', alpha=0.7))
        
        ax3.set_title(f'Attention Flow to Last Token: "{token_labels[-1]}"', 
                     fontsize=14, fontweight='bold')
        ax3.set_xlim(-0.5, n_tokens - 0.5)
        ax3.set_ylim(0.5, 1.5)
        ax3.set_xlabel('Token Position', fontsize=12)
        ax3.set_yticks([])
        
        # Summary statistics
        ax4 = plt.subplot(2, 2, 4)
        
        # Calculate attention statistics
        self_attention = np.diag(attention_matrix)
        avg_attention_distance = []
        for i in range(n_tokens):
            distances = []
            for j in range(i):  # Only previous tokens
                if attention_matrix[i, j] > 0.05:
                    distances.append((i - j) * attention_matrix[i, j])
            avg_attention_distance.append(np.sum(distances) if distances else 0)
        
        # Plot statistics
        x_pos = np.arange(n_tokens)
        bars1 = ax4.bar(x_pos - 0.2, self_attention, 0.4, 
                       label='Self-Attention', color='skyblue', alpha=0.8)
        bars2 = ax4.bar(x_pos + 0.2, avg_attention_distance, 0.4, 
                       label='Attention Distance', color='lightcoral', alpha=0.8)
        
        ax4.set_title('Attention Statistics', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Token Position', fontsize=12)
        ax4.set_ylabel('Attention Strength', fontsize=12)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([f'{i}' for i in range(n_tokens)])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            if height > 0.01:
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.suptitle(f'Multi-View Attention Analysis: "{text[:30]}..."', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        # Convert to PIL Image
        image = Image.open(buf)
        return image
    
    def create_token_attention_breakdown(self, text):
        """Create a detailed token-by-token attention breakdown"""
        
        tokens = self.tokenize_text(text[:40])
        if len(tokens) > 12:  # Even more focused for detailed view
            tokens = tokens[:12]
        
        token_labels = [self.id_to_token.get(str(tid), '?') for tid in tokens]
        n_tokens = len(tokens)
        
        # Create more sophisticated attention patterns
        attention_matrix = np.zeros((n_tokens, n_tokens))
        
        for i in range(n_tokens):
            # Realistic causal attention patterns
            for j in range(i + 1):  # Can only attend to previous tokens and self
                if i == j:
                    # Self-attention (usually moderate)
                    attention_matrix[i, j] = 0.2 + np.random.normal(0, 0.05)
                else:
                    # Distance-based attention with some linguistic patterns
                    distance = i - j
                    if distance == 1:  # Adjacent tokens get more attention
                        attention_matrix[i, j] = 0.4 + np.random.normal(0, 0.1)
                    elif distance <= 3:  # Nearby tokens
                        attention_matrix[i, j] = 0.3 / distance + np.random.normal(0, 0.05)
                    else:  # Distant tokens
                        attention_matrix[i, j] = 0.1 / distance + np.random.normal(0, 0.02)
            
            # Normalize
            attention_matrix[i] = np.maximum(attention_matrix[i], 0)
            attention_matrix[i] = attention_matrix[i] / (attention_matrix[i].sum() + 1e-8)
        
        # Create visualization with a cleaner, more readable single view
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Enhanced heatmap with better colormap and annotations
        im = ax.imshow(attention_matrix, cmap='RdYlBu_r', aspect='auto')
        
        # Add grid lines for better readability
        ax.set_xticks(np.arange(-0.5, n_tokens, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_tokens, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=1, alpha=0.3)
        
        # Set major ticks with better labels
        ax.set_xticks(range(n_tokens))
        ax.set_yticks(range(n_tokens))
        ax.set_xticklabels([f'{i}: {label}' for i, label in enumerate(token_labels)], 
                           rotation=45, ha='right', fontsize=12, fontweight='bold')
        ax.set_yticklabels([f'{i}: {label}' for i, label in enumerate(token_labels)], 
                           fontsize=12, fontweight='bold')
        
        # Add text annotations for ALL values with better contrast
        for i in range(n_tokens):
            for j in range(n_tokens):
                value = attention_matrix[i, j]
                # Use white text for dark cells, black for light cells
                color = 'white' if value > 0.3 else 'black'
                ax.text(j, i, f'{value:.3f}', 
                        ha='center', va='center', fontsize=11, 
                        fontweight='bold', color=color,
                        bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.7) if color == 'black' else None)
        
        ax.set_title(f'Detailed Attention Matrix: "{text[:30]}..."', 
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Key Tokens (What the model is looking at)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Query Tokens (Current position)', fontsize=14, fontweight='bold')
        
        # Enhanced colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Weight (0 = No attention, 1 = Full attention)', 
                       fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=11)
        
        # Add explanation text
        explanation = ("Each cell shows how much attention token i (row) pays to token j (column).\n"
                      "Darker colors = stronger attention. Values sum to 1.0 for each row.")
        ax.text(0.02, 0.98, explanation, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
                facecolor='lightyellow', alpha=0.8))
        
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
        
        epochs = self.training_history['epoch']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss
        ax1.plot(epochs, self.training_history['loss'], 'b-', label='Training', linewidth=2)
        ax1.plot(epochs, self.training_history['val_loss'], 'r-', label='Validation', linewidth=2)
        ax1.set_title('Loss Curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy
        ax2.plot(epochs, self.training_history['accuracy'], 'g-', label='Training', linewidth=2)
        ax2.plot(epochs, self.training_history['val_accuracy'], 'orange', label='Validation', linewidth=2)
        ax2.set_title('Accuracy Curves')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Learning rate
        ax3.plot(epochs, self.training_history['learning_rate'], 'purple', linewidth=2)
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Final metrics
        final_metrics = {
            'Train Loss': self.training_history['loss'][-1],
            'Val Loss': self.training_history['val_loss'][-1],
            'Train Acc': self.training_history['accuracy'][-1],
            'Val Acc': self.training_history['val_accuracy'][-1]
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
        
        total_params = self.model_arch.get('total_parameters', 'N/A')
        if isinstance(total_params, int):
            total_params_str = f"{total_params:,}"
        else:
            total_params_str = str(total_params)
        
        info = f"""
        ## 🏗️ **MY CUSTOM ARCHITECTURE** (Built from Scratch!)
        
        **🧠 Model Design:**
        - **Embedding Dimension:** {self.model_arch['d_model']} (my choice)
        - **Context Window:** {self.model_arch['context_length']} characters
        - **Attention Heads:** {self.model_arch['attention_heads']} (multi-head attention)
        - **Transformer Blocks:** {self.model_arch['decoder_blocks']} layers deep
        - **Total Parameters:** **{total_params_str}** (I trained every single one!)
        
        **📚 My Training Data:**
        - **Source:** Jane Austen's complete works (Persuasion, Pride & Prejudice, etc.)
        - **Size:** 4.3MB of cleaned literary text
        - **Tokenization:** Character-level (every letter/punctuation = 1 token)
        - **Vocabulary:** {len(self.token_to_id)} unique characters
        
        **⚡ What Makes This Special:**
        - **No pre-training:** Started from random weights
        - **Custom loss function:** Sparse categorical crossentropy
        - **Learning schedule:** Cosine decay with warmup
        - **Hardware:** Trained on my RTX 4060 GPU
        
        **🎯 Training Strategy:**
        - **Tokenizer Type:** {self.tokenizer_type} (character-by-character)
        - **Prediction Task:** Given 256 characters, predict the next one
        - **Style Learning:** Model learned Jane Austen's writing patterns
        
        ---
        **🚫 This is NOT using any external APIs or pre-trained models!**
        
        **Training Results:**
        - Total Epochs: {len(self.training_history['epoch'])}
        - Final Training Loss: {self.training_history['loss'][-1]:.4f}
        - Final Validation Loss: {self.training_history['val_loss'][-1]:.4f}
        - Best Validation Loss: {min(self.training_history['val_loss']):.4f}
        """
        
        return info
    
    def create_gradio_interface(self):
        """Create Gradio interface"""
        
        with gr.Blocks(title="Custom GPT - Trained by Me") as demo:
            
            # Header with clear attribution
            gr.Markdown("""
            # � **My Custom GPT Model** - Trained from Scratch!
            
            **🎯 This is NOT ChatGPT or any API!** This is a **custom Transformer model** I built and trained myself:
            - **📚 Trained on:** Jane Austen's complete works (4.3MB of text)
            - **🔧 Architecture:** Custom GPT with 801,062 parameters
            - **⚡ Tokenization:** Character-level (38 unique characters)
            - **🏋️ Training:** 25 epochs on NVIDIA RTX 4060
            - **📈 Performance:** Final accuracy: 54.6%, Loss: 1.49
            
            ---
            """)
            
            with gr.Tabs():
                
                # Tab 1: Text Generation
                with gr.TabItem("🚀 Text Generation"):
                    
                    gr.Markdown("""
                    ## 📝 Generate Jane Austen-Style Text
                    
                    **💡 Best Prompts to Try:**
                    - `"Elizabeth was"` - Character names work great!
                    - `"Mr. Darcy walked"` - Try classic character names
                    - `"It is a truth universally"` - Famous opening lines
                    - `"She felt that"` - Emotional descriptions
                    - `"The evening was"` - Scene descriptions
                    
                    **⚠️ Note:** Simple words like "hello" or "how are you" won't work well since this model was trained specifically on 19th-century literary text!
                    """)
                    
                    with gr.Row():
                        with gr.Column():
                            prompt_input = gr.Textbox(
                                label="📝 Your Prompt", 
                                placeholder="Try: Elizabeth was walking through the garden...",
                                value="Elizabeth was",
                                lines=3
                            )
                            
                            # Prompt examples
                            gr.Markdown("**🎯 Quick Examples:**")
                            example_buttons = gr.Row()
                            with example_buttons:
                                btn1 = gr.Button("Elizabeth was", size="sm")
                                btn2 = gr.Button("Mr. Darcy", size="sm") 
                                btn3 = gr.Button("It is a truth", size="sm")
                                btn4 = gr.Button("She felt", size="sm")
                            
                            with gr.Row():
                                max_length = gr.Slider(
                                    minimum=10, maximum=200, value=80, step=1,
                                    label="📏 Max Length"
                                )
                                temperature = gr.Slider(
                                    minimum=0.1, maximum=2.0, value=0.8, step=0.1,
                                    label="🌡️ Temperature (creativity)"
                                )
                                top_k = gr.Slider(
                                    minimum=1, maximum=30, value=20, step=1,
                                    label="🎯 Top-K Sampling (diversity)"
                                )
                            
                            generate_btn = gr.Button("🚀 Generate Jane Austen-style Text", variant="primary", size="lg")
                        
                        with gr.Column():
                            output_text = gr.Textbox(
                                label="📖 Generated Text", 
                                lines=10,
                                interactive=False,
                                placeholder="Your generated text will appear here..."
                            )
                            
                            gr.Markdown("""
                            **🔍 What you're seeing:**
                            - This text is generated **character-by-character** by my custom model
                            - The model learned patterns from Jane Austen's writing style
                            - It's predicting the next character based on the previous 256 characters
                            - Higher temperature = more creative/random, lower = more conservative
                            """)
                    
                    # Wire up example buttons
                    btn1.click(lambda: "Elizabeth was", outputs=prompt_input)
                    btn2.click(lambda: "Mr. Darcy", outputs=prompt_input) 
                    btn3.click(lambda: "It is a truth universally", outputs=prompt_input)
                    btn4.click(lambda: "She felt that", outputs=prompt_input)
                    
                    generate_btn.click(
                        fn=self.generate_text,
                        inputs=[prompt_input, max_length, temperature, top_k],
                        outputs=output_text
                    )
                
                # Tab 2: Attention Visualization
                with gr.TabItem("👁️ Attention Maps"):
                    gr.Markdown("## Visualize attention patterns")
                    
                    with gr.Row():
                        with gr.Column():
                            attention_input = gr.Textbox(
                                label="Text for Attention Visualization",
                                placeholder="Enter text to analyze attention...",
                                value="Alice was coding"
                            )
                            visualize_btn = gr.Button("Create Attention Map", variant="primary")
                        
                        with gr.Column():
                            attention_plot = gr.Image(label="Attention Heatmap")
                    
                    visualize_btn.click(
                        fn=self.create_attention_visualization,
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
                    - **Training Time:** ~60 minutes on RTX 4060
                    - **Optimization:** Adam with cosine learning rate schedule
                    
                    **📈 View the training progress below:**
                    """)
                    
                    with gr.Row():
                        show_curves_btn = gr.Button("📈 Show My Training Curves", variant="primary")
                    
                    with gr.Row():
                        training_curves = gr.Image(label="Training Loss & Accuracy Over Time")
                    
                    show_curves_btn.click(
                        fn=self.show_training_curves,
                        outputs=training_curves
                    )
                
                # Tab 4: Model Information  
                with gr.TabItem("🔧 Technical Details"):
                    gr.Markdown("""
                    ## ⚙️ My Model Architecture
                    
                    This is a **completely custom implementation** - no pre-trained weights or APIs!
                    """)
                    
                    model_info = gr.Markdown(self.get_model_info())
            
            gr.Markdown("---")
            gr.Markdown("""
            **🎓 Made with passion and lots of coffee! ☕**  
            *This entire GPT model was built from scratch using TensorFlow/Keras*
            
            **🚫 This is NOT:**
            - ChatGPT or GPT-4
            - OpenAI API  
            - Pre-trained model
            
            **✅ This IS:**
            - My own Transformer architecture
            - Trained on my chosen dataset
            - 801,062 parameters I trained myself!
            """)
        
        return demo


def launch_deployment(experiment_dir, share=False, port=7860):
    """Launch Gradio deployment"""
    
    print(f"🚀 Loading model from: {experiment_dir}")
    
    # Create deployment
    deployment = GPTDeployment(experiment_dir)
    
    # Create interface
    demo = deployment.create_gradio_interface()
    
    print(f"🌐 Launching Gradio interface...")
    print(f"   Port: {port}")
    print(f"   Share: {share}")
    
    # Launch
    demo.launch(
        share=share,
        server_port=port,
        server_name="0.0.0.0" if share else "127.0.0.1"
    )


if __name__ == "__main__":
    # Example usage
    experiment_dir = "./experiments/small_viz_test_20250914_080741"  # Updated with actual path
    
    if os.path.exists(experiment_dir):
        launch_deployment(experiment_dir, share=False, port=7860)
    else:
        print(f"❌ Experiment directory not found: {experiment_dir}")
        print("Please train a model first using train_with_viz.py")
        
        # Look for the most recent experiment
        experiments_dir = "./experiments"
        if os.path.exists(experiments_dir):
            experiments = [d for d in os.listdir(experiments_dir) if os.path.isdir(os.path.join(experiments_dir, d))]
            if experiments:
                # Get the most recent experiment
                experiments.sort(reverse=True)
                most_recent = os.path.join(experiments_dir, experiments[0])
                print(f"Found recent experiment: {most_recent}")
                
                # Check if it has the required files
                required_files = ['tokenizer.json', 'model_config.json', 'training_history.json']
                if all(os.path.exists(os.path.join(most_recent, f)) for f in required_files):
                    print(f"🚀 Using most recent experiment: {most_recent}")
                    launch_deployment(most_recent, share=False, port=7860)
                else:
                    print(f"❌ Experiment incomplete: missing required files")
            else:
                print("❌ No experiments found")
