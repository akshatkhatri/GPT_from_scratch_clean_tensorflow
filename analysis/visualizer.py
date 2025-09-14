#!/usr/bin/env python3
"""
Comprehensive visualization utilities for GPT training
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
import json
import tensorflow as tf
from matplotlib.patches import Rectangle
import io
import base64

class GPTVisualizer:
    """Comprehensive visualization toolkit for GPT training"""
    
    def __init__(self, experiment_dir, training_history, model, tokenizer_data):
        self.experiment_dir = experiment_dir
        self.viz_dir = f"{experiment_dir}/visualizations"
        self.training_history = training_history
        self.model = model
        self.tokenizer_data = tokenizer_data
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_training_curves(self):
        """Create comprehensive training curve visualizations"""
        
        # 1. Main training curves
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = self.training_history['epoch']
        
        # Loss curves
        ax1.plot(epochs, self.training_history['loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, self.training_history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2.plot(epochs, self.training_history['accuracy'], 'g-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, self.training_history['val_accuracy'], 'orange', label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Learning rate schedule
        ax3.plot(epochs, self.training_history['learning_rate'], 'purple', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Batch loss distribution (last epoch)
        if self.training_history['batch_losses']:
            ax4.hist(self.training_history['batch_losses'][-50:], bins=20, alpha=0.7, color='skyblue')
            ax4.set_xlabel('Batch Loss')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Recent Batch Loss Distribution')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.viz_dir}/training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Interactive Plotly version
        fig_plotly = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Loss Curves', 'Accuracy Curves', 'Learning Rate', 'Loss Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "histogram"}]]
        )
        
        # Loss
        fig_plotly.add_trace(
            go.Scatter(x=epochs, y=self.training_history['loss'], 
                      name='Train Loss', line=dict(color='blue')),
            row=1, col=1
        )
        fig_plotly.add_trace(
            go.Scatter(x=epochs, y=self.training_history['val_loss'], 
                      name='Val Loss', line=dict(color='red')),
            row=1, col=1
        )
        
        # Accuracy
        fig_plotly.add_trace(
            go.Scatter(x=epochs, y=self.training_history['accuracy'], 
                      name='Train Acc', line=dict(color='green')),
            row=1, col=2
        )
        fig_plotly.add_trace(
            go.Scatter(x=epochs, y=self.training_history['val_accuracy'], 
                      name='Val Acc', line=dict(color='orange')),
            row=1, col=2
        )
        
        # Learning rate
        fig_plotly.add_trace(
            go.Scatter(x=epochs, y=self.training_history['learning_rate'], 
                      name='LR', line=dict(color='purple')),
            row=2, col=1
        )
        
        # Batch losses histogram
        if self.training_history['batch_losses']:
            fig_plotly.add_trace(
                go.Histogram(x=self.training_history['batch_losses'][-100:], 
                           name='Batch Losses', marker_color='skyblue'),
                row=2, col=2
            )
        
        fig_plotly.update_layout(
            title="GPT Training Dashboard",
            height=600,
            showlegend=True
        )
        
        fig_plotly.write_html(f"{self.viz_dir}/training_dashboard.html")
    
    def visualize_attention(self, sample_text="Alice was coding", save_name="attention_map"):
        """Create attention visualizations"""
        
        # Tokenize sample text
        tokens = self.tokenizer_data['token_to_id']
        if isinstance(sample_text, str):
            # For character tokenization
            if self.tokenizer_data['tokenizer_type'] == 'char':
                token_ids = [tokens.get(char, 0) for char in sample_text]
            else:
                # Basic word splitting for demo
                words = sample_text.split()
                token_ids = [tokens.get(word, 0) for word in words]
        else:
            token_ids = sample_text
        
        context_len = min(len(token_ids), 32)  # Limit for visualization
        token_ids = token_ids[:context_len]
        
        # Pad if necessary
        if len(token_ids) < context_len:
            token_ids = token_ids + [0] * (context_len - len(token_ids))
        
        # Prepare input
        input_ids = tf.constant([token_ids], dtype=tf.int32)
        attention_mask = tf.ones_like(input_ids)
        
        # Get predictions (this would need model modification to return attention)
        # For now, create a mock attention matrix
        attention_weights = np.random.rand(8, context_len, context_len)  # [heads, seq, seq]
        attention_weights = attention_weights / attention_weights.sum(axis=-1, keepdims=True)
        
        # Create visualization
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        # Convert token IDs back to characters/words
        id_to_token = self.tokenizer_data['id_to_token']
        token_labels = [id_to_token.get(str(tid), '?') for tid in token_ids[:context_len]]
        
        for head in range(min(8, attention_weights.shape[0])):
            ax = axes[head]
            
            # Plot attention heatmap
            im = ax.imshow(attention_weights[head], cmap='Blues', aspect='auto')
            
            # Set ticks and labels
            ax.set_xticks(range(context_len))
            ax.set_yticks(range(context_len))
            ax.set_xticklabels(token_labels, rotation=45, ha='right')
            ax.set_yticklabels(token_labels)
            
            ax.set_title(f'Attention Head {head + 1}')
            ax.set_xlabel('Key Tokens')
            ax.set_ylabel('Query Tokens')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.suptitle(f'Attention Maps: "{sample_text}"', fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{self.viz_dir}/{save_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create interactive version
        self.create_interactive_attention(attention_weights, token_labels, sample_text, save_name)
    
    def create_interactive_attention(self, attention_weights, token_labels, sample_text, save_name):
        """Create interactive attention visualization"""
        
        num_heads = attention_weights.shape[0]
        
        # Create subplots for each attention head
        fig = make_subplots(
            rows=2, cols=4,
            subplot_titles=[f'Head {i+1}' for i in range(num_heads)],
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )
        
        for head in range(num_heads):
            row = head // 4 + 1
            col = head % 4 + 1
            
            heatmap = go.Heatmap(
                z=attention_weights[head],
                x=token_labels,
                y=token_labels,
                colorscale='Blues',
                showscale=(head == 0),  # Only show colorbar for first plot
                name=f'Head {head + 1}'
            )
            
            fig.add_trace(heatmap, row=row, col=col)
        
        fig.update_layout(
            title=f'Interactive Attention Maps: "{sample_text}"',
            height=600,
            width=1200
        )
        
        # Update axes
        for i in range(1, num_heads + 1):
            fig.update_xaxes(title_text="Key Tokens", row=(i-1)//4 + 1, col=(i-1)%4 + 1)
            fig.update_yaxes(title_text="Query Tokens", row=(i-1)//4 + 1, col=(i-1)%4 + 1)
        
        fig.write_html(f"{self.viz_dir}/{save_name}_interactive.html")
    
    def generate_text_samples(self, prompts=None, max_length=50):
        """Generate and visualize text samples"""
        
        if prompts is None:
            prompts = ["Alice", "Once upon", "The code", "In the"]
        
        results = []
        
        for prompt in prompts:
            # Generate text (simplified for demo)
            generated_text = self.simple_generate(prompt, max_length)
            
            results.append({
                'prompt': prompt,
                'generated': generated_text,
                'length': len(generated_text)
            })
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        
        y_pos = range(len(results))
        lengths = [r['length'] for r in results]
        
        bars = ax.barh(y_pos, lengths, color='lightblue', alpha=0.7)
        
        # Add text annotations
        for i, (bar, result) in enumerate(zip(bars, results)):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                   f'"{result["prompt"]}" → "{result["generated"][:30]}..."', 
                   va='center', fontsize=10)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([r['prompt'] for r in results])
        ax.set_xlabel('Generated Text Length')
        ax.set_title('Text Generation Samples')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.viz_dir}/text_generation_samples.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed results
        with open(f"{self.viz_dir}/generation_samples.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def simple_generate(self, prompt, max_length=50):
        """Simple text generation for demonstration"""
        
        # This would use the actual model for generation
        # For demo, return a simple concatenation
        continuations = [
            " was a programmer who loved to code.",
            " upon a time there was a beautiful algorithm.",
            " was elegant and efficient.",
            " beginning of a new era in AI."
        ]
        
        import random
        continuation = random.choice(continuations)
        
        return prompt + continuation[:max_length - len(prompt)]
    
    def visualize_model_architecture(self):
        """Create model architecture visualization"""
        
        # Model summary visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create a simple block diagram
        components = [
            ('Input Embedding', 'lightblue'),
            ('Positional Encoding', 'lightgreen'),
            ('Transformer Blocks', 'lightcoral'),
            ('Layer Norm', 'lightyellow'),
            ('Output Projection', 'lightpink')
        ]
        
        y_positions = np.arange(len(components))
        
        for i, (component, color) in enumerate(components):
            rect = Rectangle((0, i-0.4), 8, 0.8, facecolor=color, edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(4, i, component, ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Add arrows
        for i in range(len(components) - 1):
            ax.arrow(4, i + 0.4, 0, 0.2, head_width=0.2, head_length=0.1, fc='black', ec='black')
        
        ax.set_xlim(-1, 9)
        ax.set_ylim(-1, len(components))
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('GPT Model Architecture', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.viz_dir}/model_architecture.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_training_summary_dashboard(self):
        """Create a comprehensive training summary dashboard"""
        
        # Final metrics
        final_metrics = {
            'Final Training Loss': self.training_history['loss'][-1],
            'Final Validation Loss': self.training_history['val_loss'][-1],
            'Final Training Accuracy': self.training_history['accuracy'][-1],
            'Final Validation Accuracy': self.training_history['val_accuracy'][-1],
            'Best Validation Loss': min(self.training_history['val_loss']),
            'Total Epochs': len(self.training_history['epoch'])
        }
        
        # Create dashboard
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Training Progress', 'Final Metrics', 'Attention Sample',
                          'Generation Sample', 'Loss Distribution', 'Model Info'),
            specs=[[{"type": "scatter"}, {"type": "bar"}, {"type": "scatter"}],
                   [{"colspan": 2}, None, {"type": "table"}]]
        )
        
        # Training progress
        epochs = self.training_history['epoch']
        fig.add_trace(
            go.Scatter(x=epochs, y=self.training_history['loss'], name='Train Loss'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=self.training_history['val_loss'], name='Val Loss'),
            row=1, col=1
        )
        
        # Final metrics bar chart
        fig.add_trace(
            go.Bar(x=list(final_metrics.keys()), y=list(final_metrics.values()),
                  name='Final Metrics'),
            row=1, col=2
        )
        
        # Model info table
        model_info = [
            ['Parameter', 'Value'],
            ['Total Parameters', '~2M'],
            ['Model Dimension', str(self.model.d_model) if hasattr(self.model, 'd_model') else 'N/A'],
            ['Attention Heads', '8'],
            ['Decoder Blocks', '3']
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=model_info[0]),
                cells=dict(values=list(zip(*model_info[1:])))
            ),
            row=2, col=3
        )
        
        fig.update_layout(
            title="GPT Training Summary Dashboard",
            height=800,
            showlegend=True
        )
        
        fig.write_html(f"{self.viz_dir}/training_summary_dashboard.html")
        
        # Save metrics
        with open(f"{self.viz_dir}/final_metrics.json", 'w') as f:
            json.dump(final_metrics, f, indent=2)


if __name__ == "__main__":
    print("Visualization utilities loaded!")
    print("Use with: GPTVisualizer(experiment_dir, training_history, model, tokenizer_data)")
