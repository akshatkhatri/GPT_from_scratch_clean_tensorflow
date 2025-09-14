#!/usr/bin/env python3
"""
Enhanced training script with comprehensive visualizations and model saving
"""

import sys
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.utils import create_tokenizer, load_config
from data.data_processing import prepare_tfrecords
from models.layers import GPT, CosineDecayWithWarmup
from analysis.visualizer import GPTVisualizer

class VisualizationTrainer:
    """Enhanced trainer with comprehensive visualizations"""
    
    def __init__(self, config_path, text_file_path, experiment_name="gpt_experiment"):
        self.config = load_config(config_path)
        self.text_file_path = text_file_path
        self.experiment_name = experiment_name
        
        # Create experiment directory
        self.experiment_dir = f"./experiments/{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(f"{self.experiment_dir}/visualizations", exist_ok=True)
        os.makedirs(f"{self.experiment_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{self.experiment_dir}/logs", exist_ok=True)
        
        print(f"🎯 Experiment: {self.experiment_name}")
        print(f"📁 Directory: {self.experiment_dir}")
        
        # Initialize tracking
        self.training_history = {
            'epoch': [],
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': [],
            'learning_rate': [],
            'batch_losses': [],
            'attention_maps': [],
            'generation_samples': []
        }
        
        # Initialize visualizer
        self.visualizer = None  # Will be initialized after model creation
        
    def setup_data_and_model(self):
        """Setup tokenizer, data, and model"""
        print("\n1. Setting up tokenizer...")
        
        # Create tokenizer
        self.token_to_id_dict, self.tokenize_func, self.extra_info = create_tokenizer(
            self.config, [self.text_file_path]
        )
        self.vocab_size = len(self.token_to_id_dict)
        self.id_to_token_dict = {v: k for k, v in self.token_to_id_dict.items()}
        
        print(f"✅ Tokenizer: {self.config['TOKENIZER_TYPE']}, vocab size: {self.vocab_size}")
        
        # Save tokenizer
        tokenizer_data = {
            'token_to_id': self.token_to_id_dict,
            'id_to_token': self.id_to_token_dict,
            'tokenizer_type': self.config['TOKENIZER_TYPE'],
            'extra_info': self.extra_info
        }
        
        with open(f"{self.experiment_dir}/tokenizer.json", 'w') as f:
            json.dump(tokenizer_data, f, indent=2)
        
        print("\n2. Preparing data...")
        
        # Prepare data
        self.train_ds, self.val_ds, self.steps_per_epoch = prepare_tfrecords(
            text_file_path=self.text_file_path,
            token_to_id_dict=self.token_to_id_dict,
            tokenize_func=self.tokenize_func,
            context_length=self.config['CONTEXT_LEN'],
            batch_size=self.config.get('BATCH_SIZE', 4),
            records_per_file=self.config.get('RECORDS_PER_FILE', 50),
            version_name=f"{self.experiment_name}_data"
        )
        
        # Use actual calculated steps, but limit to config max
        self.actual_steps = min(self.config['STEPS_PER_EPOCH'], self.steps_per_epoch)
        print(f"✅ Data prepared, using {self.actual_steps} steps per epoch")
        
        print("\n3. Creating model...")
        
        # Create model
        self.model = GPT(
            d_model=self.config['D_MODEL'],
            vocab_size=self.vocab_size,
            context_length=self.config['CONTEXT_LEN'],
            attention_heads=self.config['ATTENTION_HEADS'],
            epsilon=self.config['LEARNING_RATE'],
            decoder_blocks=self.config['DECODER_BLOCKS'],
            dropout_rate=self.config['DROPOUT_RATE']
        )
        
        # Build the model by calling it with dummy input to initialize variables
        dummy_input = tf.zeros((1, self.config['CONTEXT_LEN']), dtype=tf.int32)
        dummy_mask = tf.ones((1, self.config['CONTEXT_LEN']), dtype=tf.float32)
        _ = self.model([dummy_input, dummy_mask])
        
        # Now count parameters
        try:
            total_params = sum(int(tf.size(var)) for var in self.model.trainable_variables)
            print(f"✅ Model created: {total_params:,} parameters")
        except Exception as e:
            print(f"✅ Model created (parameter counting failed: {e})")
            # Alternative method using Keras
            try:
                self.model.build([(None, self.config['CONTEXT_LEN']), (None, self.config['CONTEXT_LEN'])])
                total_params = self.model.count_params()
                print(f"✅ Model parameters: {total_params:,}")
            except Exception as e2:
                print(f"✅ Model created (both parameter counting methods failed)")
        
        # Initialize visualizer after model is created (will be initialized when needed)
        self.visualizer = None
        
        # Save model config
        model_config = {
            'architecture': {
                'd_model': self.config['D_MODEL'],
                'vocab_size': self.vocab_size,
                'context_length': self.config['CONTEXT_LEN'],
                'attention_heads': self.config['ATTENTION_HEADS'],
                'decoder_blocks': self.config['DECODER_BLOCKS'],
                'dropout_rate': self.config['DROPOUT_RATE']
            },
            'total_parameters': int(total_params),
            'training_config': self.config
        }
        
        with open(f"{self.experiment_dir}/model_config.json", 'w') as f:
            json.dump(model_config, f, indent=2)
    
    def setup_training(self):
        """Setup training components"""
        print("\n4. Setting up training...")
        
        # Learning rate schedule
        total_steps = self.actual_steps * self.config['EPOCHS']
        warmup_steps = int(total_steps * self.config['WARMUP_RATIO'])
        
        self.lr_schedule = CosineDecayWithWarmup(
            min_learning_rate=self.config['MIN_LEARNING_RATE'],
            peak_learning_rate=self.config['PEAK_LEARNING_RATE'],
            warmup_steps=warmup_steps,
            total_steps=total_steps
        )
        
        # Optimizer and loss
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
        
        print(f"✅ Training setup complete")
        print(f"   Total steps: {total_steps}, Warmup: {warmup_steps}")
    
    def train_with_visualizations(self):
        """Main training loop with visualizations"""
        print(f"\n5. Starting training for {self.config['EPOCHS']} epochs...")
        print("=" * 60)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config['EPOCHS']):
            print(f"\nEpoch {epoch + 1}/{self.config['EPOCHS']}")
            
            # Reset metrics
            self.train_loss.reset_state()
            self.train_accuracy.reset_state()
            self.val_loss.reset_state()
            self.val_accuracy.reset_state()
            
            # Training step
            epoch_batch_losses = []
            for step, (inputs, targets) in enumerate(self.train_ds.take(self.actual_steps)):
                loss, accuracy = self.train_step(inputs, targets)
                epoch_batch_losses.append(float(loss))
                
                if step % max(1, self.actual_steps // 5) == 0:
                    current_lr = float(self.optimizer.learning_rate.numpy())
                    print(f"  Step {step + 1}/{self.actual_steps}: loss={loss:.4f}, acc={accuracy:.4f}, lr={current_lr:.6f}")
            
            # Validation step
            for inputs, targets in self.val_ds.take(max(1, self.actual_steps // 4)):
                self.val_step(inputs, targets)
            
            # Record metrics
            train_loss_val = float(self.train_loss.result())
            train_acc_val = float(self.train_accuracy.result())
            val_loss_val = float(self.val_loss.result())
            val_acc_val = float(self.val_accuracy.result())
            current_lr = float(self.optimizer.learning_rate.numpy())
            
            self.training_history['epoch'].append(epoch + 1)
            self.training_history['loss'].append(train_loss_val)
            self.training_history['accuracy'].append(train_acc_val)
            self.training_history['val_loss'].append(val_loss_val)
            self.training_history['val_accuracy'].append(val_acc_val)
            self.training_history['learning_rate'].append(current_lr)
            self.training_history['batch_losses'].extend(epoch_batch_losses)
            
            print(f"  Epoch {epoch + 1} Summary:")
            print(f"    Train Loss: {train_loss_val:.4f}, Train Acc: {train_acc_val:.4f}")
            print(f"    Val Loss: {val_loss_val:.4f}, Val Acc: {val_acc_val:.4f}")
            print(f"    Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_loss_val < best_val_loss:
                best_val_loss = val_loss_val
                self.model.save_weights(f"{self.experiment_dir}/checkpoints/best_model.weights.h5")
                print(f"    🏆 New best model saved! (val_loss: {val_loss_val:.4f})")
            
            # Save checkpoint every few epochs
            if (epoch + 1) % max(1, self.config['EPOCHS'] // 5) == 0:
                self.model.save_weights(f"{self.experiment_dir}/checkpoints/epoch_{epoch + 1:02d}.weights.h5")
            
            # Generate visualizations every few epochs
            if (epoch + 1) % max(1, self.config['EPOCHS'] // 3) == 0:
                self.generate_epoch_visualizations(epoch + 1)
        
        print("\n✅ Training completed!")
        
        # Final save
        self.model.save_weights(f"{self.experiment_dir}/checkpoints/final_model.weights.h5")
        
        # Save training history
        with open(f"{self.experiment_dir}/training_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Generate final visualizations
        self.generate_final_visualizations()
    
    @tf.function
    def train_step(self, inputs, targets):
        """Single training step"""
        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            loss = self.loss_fn(targets, predictions)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        self.train_loss.update_state(loss)
        self.train_accuracy.update_state(targets, predictions)
        
        return loss, self.train_accuracy.result()
    
    @tf.function
    def val_step(self, inputs, targets):
        """Single validation step"""
        predictions = self.model(inputs, training=False)
        loss = self.loss_fn(targets, predictions)
        
        self.val_loss.update_state(loss)
        self.val_accuracy.update_state(targets, predictions)
    
    def generate_epoch_visualizations(self, epoch):
        """Generate visualizations during training"""
        print(f"  📊 Generating visualizations for epoch {epoch}...")
        
        # Initialize visualizer if not done yet
        if self.visualizer is None:
            tokenizer_data = {
                'token_to_id': self.token_to_id_dict,
                'id_to_token': self.id_to_token_dict,
                'tokenizer_type': self.config['TOKENIZER_TYPE']
            }
            
            self.visualizer = GPTVisualizer(
                experiment_dir=self.experiment_dir,
                training_history=self.training_history,
                model=self.model,
                tokenizer_data=tokenizer_data
            )
        
        # Update visualizer with current training history
        self.visualizer.training_history = self.training_history
        
        # Generate sample text
        try:
            sample_text = "Alice was"
            generated = self.visualizer.generate_text_samples([sample_text])
            self.training_history['generation_samples'].append(generated)
        except Exception as e:
            print(f"    ⚠️ Text generation failed: {e}")
    
    def visualize_attention(self, epoch):
        """Generate attention visualization"""
        self.generate_epoch_visualizations(epoch)
        
    def generate_text_samples(self, epoch):
        """Generate text samples"""
        # Already handled in generate_epoch_visualizations
        pass
    
    def generate_final_visualizations(self):
        """Generate comprehensive final visualizations"""
        print("\n📊 Generating final visualizations...")
        
        # Initialize visualizer if not done yet
        if self.visualizer is None:
            tokenizer_data = {
                'token_to_id': self.token_to_id_dict,
                'id_to_token': self.id_to_token_dict,
                'tokenizer_type': self.config['TOKENIZER_TYPE']
            }
            
            self.visualizer = GPTVisualizer(
                experiment_dir=self.experiment_dir,
                training_history=self.training_history,
                model=self.model,
                tokenizer_data=tokenizer_data
            )
        
        # Update visualizer with final training history
        self.visualizer.training_history = self.training_history
        
        # Generate all visualizations
        try:
            self.visualizer.plot_training_curves()
            print("  ✅ Training curves saved")
        except Exception as e:
            print(f"  ⚠️ Training curves failed: {e}")
        
        try:
            self.visualizer.create_interactive_attention("Alice was walking")
            print("  ✅ Attention visualization saved")
        except Exception as e:
            print(f"  ⚠️ Attention visualization failed: {e}")
        
        try:
            self.visualizer.generate_text_samples(["Alice", "The cat", "Once upon"])
            print("  ✅ Text generation samples saved")
        except Exception as e:
            print(f"  ⚠️ Text generation failed: {e}")
    
    def plot_training_curves(self):
        """Plot training curves"""
        if self.visualizer:
            self.visualizer.plot_training_curves()
            
    def plot_learning_rate_schedule(self):
        """Plot learning rate schedule"""
        # Simple learning rate plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_history['epoch'], self.training_history['learning_rate'])
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{self.experiment_dir}/learning_rate_schedule.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_attention_dashboard(self):
        """Create attention dashboard"""
        if self.visualizer:
            self.visualizer.create_interactive_attention("Alice was walking")
    
    def create_generation_dashboard(self):
        """Create generation dashboard"""
        if self.visualizer:
            self.visualizer.generate_text_samples(["Alice", "The cat", "Once upon"])
    
    def visualize_model_architecture(self):
        """Visualize model architecture"""
        # Simple architecture summary
        arch_info = f"""
Model Architecture Summary:
- D_MODEL: {self.config['D_MODEL']}
- CONTEXT_LEN: {self.config['CONTEXT_LEN']}
- ATTENTION_HEADS: {self.config['ATTENTION_HEADS']}
- DECODER_BLOCKS: {self.config['DECODER_BLOCKS']}
- VOCAB_SIZE: {self.vocab_size}
- DROPOUT_RATE: {self.config['DROPOUT_RATE']}
"""
        
        with open(f"{self.experiment_dir}/model_architecture.txt", 'w') as f:
            f.write(arch_info)
        
        print(f"✅ All visualizations saved to {self.experiment_dir}/visualizations/")


def main():
    """Main function to run training with visualizations"""
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train GPT with visualizations')
    parser.add_argument('--config', type=str, default='configs/config.txt', help='Configuration file path')
    parser.add_argument('--text_file', type=str, default='/home/akshat/GPT_from_scratch/text_data/alice_extended.txt', help='Text file path')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (overrides config)')
    parser.add_argument('--experiment_name', type=str, default='gpt_viz_experiment', help='Experiment name')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = VisualizationTrainer(args.config, args.text_file, args.experiment_name)
    
    # Override epochs if specified
    if args.epochs is not None:
        trainer.config['EPOCHS'] = args.epochs
    
    try:
        # Setup
        trainer.setup_data_and_model()
        trainer.setup_training()
        
        # Train with visualizations
        trainer.train_with_visualizations()
        
        print(f"\n🎉 Training completed successfully!")
        print(f"📁 Results saved to: {trainer.experiment_dir}")
        
        # Ask user if they want to deploy immediately
        deploy_choice = input("\n🚀 Would you like to deploy the model now? (y/n): ").lower().strip()
        
        if deploy_choice in ['y', 'yes']:
            print("🌐 Starting deployment...")
            try:
                # Import deployment module
                deployment_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'deployment')
                sys.path.insert(0, deployment_path)
                import deploy
                
                print(f"🚀 Launching Gradio interface for: {trainer.experiment_dir}")
                deploy.launch_deployment(trainer.experiment_dir, share=False, port=7860)
                
            except Exception as e:
                print(f"❌ Deployment failed: {e}")
                print(f"📋 You can manually deploy later using:")
                print(f"    cd /home/akshat/GPT_from_scratch/cleaned_code")
                print(f"    python deployment/deploy.py")
        else:
            print(f"📋 To deploy later, run:")
            print(f"    cd /home/akshat/GPT_from_scratch/cleaned_code")
            print(f"    python deployment/deploy.py")
            print(f"    # Make sure to update the experiment_dir in deploy.py to:")
            print(f"    # {trainer.experiment_dir}")
        
        return trainer.experiment_dir
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    experiment_dir = main()
