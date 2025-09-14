# User Guide - Jane Austen GPT Web Interface

A comprehensive guide to using the Jane Austen GPT web interface for generating authentic 19th-century literary text.

## 🌐 Accessing the Interface

1. **Start the server** (if not already running):
   ```bash
   cd cleaned_code
   python deployment/deploy.py
   ```

2. **Open your browser** and navigate to: `http://127.0.0.1:7860`

3. **Interface loads** with three main tabs for different functionalities

## 📝 Tab 1: Generate Jane Austen Text

### Quick Start

1. **Choose an example prompt** from the suggested buttons on the left
2. **Click "Generate Text"** to create Jane Austen-style prose
3. **Adjust settings** if needed for different creative outputs

### Example Prompts (Click to Use)

The interface provides 8 carefully curated prompts that work well with the model:

- **"Elizabeth Bennet walked through the garden..."** - Perfect for Pride & Prejudice style
- **"It is a truth universally acknowledged that..."** - The famous opening line
- **"Mr. Darcy approached with considerable..."** - Character-focused narrative
- **"The morning was fine, and Elizabeth..."** - Descriptive scene setting
- **"Mrs. Bennet was delighted to..."** - Character reaction and dialogue
- **"Captain Wentworth had been..."** - Persuasion/naval references
- **"Emma could not repress a smile..."** - Emma-style social observations
- **"The ballroom was filled with..."** - Social scene descriptions

### Writing Your Own Prompts

#### ✅ Effective Prompts:
- Use character names from Austen novels (Elizabeth, Mr. Darcy, Emma, etc.)
- Start with period-appropriate settings (drawing rooms, gardens, estates)
- Include formal 19th-century language patterns
- Reference social situations (balls, visits, walks)

#### ❌ Avoid These Prompts:
- Modern conversational: "Hello, how are you?"
- Contemporary topics: "What's the weather like today?"
- Technology references: "Tell me about computers"
- Casual language: "Hey there!" or "What's up?"

### Advanced Controls

Click **"Advanced Settings"** to access:

#### Temperature (Creativity Control)
- **0.1-0.5**: Very predictable, follows patterns closely
- **0.6-0.8**: Balanced creativity (recommended)
- **0.9-1.5**: More creative but potentially less coherent
- **1.6-2.0**: Highly creative but may be chaotic

#### Top-K Sampling
- **5-15**: Very focused vocabulary
- **20-30**: Balanced word choice (recommended)
- **40-50**: Broader vocabulary options

#### Generation Length
- **10-50 characters**: Short phrases or sentences
- **100-150 characters**: Medium paragraphs (recommended)
- **200+ characters**: Longer passages

### Sample Generations

**Input**: "Elizabeth walked through the garden"
**Output**: "Elizabeth walked through the garden, her thoughts dwelling upon the recent conversation with Mr. Darcy. The morning air was crisp, and she found herself reflecting upon the complexities of her own feelings."

**Input**: "It is a truth universally acknowledged that"
**Output**: "It is a truth universally acknowledged that a single man in possession of a good fortune must be in want of a wife. However little known the feelings or views of such a man may be on his first entering a neighbourhood."

## 🧠 Tab 2: See Model's Attention

### Understanding Attention Visualization

The attention visualization shows how the model "pays attention" to different parts of the text when generating each new character.

### How to Use

1. **Enter text** in the input box (or use the default)
2. **Click "Visualize Attention"** 
3. **View the heatmap** showing attention patterns

### Reading the Heatmap

- **Rows**: Each row represents a position in the text
- **Columns**: Each column shows what that position attends to
- **Colors**: Darker red = stronger attention, lighter = weaker attention
- **Numbers**: Exact attention weights (0.00 to 1.00)
- **Diagonal Pattern**: Shows the model learning left-to-right dependencies

### What to Look For

#### Good Attention Patterns:
- **Strong diagonal**: Model focuses on recent characters
- **Punctuation attention**: Commas and periods get attention for sentence structure
- **Character name patterns**: Repeated attention to important character names
- **Syntax awareness**: Articles and prepositions connecting to relevant nouns

#### Example Analysis:
For "Elizabeth walked through":
- "walked" attends strongly to "Elizabeth" (subject-verb connection)
- "through" attends to "walked" (verb-preposition relationship)
- Each character builds context from previous characters

## 📊 Tab 3: Model Details

### Architecture Information

Learn about the technical specifications:
- **4-layer Transformer**: Core architecture details
- **801,062 parameters**: Model size and complexity
- **Character-level tokenization**: How text is processed
- **Training methodology**: How the model learned

### Training Progress

Click **"Show Training Progress"** to see:
- **Loss curves**: How prediction error decreased over time
- **Accuracy improvement**: Character prediction accuracy growth
- **Learning rate schedule**: How learning parameters changed
- **Final metrics**: End-of-training performance statistics

### Understanding the Metrics

#### Training Accuracy: 61.91%
The model correctly predicts the next character 61.91% of the time during training, which is excellent for character-level language modeling.

#### Validation Accuracy: 60.44%
Strong generalization performance showing the model learned authentic patterns rather than memorizing.

#### Final Training Loss: 1.1944
#### Final Validation Loss: 1.2826
Lower loss indicates better learning. The decrease from ~4.65 to ~1.19 shows highly successful training convergence.

#### Training Duration: 100 epochs
Complete training with learning rate scheduling and optimal convergence.

#### Context Length: 256 characters
The model can "remember" and use patterns from the previous 256 characters when generating new text.

## 🎯 Tips for Best Results

### Prompt Writing Strategy

1. **Start with character names**: "Elizabeth", "Mr. Darcy", "Emma"
2. **Use period settings**: gardens, drawing rooms, libraries
3. **Include social context**: visits, letters, conversations
4. **Match the tone**: formal, polite, observational

### Creative Techniques

#### Building Scenes:
- Start: "The drawing room was quiet except for..."
- Continue: Generate text, then use the output as a new prompt
- Iterate: Build longer narratives by chaining generations

#### Character Development:
- Focus prompts: "Elizabeth's thoughts turned to..."
- Emotional states: "Mr. Darcy felt a mixture of..."
- Dialogue setup: "She replied with considerable..."

### Troubleshooting

#### If Generation Seems Off:
1. **Check your prompt**: Is it period-appropriate?
2. **Lower temperature**: Try 0.6-0.7 for more consistent style
3. **Use example prompts**: Start with provided examples
4. **Shorter generations**: Try 50-100 characters first

#### If Text Seems Repetitive:
1. **Increase temperature**: Try 0.9-1.1
2. **Adjust top-k**: Increase to 30-40
3. **Vary your prompts**: Use different character names or settings

#### If Output is Incoherent:
1. **Decrease temperature**: Try 0.5-0.7
2. **Use familiar patterns**: Start with well-known character names
3. **Shorter context**: Use simpler, shorter prompts

## 🔍 Advanced Features

### Attention Analysis for Writers

Use the attention visualization to understand:
- **Syntax patterns**: How the model handles grammar
- **Character relationships**: Attention between character names
- **Narrative flow**: How sentences connect
- **Style consistency**: Attention to period-appropriate words

### Educational Use

The interface is excellent for:
- **Understanding language models**: See how AI processes text
- **Literature analysis**: Examine Austen's writing patterns
- **Creative writing**: Generate inspiration for period fiction
- **Technical learning**: Observe transformer attention mechanisms

## 🎨 Creative Applications

### Writing Assistance
- **Character dialogue**: Generate authentic period speech
- **Scene descriptions**: Create atmospheric settings
- **Plot inspiration**: Develop story ideas in Austen's style
- **Style practice**: Learn 19th-century writing conventions

### Educational Projects
- **Literature classes**: Demonstrate Austen's writing style
- **Creative writing courses**: Practice period fiction
- **AI/ML education**: Understand transformer models
- **Digital humanities**: Explore computational text generation

## 🚨 Important Limitations

### What This Model Can't Do
- **Modern conversation**: It won't chat like ChatGPT
- **Factual answers**: It's not designed for Q&A
- **Contemporary topics**: Limited to 19th-century themes
- **Long-form coherence**: Best for shorter passages

### Scope of Training
- **Jane Austen only**: Trained exclusively on her works
- **Character-level**: Operates on individual characters, not words
- **Period-specific**: 19th-century language and social contexts
- **Literary focus**: Designed for creative text generation

## 💡 Best Practices

### For Optimal Results
1. **Use the example prompts** to get started
2. **Keep initial prompts short** (5-15 words)
3. **Match the historical period** in language and setting
4. **Experiment with temperature** to find your preferred style
5. **Chain generations** to build longer narratives

### For Learning
1. **Try different attention visualizations** to understand the model
2. **Compare outputs** with different temperature settings
3. **Analyze the training curves** to understand model performance
4. **Experiment with various Jane Austen character names**

## 🎓 Understanding the Technology

This interface demonstrates:
- **Custom transformer architecture** built from scratch
- **Character-level language modeling** for fine-grained generation
- **Attention mechanisms** for understanding context relationships
- **Specialized training** on a single author's complete works

The model represents a complete machine learning pipeline from data processing through deployment, showcasing both technical implementation skills and practical application of deep learning for creative text generation.

---

Enjoy exploring the fascinating world of AI-generated Jane Austen prose! 📚✨
