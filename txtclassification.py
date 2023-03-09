# Import required libraries
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load dataset
df = pd.read_csv('bbc-news.csv')

# Preprocessing
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_text(text):
    # Remove punctuation and special characters
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize text
    tokens = tokenizer.tokenize(text)
    
    # Remove stop words
    tokens = [token for token in tokens if token not in tokenizer.get_vocab().keys()]
    
    # Add padding
    if len(tokens) < max_len:
        tokens += ['[PAD]'] * (max_len - len(tokens))
        
    return tokens[:max_len]

# Convert labels to numeric values
label_map = {'business': 0, 'entertainment': 1, 'politics': 2, 'sport': 3, 'tech': 4}
df['label'] = df['category'].map(label_map)

# Split dataset into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Prepare data for model input
max_len = 512

train_tokens = train_df['text'].apply(preprocess_text).tolist()
test_tokens = test_df['text'].apply(preprocess_text).tolist()

train_labels = train_df['label'].tolist()
test_labels = test_df['label'].tolist()

train_inputs = tokenizer.batch_encode_plus(train_tokens, pad_to_max_length=True, max_length=max_len, truncation=True, return_tensors='pt')
test_inputs = tokenizer.batch_encode_plus(test_tokens, pad_to_max_length=True, max_length=max_len, truncation=True, return_tensors='pt')

train_dataset = TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], torch.tensor(train_labels))
test_dataset = TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'], torch.tensor(test_labels))

train_dataloader = DataLoader(train_dataset, batch_size=16)
test_dataloader = DataLoader(test_dataset, batch_size=16)

# Instantiate BERT model and optimizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)
optimizer = AdamW(model.parameters(), lr=2e-5)

# Train model
epochs = 5

for epoch in range(epochs):
    model.train()
    
    for batch in train_dataloader:
        input_ids = batch[0]
        attention_mask = batch[1]
        labels = batch[2]
        
        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        
    model.eval()
    
    with torch.no_grad():
        test_loss = 0
        test_preds = []
        test_labels = []

        for batch in test_dataloader:
            input_ids = batch[0]
            attention_mask = batch[1]
            labels = batch[2]

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            test_loss += outputs.loss.item()

            test_preds += torch.argmax(outputs.logits, axis=1).tolist()
            test_labels
