# %%
!pip install snac
!pip install wandb
!pip install accelerate
import torch
import torchaudio
from snac import SNAC
!pip install datasets
from datasets import load_dataset, concatenate_datasets, Audio, load_from_disk, interleave_datasets

from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset

# %%
from huggingface_hub import login
login('')


# %%
dataone = load_from_disk('/kaggle/input/libri-tts-hf-10s/libritts.hf')
datatwo = load_from_disk('/kaggle/input/librispeech-hf-10s/hifitts.hf')
datathree = load_from_disk('/kaggle/input/hifi-tts-10s/hifitts.hf')
data_four = load_from_disk('/kaggle/input/globe-tts-25/globe25.hf')
data_five = load_from_disk('/kaggle/input/globe-2550-tts/globe25_50.hf')
data_six = load_from_disk('/kaggle/input/globe-50-75/globe50_75.hf')
data_seven = load_from_disk('/kaggle/input/globe-75-100tt/globe75:100_hf')

# %%

dataone = dataone.select_columns(['audio', 'text_normalized',])
dataone

# %%
datatwo  = concatenate_datasets([datatwo['train.100'], datatwo['train.360'], datatwo['test'], datatwo['validation'],])
datatwo = datatwo.rename_column("text", "text_normalized")
datatwo = datatwo.select_columns(['audio', 'text_normalized',])
datatwo

# %%
datathree = concatenate_datasets([datathree['train'], datathree['dev'], datathree['test'],])
datathree = datathree.select_columns(['audio', 'text_normalized',])
datathree

# %%
data_four = data_four.rename_column("transcript", "text_normalized")
data_four = data_four.select_columns(['audio', 'text_normalized',])

# %%
data_five = data_five.rename_column("transcript", "text_normalized")
data_five = data_five.select_columns(['audio', 'text_normalized',])

# %%
data_six = data_six.rename_column("transcript", "text_normalized")
data_six = data_six.select_columns(['audio', 'text_normalized',])

# %%
data_seven = data_seven.rename_column("transcript", "text_normalized")
data_seven = data_seven.select_columns(['audio', 'text_normalized',])

# %%
audio_data = interleave_datasets([dataone, datatwo, datathree, data_four, data_five, data_six, data_seven],stopping_strategy="all_exhausted", seed=42)

# %%
audio_data

# %%
# Step 2: Calculate split sizes
total_samples = len(audio_data)
split_size = total_samples // 12 # Integer division to get roughly equal splits

# Step 3: Split the dataset into 8 parts
dataset_splits = []
start_idx = 0
for i in range(11):  # Create 7 parts
    dataset_splits.append(audio_data.select(list(range(start_idx, start_idx + split_size))))
    start_idx += split_size
# Add the last part to ensure all examples are included
dataset_splits.append(audio_data.select(list(range(start_idx, total_samples))))

# %%


# %%
audio_data_part = dataset_splits[11] 

# %%
audio_data_part

# %%
audio_data_part.save_to_disk('picked_data')

# %%
audio_data_part = load_from_disk("/kaggle/working/picked_data")

# %%
!pip install librosa
import librosa


# use 🤗 Datasets' `filter` method to apply the filtering function
audio_data_part = audio_data_part.filter(lambda example: librosa.get_duration(y = example['audio']['array'], sr = 24000) <= 10 )



# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%

from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', add_bos_token = True, pad_token_id=50257)
config = GPT2Config(vocab_size=tokenizer.vocab_size, n_ctx=1024, n_layer = 20, n_embd = 1024, n_head = 16)
config.max_position_embeddings = 1024
model = GPT2LMHeadModel(config)

tokenizer.add_special_tokens({'pad_token': '[PAD]'}) ## add pad token which is [50257]
tokenizer.add_tokens(["SPACER"])
## resize embedding according to docs 
model.resize_token_embeddings(len(tokenizer))
model.to(device)



# %%
# Load the SNAC model
snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()

# %%
def flatten_tensors_adjusted(tensors):
    """Safely flattens a list of tensors into a flat list of integers."""
    flattened_list = []


    if len(tensors)==3:
      for i in range(tensors[0].size()[1]):
        flattened_list.append(50258)
        flattened_list.append(tensors[0][0][i].item())
        for j in range(2):
          flattened_list.append(tensors[1][0][j+i*2].item())
          for k in range(2):
            #print(k,i)
            flattened_list.append(tensors[2][0][k+j*2+i*4].item())

    if len(tensors)==4:
      for i in range(tensors[0].size()[1]):
        flattened_list.append(50258)
        flattened_list.append(tensors[0][0][i].item())
        for j in range(2):
          flattened_list.append(tensors[1][0][j+i*2].item())
          for k in range(2):
            #print(k,i)
            flattened_list.append(tensors[2][0][k+j*2+i*4].item())
            for l in range(2):

             flattened_list.append(tensors[3][0][l+k*2+j*4+i*8].item())

    return flattened_list

# %%
class CustomDataSet(Dataset):
    def __init__(self, dataset, model):
        self.ds = dataset
        self.model = model.to(device)
       
        super().__init__()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        
        codes = None

       
        audio = torch.tensor(self.ds[idx]["audio"]["array"], dtype=torch.float32).to(device)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        

        # Convert to mono by averaging the channels if the audio is stereo
        if audio.size(0) > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
            
        audio = torch.unsqueeze(audio, 0)
        
        with torch.inference_mode():
            audio_hat, codes = self.model(audio)

#         # Confirm audio is in the shape [1, 1, T] where T is the sequence length
#         print("Audio size after processing:", audio.size(), audio.shape)
        
   
        text = self.ds[idx]['text_normalized']
       
    
        
    
      
        
        text_tensor = tokenizer(text)
        text_attention_mask = text_tensor['attention_mask']
        
        audio_codes = flatten_tensors_adjusted(codes)
        output_tokens = text_tensor['input_ids'] + audio_codes + [50256]
        attention = [1] * len(output_tokens)
       
        
        pad_tokens = [50257] * (1024 - len(output_tokens))
        attention_pad = [0] * len(pad_tokens)
        
        
        output_tokens = output_tokens + pad_tokens
        attention = attention + attention_pad
        text_attention_mask = text_attention_mask + ([0] * (1023 - len(text_attention_mask) )) 
        
        output_tokens  = torch.as_tensor(output_tokens)
        attention  = torch.as_tensor(attention) 
        text_attention_mask = torch.as_tensor(text_attention_mask)
        
                # Define token type weights
        text_token_weight = 0.01
        other_token_weight = 1.0

        # Create weights tensor
        weights = torch.where(text_attention_mask == 1, text_token_weight, other_token_weight)
        
#            # Debugging prints
#         print(f"output_tokens: {output_tokens}, dtype: {output_tokens.dtype}")
#         print(f"attention: {attention}, dtype: {attention.dtype}")
        
        
       

        return {'input_ids': output_tokens, 'attention_mask': attention , "text_attention_mask": text_attention_mask, 'weights': weights, }

# %%
batch_size = 1
learning_rate = 1e-4
n_layers = 12

# %%
max_position_embeddings = model.config.max_position_embeddings
max_position_embeddings

# %%
from tqdm import tqdm

# %%

import wandb
wandb.login(key = '')
wandb.init(project="text to speech", name="SNAC GPT_2-36th epoch-pass3", id = 'nec68627', resume = 'must',  )
    # Initialize WandB configuration

wandb.config.batch_size = batch_size
wandb.config.learning_rate = 0.0001 #actually 1e-7

wandb.config.n_layers = n_layers

# %%
continue_training = True
if continue_training:
    checkpoint = torch.load("/kaggle/input/big-snac-pass3-b/tts_transformer-epoch-36.pt",map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     epoch_start = checkpoint['epoch']
#     loss = checkpoint['loss']

# %%
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# %%
def find_last_instance_of_seperator(lst, element=50256):
    reversed_list = lst[::-1]
    try:
        reversed_index = reversed_list.index(element)
        return len(lst) - 1 - reversed_index
    except ValueError:
        raise ValueError


# %%
def reconstruct_tensors(flattened_output):
    """Reconstructs the list of tensors from the flattened output."""

    def count_elements_between_hashes(lst):
        try:
            
            # Find the index of the first '#'
            first_index = lst.index(50258)
            
            # Find the index of the second '#' after the first
            second_index = lst.index(50258, first_index + 1)
            # Count the elements between the two indices
            return second_index - first_index - 1
        except ValueError:
            # Handle the case where there aren't enough '#' symbols
            return "List does not contain two '#' symbols"


    def remove_elements_before_hash(flattened_list):
        
        try:
            # Find the index of the first '#'
            first_hash_index = flattened_list.index(50258)
              # Return the list starting from the first '#'
            return flattened_list[first_hash_index:]
        except ValueError:
              # Handle the case where there is no '#'
            return "List does not contain the symbol '#'"


    def list_to_torch_tensor(tensor1):
        # Convert the list to a torch tensor
        tensor = torch.tensor(tensor1)
        # Reshape the tensor to have size (1, n)
        tensor = tensor.unsqueeze(0)
        return tensor
    
    flattened_output= remove_elements_before_hash(flattened_output)
    last_index = find_last_instance_of_seperator(flattened_output)
    flattened_output = flattened_output[:last_index]
       
    print(flattened_output)
    codes = []
    tensor1=[]
    tensor2=[]
    tensor3=[]
    tensor4=[]

    n_tensors= count_elements_between_hashes(flattened_output)
    print("n_tensors:", n_tensors)
    if n_tensors==7:
      for i in range(0,len(flattened_output),8):

        
        tensor1.append(flattened_output[i+1])
        tensor2.append(flattened_output[i+2])
        tensor3.append(flattened_output[i+3])
        tensor3.append(flattened_output[i+4])

        tensor2.append(flattened_output[i+5])
        tensor3.append(flattened_output[i+6])
        tensor3.append(flattened_output[i+7])
        codes=[list_to_torch_tensor(tensor1).to(device),list_to_torch_tensor(tensor2).to(device),list_to_torch_tensor(tensor3).to(device) ]


    if n_tensors==15:
      for i in range(0,len(flattened_output),16):

        tensor1.append(flattened_output[i+1])
        tensor2.append(flattened_output[i+2])
        tensor3.append(flattened_output[i+3])
        tensor4.append(flattened_output[i+4])
        tensor4.append(flattened_output[i+5])
        tensor3.append(flattened_output[i+6])
        tensor4.append(flattened_output[i+7])
        tensor4.append(flattened_output[i+8])

        tensor2.append(flattened_output[i+9])
        tensor3.append(flattened_output[i+10])
        tensor4.append(flattened_output[i+11])
        tensor4.append(flattened_output[i+12])
        tensor3.append(flattened_output[i+13])
        tensor4.append(flattened_output[i+14])
        tensor4.append(flattened_output[i+15])

        codes=[list_to_torch_tensor(tensor1).to(device), list_to_torch_tensor(tensor2).to(device),list_to_torch_tensor(tensor3).to(device),list_to_torch_tensor(tensor4).to(device) ]

    return codes

# %%

import torch

# data = CustomDataSet(audio_data,snac_model)


# train_size = int(0.96 * len(audio_data))
# val_size = len(audio_data) - train_size
# train_dataset, val_dataset = torch.utils.data.random_split(audio_data, [train_size, val_size])
train_ds = CustomDataSet(audio_data_part,snac_model)
    # val_ds = CustomDataSet(val_dataset,snac_model)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
def training(model):




    optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
    # Learning Rate Scheduler: Cosine Annealing with Warm Restarts
    T_0 = 500  # Number of steps for the first restart (warm-up steps)
    T_mult = 1  # A factor increases T_i after a restart
    eta_min = 1e-7  # Minimum learning rate

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
    epoch_start = 1
    global_step = 0

 


    continue_training = True
    if continue_training:
        checkpoint = torch.load('/kaggle/input/big-snac-pass3-b/tts_transformer-epoch-35.pt')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        global_step = checkpoint['global_step']




    model.train()
    global_step = global_step
    accumulation_steps = 32
    current_loss = 0
    total_loss = 0

    for epoch in range(epoch_start, 100):
        batch_iterator = tqdm(train_loader, desc=f"Processing Epoch {epoch:02d}")
        optimizer.zero_grad()

        for i, batch in enumerate(batch_iterator):
            input_ids = batch['input_ids'].to(device).long()
            attention_mask = batch['attention_mask'].to(device).long()
            text_attention_mask  = batch['text_attention_mask'].to(device)
            weights = batch['weights'].to(device)



            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            logits = outputs.logits

    
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_attention_mask = attention_mask[..., 1:].contiguous()

            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss(reduction = 'none',ignore_index=50257)

            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

#                         # Apply weights
            
            loss = loss * weights.view(-1)
            
#             weighted_loss = weighted_loss.mean()
             # Compute masked mean
          
            total_tokens = shift_attention_mask.sum()
            loss = loss.sum() / (total_tokens + 1e-8)
                     # Scale loss by accumulation steps
            loss = loss / accumulation_steps
            
            loss.backward()
            
          
           
            

            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
                global_step += 1
                
                
                # Get and format the learning rate
                lr_rate = scheduler.get_last_lr()[0]
                batch_iterator.set_postfix({"loss": f"{loss.item()*accumulation_steps}", 'LR' :f"{lr_rate:.2e}"})  # Multiply by accumulation steps to show actual loss
                wandb.log({"Training Loss": loss.item()*accumulation_steps, "Global Step": global_step,  'LR' : lr_rate})
                
               


        torch.save({
                       'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict':scheduler.state_dict(),
                        'loss': loss,
                        'global_step':global_step
                  },f'/kaggle/working/tts_transformer-epoch-{epoch}.pt')
                # Define the input text
        for j in range(10):    
            input_text = "sixteen or seventeen, i should say, replied another voice."

                    # Tokenize the input text
            input_ids = tokenizer(input_text, return_tensors='pt').to(device)


                    # Generate text
            with torch.no_grad():
                output_ids = model.generate(input_ids['input_ids'],attention_mask=input_ids['attention_mask'],max_length = 1024, 
                num_beams = 4,
                top_p=0.95,
                temperature=0.8,
                do_sample = True,                           
                repetition_penalty=2.0)
                print(output_ids) 
                print(output_ids.shape)
                reconstructed_codes = reconstruct_tensors(output_ids.squeeze(0).tolist())
                import soundfile as sf
                audio_hat = snac_model.to(device).decode(reconstructed_codes)
                sf.write(f"/kaggle/working/reconstructed_audio_{j}.wav", audio_hat.squeeze().cpu().numpy(), 24000)
        for j in range(10, 20):    
            input_text = "misses march, had agreed to the visit rather slowly fearing that margaret would come back more lucky than she went."

                    # Tokenize the input text
            input_ids = tokenizer(input_text, return_tensors='pt').to(device)


                    # Generate text
            with torch.no_grad():
                output_ids = model.generate(input_ids['input_ids'],attention_mask=input_ids['attention_mask'],max_length = 1024, 
#                 top_k=50,
                top_p=0.95,
                num_beams = 4,                            
                temperature=0.9,
                do_sample = True,                           
                repetition_penalty=2.0)
                print(output_ids) 
                print(output_ids.shape)
                reconstructed_codes = reconstruct_tensors(output_ids.squeeze(0).tolist())
                import soundfile as sf
                audio_hat = snac_model.to(device).decode(reconstructed_codes)
                sf.write(f"/kaggle/working/reconstructed_audio_{j}.wav", audio_hat.squeeze().cpu().numpy(), 24000)
                model.train()
#                 model.train()


        # Validation
    #     model.eval()
    #     val_loss = 0
    #     with torch.no_grad():
    #         for batch in val_loader:
    #             inputs, masks = batch
    #             outputs = model(input_ids=inputs, attention_mask=masks, labels=inputs)
    #             logits = outputs.logits

    #             # Shift so that tokens < n predict n
    #             shift_logits = logits[..., :-1, :].contiguous()
    #             shift_labels = inputs[..., 1:].contiguous()

    #             # Flatten the tokens
    #             loss_fct = torch.nn.CrossEntropyLoss(ignore_index=50257)
    #             loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    #             val_loss += loss.item()

    #     val_loss /= len(val_loader)
    #     print(f'Epoch {epoch+1}, Validation Loss: {val_loss}')
    #     model.train()


# %%
training(model)
    
    

# %%
from IPython.display import Audio

# for i in range(10):
#     with torch.no_grad():
#         input_text = "in a land far way, there was a woman named Mary."
#         input_ids = tokenizer(input_text, return_tensors='pt').to(device)
#         output_ids = model.generate(input_ids['input_ids'],attention_mask=input_ids['attention_mask'],max_length = 1024,    top_k=50,
#         top_p=0.95,
#         temperature=0.8,
#         do_sample = True,                           
#         repetition_penalty=2.0)

# #         print(output_ids) 
# #         print(output_ids.shape)
#         print(i)
#         reconstructed_codes = reconstruct_tensors(output_ids.squeeze(0).tolist())
#         import soundfile as sf
#         audio_hat = snac_model.to(device).decode(reconstructed_codes)
#         audio_path = f"reconstructed_audio_{i}.wav"
#         sf.write(audio_path, audio_hat.squeeze().cpu().detach().numpy(), 24000)

#         # Display and play the audio
#         display(Audio(audio_path))
for i in range(10,20):
    with torch.no_grad():
        input_text = "for i have heard what he has heard, and i have seen what he has seen?."
        input_ids = tokenizer(input_text, return_tensors='pt').to(device)
        output_ids = model.generate(input_ids['input_ids'],attention_mask=input_ids['attention_mask'],max_length = 1024,  
#         top_k=50,
        num_beams=4,
        top_p=0.95,
        temperature=0.8,
        do_sample = True,                           
        repetition_penalty=2.0)

#         print(output_ids) 
#         print(output_ids.shape)
        print(i)
        reconstructed_codes = reconstruct_tensors(output_ids.squeeze(0).tolist())
        import soundfile as sf
        audio_hat = snac_model.to(device).decode(reconstructed_codes)
        audio_path = f"reconstructed_audio_{i}.wav"
        sf.write(audio_path, audio_hat.squeeze().cpu().detach().numpy(), 24000)

        # Display and play the audio
        display(Audio(audio_path))
        

# %%
input_text = "in being modern"

#             # Tokenize the input text
# tokenizer.pad_token_id = 50257   
output_ids = None
input_ids = tokenizer(input_text, return_tensors='pt').to(device)
input_ids['input_ids'] = torch.cat((input_ids['input_ids'], torch.tensor([[50258]]), ), dim =1)
input_ids['attention_mask'] = torch.cat((input_ids['attention_mask'], torch.tensor([[1]]), ), dim =1)
print(input_ids['input_ids'])
print(tokenizer.pad_token_id)
with torch.no_grad():
    output_ids = model.generate(input_ids['input_ids'],attention_mask=input_ids['attention_mask'],max_length = 1024, )
    print(output_ids[0][0:100]) 
    reconstructed_codes = reconstruct_tensors(output_ids.squeeze(0).tolist())
    import soundfile as sf
#     with torch.inference_mode():

    audio_hat = snac_model.to(device).decode(reconstructed_codes)

# Save the reconstructed audio file
# First, move the tensor to CPU, then convert to NumPy array
    sf.write('/kaggle/working/reconstructed_audio1.wav', audio_hat.squeeze().cpu().numpy(), 24000)
#         model.train

# %%

reconstructed_codes = reconstruct_tensors(output_ids.squeeze(0).tolist())


