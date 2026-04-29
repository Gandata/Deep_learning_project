import os
import torch
import numpy as np
import open_clip
from huggingface_hub import login
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def init_hf():
    token = os.getenv('HF_TOKEN')
    if token and token != 'your_huggingface_token_here':
        login(token=token)
    else:
        print('Warning: No valid HF_TOKEN found in .env. Downloading models may fail if they require authentication.')

def get_text_embedding(model, tokenizer, text: str, device: str = 'cpu') -> torch.Tensor:
    """Gets the CLIP text embedding for a single string."""
    text_tokens = tokenizer([text]).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features[0]

def get_class_embeddings(model, tokenizer, class_names: list, templates: list, device: str = 'cpu') -> dict:
    """Computes averaged CLIP embeddings for a list of classes using multiple templates."""
    class_embeddings = {}
    for class_name in class_names:
        embeddings = []
        for template in templates:
            text = template.format(class_name)
            emb = get_text_embedding(model, tokenizer, text, device)
            embeddings.append(emb)
        # Average the embeddings across templates
        avg_embedding = torch.stack(embeddings).mean(dim=0)
        avg_embedding /= avg_embedding.norm(dim=-1, keepdim=True)
        class_embeddings[class_name] = avg_embedding.cpu().numpy()
    return class_embeddings

if __name__ == '__main__':
    from pathlib import Path
    import sys
    
    # Add the root directory to sys.path so we can import src.dataset
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.dataset import LABEL_TEXT, NUM_CLASSES
    
    init_hf()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Loading CLIP model on {device}...')
    # Load from huggingface hub
    model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model.eval()
    
    templates = [
        'a photo of a {}',
        'a 3D point cloud of a {}',
        'a {}',
        'this is a {}',
        'an indoor scene with a {}'
    ]
    
    print('Computing CLIP embeddings for S3DIS classes...')
    # Extract class descriptions in order of their label IDs
    class_descriptions = [LABEL_TEXT[i] for i in range(NUM_CLASSES)]
    class_embeddings_dict = get_class_embeddings(model, tokenizer, class_descriptions, templates, device)
    
    # Convert to an array shape (NUM_CLASSES, 512)
    embeddings_array = np.zeros((NUM_CLASSES, 512), dtype=np.float32)
    for i, desc in enumerate(class_descriptions):
        embeddings_array[i] = class_embeddings_dict[desc]
        
    out_dir = Path('data/s3dis_processed')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / 'label_to_clip_embeddings.npy'
    np.save(out_file, embeddings_array)
    print(f'Saved CLIP embeddings to {out_file} (shape: {embeddings_array.shape})')
