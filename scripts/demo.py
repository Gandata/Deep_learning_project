import argparse
import numpy as np
import torch
import plotly.graph_objects as go
from pathlib import Path
import sys

# Allow importing src
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.encoder import ConcertoEncoder
from src.translation_head import MLPTranslationHead
import open_clip
from src.visualize import plot_heatmap, save_figure

DEMO_QUERY_COLORSCALE = [
    [0.00, "rgb(20,120,255)"],
    [0.25, "rgb(110,220,255)"],
    [0.50, "rgb(250,235,170)"],
    [0.75, "rgb(245,125,70)"],
    [1.00, "rgb(210,45,45)"],
]

def query_scene(coord, features_clip, text_query, clip_model, tokenizer, device, top_percent=10.0):
    print(f'\nQuery: "{text_query}"')

    # 1. Embed the text query with CLIP
    tokens = tokenizer([text_query]).to(device)
    with torch.no_grad():
        text_emb = clip_model.encode_text(tokens)           # (1, 512)
        text_emb = torch.nn.functional.normalize(text_emb, dim=-1)

    # 2. Cosine similarity: (N, 512) · (512, 1) → (N,)
    similarity = (features_clip @ text_emb.T).squeeze(-1)  # (N,)
    similarity = similarity.cpu().numpy()

    # 3. Normalise similarity to [0, 1] for visualisation
    sim_min, sim_max = similarity.min(), similarity.max()
    sim_norm = (similarity - sim_min) / (sim_max - sim_min + 1e-8)

    print(f'Similarity range: {sim_min:.3f} to {sim_max:.3f}')
    print(f'Highlighting top {top_percent}% of points')

    # 4. Visualise heatmap using src.visualize
    fig = plot_heatmap(
        points=coord,
        scores=sim_norm,
        query_text=text_query,
        top_percent=top_percent,
        colorscale=DEMO_QUERY_COLORSCALE,
        show_colorbar=True,
        reverse_colorbar=False,
    )
    return fig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--polycam_dir", type=str, required=True, help="Directory with coord.npy, color.npy, normal.npy")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained MLP head checkpoint")
    parser.add_argument("--query", type=str, default="chair", help="Text query to search for")
    parser.add_argument("--top_percent", type=float, default=10.0, help="Percentage of top similar points to highlight")
    parser.add_argument("--output_html", type=str, default=None, help="If set, save the Plotly visualization to this HTML file")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g. 'cuda' or 'cpu')")
    args = parser.parse_args()

    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Polycam scan
    poly_dir = Path(args.polycam_dir)
    if not poly_dir.exists():
        print(f"Error: Polycam directory {poly_dir} not found.")
        sys.exit(1)

    coord = np.load(poly_dir / 'coord.npy')
    color = np.load(poly_dir / 'color.npy')
    normal = np.load(poly_dir / 'normal.npy')
    
    print(f"Loaded Polycam scan from {poly_dir}: {len(coord):,} points")

    # 2. Extract Concerto Features
    print("Loading Concerto encoder...")
    encoder = ConcertoEncoder(device=device)
    encoder.eval()
    
    encoder_input = {
        'coord': coord,
        'color': color,
        'normal': normal,
    }
    
    print("Extracting 3D features...")
    with torch.no_grad():
        features_3d = encoder(encoder_input)

    # 3. Apply MLP Translation Head
    print("Loading MLP Translation Head...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    if "config" in checkpoint:
        model_cfg = checkpoint["config"].get("model", {})
        mlp = MLPTranslationHead(
            input_dim=model_cfg.get("input_dim", 256),
            hidden_dims=model_cfg.get("hidden_dims", [512, 512]),
            output_dim=model_cfg.get("output_dim", 512),
            dropout=model_cfg.get("dropout", 0.1),
            activation=model_cfg.get("activation", "gelu"),
            normalize_output=model_cfg.get("normalize_output", True),
        )
    else:
        mlp = MLPTranslationHead(input_dim=256, hidden_dims=[512, 512], output_dim=512)
        
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    mlp.load_state_dict(state_dict)
    mlp = mlp.to(device)
    mlp.eval()
    
    print("Translating features into CLIP space...")
    with torch.no_grad():
        features_clip = mlp(features_3d)
        features_clip = torch.nn.functional.normalize(features_clip, dim=-1)

    # 4. Text Query
    print("Loading CLIP text encoder...")
    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    clip_model = clip_model.to(device)
    clip_model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    fig = query_scene(coord, features_clip, args.query, clip_model, tokenizer, device, args.top_percent)
    
    if args.output_html:
        save_figure(fig, args.output_html, save_png=False)
        print(f"Saved visualization to {args.output_html}")
    else:
        fig.show()

if __name__ == "__main__":
    main()
