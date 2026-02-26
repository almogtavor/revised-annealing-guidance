from img2dataset import download
from datasets import load_dataset
import os
import pandas as pd


def main():
    csv_path = "./src/data/laion/laion_pop_26k.csv"

    # Only stream+filter if CSV doesn't already exist
    if not os.path.exists(csv_path):
        print("=" * 80)
        print("Loading LAION-POP dataset from HuggingFace...")
        print("LAION-POP: 600k high-quality images with detailed captions")
        print("Taking top 26k URLs to get ~20k actual images (accounting for ~78% success rate)")
        print("=" * 80)
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

        # Load LAION-POP (600k images)
        dataset = load_dataset("laion/laion-pop", split="train", token=hf_token)

        print(f"Loaded {len(dataset):,} total images from LAION-POP")

        # Convert to DataFrame for sorting
        # LAION-POP has columns: URL, TEXT (caption), and quality scores
        print("Converting to DataFrame...")
        df = pd.DataFrame(dataset)

        # Take top 20k based on quality scores
        # Sort by aesthetic/similarity if available, otherwise just take first 20k
        if 'aesthetic' in df.columns and 'similarity' in df.columns:
            print("Sorting by quality scores...")
            df_sorted = df.sort_values(by=["aesthetic", "similarity"], ascending=False)
        else:
            print("No quality scores found, using dataset order...")
            df_sorted = df

        df_top26k = df_sorted.head(26000)

        # Save to CSV
        print(f"Saving top {len(df_top26k):,} URLs to {csv_path}...")
        df_top26k.to_csv(csv_path, index=False)
        print(f"CSV saved successfully! (Targeting 20k actual downloads)")
    else:
        print(f"CSV already exists at {csv_path}, skipping metadata download.")
        df = pd.read_csv(csv_path)
        print(f"CSV contains {len(df):,} URLs")

    print("=" * 80)
    print("Downloading images via img2dataset...")
    print("=" * 80)
    download(
        url_list=csv_path,
        input_format="csv",
        url_col="url",
        caption_col="cogvlm_caption",
        output_format="files",
        output_folder="./src/data/laion/laion_pop_images",
        processes_count=8,
        thread_count=32,
        resize_mode="no"
    )


if __name__ == '__main__':
    main()
