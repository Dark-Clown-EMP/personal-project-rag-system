import os
import shutil
import glob
import nest_asyncio
from dotenv import load_dotenv

from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# Fix for async loops
nest_asyncio.apply()
load_dotenv()

# --- CONFIG ---
DATA_DIR = "data"
PERSIST_DIR = "storage"
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

if not LLAMA_CLOUD_API_KEY:
    raise ValueError("❌ LLAMA_CLOUD_API_KEY missing! Add it to your .env file.")

def get_parser():
    """Returns the LlamaParse object (The 'Resume Star' for tables)"""
    return LlamaParse(
        result_type="markdown",
        verbose=True,
        language="en",
        api_key=LLAMA_CLOUD_API_KEY
    )

def ingest():
    print("--- 1. SETUP ---")
    
    # 1. Clean Slate: Delete old storage to avoid duplicates
    if os.path.exists(PERSIST_DIR):
        print(f"🧹 Deleting old storage at {PERSIST_DIR}...")
        shutil.rmtree(PERSIST_DIR)

    # 2. Setup Embedding Model (Local BGE-Small)
    print("⚙️  Loading Embedding Model...")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.embed_model = embed_model
    
    # 3. Setup Parent-Child Node Parser
    # Splits into sentences, hides surrounding context in metadata
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    # 4. Initialize Vector Store (ChromaDB or Simple)
    # We use a persistent index that updates on disk after every batch
    index = None
    
    # Get all PDF files
    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    total_files = len(pdf_files)
    print(f"📂 Found {total_files} PDFs to process.")

    # --- BATCH PROCESSING LOOP ---
    for i, pdf_path in enumerate(pdf_files):
        filename = os.path.basename(pdf_path)
        print(f"\n📄 [File {i+1}/{total_files}] Processing: {filename}...")
        
        try:
            # A. Parse THIS file only (Saves RAM)
            parser = get_parser()
            file_extractor = {".pdf": parser}
            
            documents = SimpleDirectoryReader(
                input_files=[pdf_path], 
                file_extractor=file_extractor
            ).load_data()
            print(f"   ✅ Parsed {len(documents)} pages.")

            # B. Create Nodes (Sentence Window)
            nodes = node_parser.get_nodes_from_documents(documents)
            print(f"   ✅ Created {len(nodes)} nodes.")

            # C. Insert into Index
            if index is None:
                # First batch: Create the index
                print("   🔨 Creating new Vector Index...")
                index = VectorStoreIndex(nodes, embed_model=embed_model)
            else:
                # Subsequent batches: Add to existing index
                print("   ➕ Adding to existing Index...")
                index.insert_nodes(nodes)

            # D. Persist Immediately (Safety Checkpoint)
            # If the next file crashes, we don't lose previous progress
            index.storage_context.persist(persist_dir=PERSIST_DIR)
            print(f"   💾 Saved progress to {PERSIST_DIR}")

        except Exception as e:
            print(f"   ❌ ERROR processing {filename}: {e}")
            print("   ⚠️ Skipping this file and continuing...")

    print("\n🎉 INGESTION COMPLETE!")
    print(f"All valid files saved to '{PERSIST_DIR}'. Ready for Backend.")

if __name__ == "__main__":
    ingest()