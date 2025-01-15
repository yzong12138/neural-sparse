import os
import shutil

def sync_files(output_dir, target_dir, allowed_extensions, blocklist_files=None):
    if blocklist_files is None:
        blocklist_files = []

    allowed_extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in allowed_extensions]

    for root, dirs, files in os.walk(output_dir):
        relative_path = os.path.relpath(root, output_dir)
        target_path = os.path.join(target_dir, relative_path)

        os.makedirs(target_path, exist_ok=True)

        for file in files:
            if (os.path.splitext(file)[1].lower() in allowed_extensions and 
                file not in blocklist_files):
                
                source_file = os.path.join(root, file)
                target_file = os.path.join(target_path, file)

                if not os.path.exists(target_file):
                    print(f"Copying: {source_file} -> {target_file}")
                    shutil.copy2(source_file, target_file)

if __name__ == "__main__":
    OUTPUT_DIR = "output"
    TARGET_DIR = "all_results"
    ALLOWED_EXTENSIONS = ['.json', '.log', '.yaml','.csv']
    BLACKLIST_FILES = ['tokenizer.json', 'tokenizer_config.log','config.json','trainer_state.json','special_tokens_map.json']

    sync_files(OUTPUT_DIR, TARGET_DIR, ALLOWED_EXTENSIONS, BLACKLIST_FILES)
