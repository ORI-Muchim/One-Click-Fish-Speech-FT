import os
import subprocess
import argparse
from pathlib import Path
from tqdm import tqdm
from faster_whisper import WhisperModel

class WhisperProcessor:
    def __init__(self, model_size="large-v3", device="cuda", compute_type="float16"):
        print("Loading Faster Whisper model...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
    
    def process_files(self, root_folder):
        """Process all WAV files in the directory and create .lab files"""
        wav_files = self._collect_wav_files(root_folder)
        print(f"Found {len(wav_files)} wav files")
        
        files_to_process = []
        for wav_path in wav_files:
            lab_path = Path(wav_path).with_suffix('.lab')
            if not lab_path.exists():
                files_to_process.append(wav_path)
        
        if not files_to_process:
            print("All WAV files already have corresponding .lab files. Skipping Whisper processing.")
            return
        
        print(f"Processing {len(files_to_process)} files that don't have .lab files")
        for wav_path in tqdm(files_to_process, desc="Processing files with Whisper"):
            try:
                segments, info = self.model.transcribe(wav_path, beam_size=5)
                text = " ".join([segment.text for segment in segments]).strip()
                
                lab_path = Path(wav_path).with_suffix('.lab')
                with open(lab_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                print(f"\nProcessed {wav_path}")
                print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
                
            except Exception as e:
                print(f"\nError processing {wav_path}: {str(e)}")
    
    def _collect_wav_files(self, root_folder):
        """Collect all WAV files in the directory"""
        wav_files = []
        for folder_path, _, files in os.walk(root_folder):
            for file in files:
                if file.endswith('.wav'):
                    full_path = str(Path(folder_path) / file)
                    wav_files.append(full_path)
        return wav_files

class FishSpeechFinetuner:
    def __init__(self, data_dir, checkpoint_dir, project_name):
        self.data_dir = Path(data_dir).absolute()
        self.checkpoint_dir = Path(checkpoint_dir).absolute()
        self.project_name = project_name
        
        # Get the project root directory (where fish_speech folder is located)
        self.project_root = Path(__file__).parent.absolute()
        
        # Create necessary directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create .project-root file if it doesn't exist
        project_root_file = self.project_root / ".project-root"
        if not project_root_file.exists():
            project_root_file.touch()
            print(f"Created .project-root file at {project_root_file}")
    
    def download_models(self):
        """Download required model weights if they don't exist"""
        print("Downloading model weights...")
        
        if not (self.checkpoint_dir / "fish-speech-1.5").exists():
            subprocess.run([
                "huggingface-cli", "download",
                "fishaudio/fish-speech-1.5",
                "--local-dir", str(self.checkpoint_dir / "fish-speech-1.5")
            ], check=True)
    
    def extract_semantic_tokens(self):
        """Extract semantic tokens using VQGAN"""
        print("Extracting semantic tokens...")
        
        subprocess.run([
            "python", "tools/vqgan/extract_vq.py",
            str(self.data_dir),
            "--num-workers", "1",
            "--batch-size", "8",
            "--config-name", "firefly_gan_vq",
            "--checkpoint-path",
            str(self.checkpoint_dir / "fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth")
        ], check=True)
    
    def build_dataset(self):
        """Pack the dataset into protobuf format"""
        print("Building dataset...")
        
        subprocess.run([
            "python", "tools/llama/build_dataset.py",
            "--input", str(self.data_dir),
            "--output", str(self.data_dir / "protos"),
            "--text-extension", ".lab",
            "--num-workers", "16"
        ], check=True)
    
    def train_model(self):
        """Fine-tune the model using LoRA"""
        print("Starting model fine-tuning...")
        
        # Change to project root directory before running the training
        original_dir = os.getcwd()
        os.chdir(self.project_root)
        
        try:
            subprocess.run([
                "python", "fish_speech/train.py",
                "--config-name", "text2semantic_finetune",
                f"project={self.project_name}",
                "+lora@model.model.lora_config=r_8_alpha_16"
            ], check=True)
        finally:
            # Change back to original directory
            os.chdir(original_dir)
    
    def get_latest_checkpoint(self, checkpoint_dir):
        """Get the path of the latest checkpoint file"""
        checkpoint_files = list(Path(checkpoint_dir).glob("step_*.ckpt"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
        
        # Extract step numbers and find the latest one
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[1]))
        return latest_checkpoint

    def merge_weights(self):
        """Merge LoRA weights with base weights"""
        print("Merging weights...")
        
        # Find the latest checkpoint
        checkpoint_dir = Path(f"results/{self.project_name}/checkpoints")
        latest_checkpoint = self.get_latest_checkpoint(checkpoint_dir)
        print(f"Using latest checkpoint: {latest_checkpoint}")
        
        subprocess.run([
            "python", "tools/llama/merge_lora.py",
            "--lora-config", "r_8_alpha_16",
            "--base-weight", str(self.checkpoint_dir / "fish-speech-1.5"),
            "--lora-weight", str(latest_checkpoint),
            "--output", str(self.checkpoint_dir / f"fish-speech-1.5-{self.project_name}-lora")
        ], check=True)
    
    def run_pipeline(self):
        """Run the complete fine-tuning pipeline"""
        try:
            # First run Whisper processing
            whisper_processor = WhisperProcessor()
            whisper_processor.process_files(self.data_dir)
            
            # Then run the rest of the pipeline
            self.download_models()
            self.extract_semantic_tokens()
            self.build_dataset()
            self.train_model()
            self.merge_weights()
            print("Fine-tuning pipeline completed successfully!")
            
        except subprocess.CalledProcessError as e:
            print(f"Error occurred during execution: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error occurred: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Fish-Speech Fine-tuning Automation with Whisper")
    parser.add_argument("--data-dir", type=str, required=True,
                      help="Directory containing the dataset")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                      help="Directory for model checkpoints")
    parser.add_argument("--project-name", type=str, required=True,
                      help="Name of the project for training")
    parser.add_argument("--whisper-only", action="store_true",
                      help="Only run Whisper processing without fine-tuning")
    
    args = parser.parse_args()
    
    if args.whisper_only:
        whisper_processor = WhisperProcessor()
        whisper_processor.process_files(args.data_dir)
    else:
        finetuner = FishSpeechFinetuner(
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            project_name=args.project_name
        )
        finetuner.run_pipeline()

if __name__ == "__main__":
    main()
