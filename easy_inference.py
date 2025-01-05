import os
import subprocess
from pathlib import Path

class FishSpeechInference:
    def __init__(self, checkpoint_dir="checkpoints/fish-speech-1.5", project_name=None):
        self.base_checkpoint_dir = Path(checkpoint_dir)
        
        # Project-specific directory for other models
        if project_name:
            self.checkpoint_dir = Path(checkpoint_dir.replace("fish-speech-1.5", f"fish-speech-1.5-{project_name}-lora"))
        else:
            self.checkpoint_dir = self.base_checkpoint_dir
            
        self.project_root = Path(__file__).parent.absolute()
    
    def get_vqgan_model_path(self):
        return str(self.base_checkpoint_dir / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth")
    
    def generate_voice_prompt(self, input_audio):
        subprocess.run([
            "python", "tools/vqgan/inference.py",
            "-i", str(input_audio),
            "--checkpoint-path", self.get_vqgan_model_path()
        ], check=True)
        return Path("fake.npy")
    
    def generate_semantic_tokens(self, text, prompt_text, prompt_tokens):
        cmd = [
            "python", "tools/llama/generate.py",
            "--text", text,
            "--checkpoint-path", str(self.checkpoint_dir),
            "--num-samples", "1",
            "--prompt-text", prompt_text,
            "--prompt-tokens", str(prompt_tokens)
        ]
        subprocess.run(cmd, check=True)
        return list(Path(".").glob("codes_*.npy"))
    
    def generate_audio(self, semantic_tokens):
        subprocess.run([
            "python", "tools/vqgan/inference.py",
            "-i", str(semantic_tokens),
            "--checkpoint-path", self.get_vqgan_model_path()
        ], check=True)
    
    def run_inference(self, text, reference_audio, prompt_text):
        os.chdir(self.project_root)
        prompt_tokens = self.generate_voice_prompt(reference_audio)
        code_files = self.generate_semantic_tokens(text, prompt_text, prompt_tokens)
        for code_file in code_files:
            self.generate_audio(code_file)

def main():
    text = "바위 아래 작은 샘물도 흘러서, 바다로 갈 뜻을 가지고 있고, 뜰 앞의 작은 나무도, 하늘을 꿰뚫는 마음을 가지고 있다."
    reference_audio = "mix.wav"
    prompt_text = "매니저들의 인맥과 노하우를 활용해서 성사시키기 어려운 계약을 따내거나, 부득이하게 겹친 스케쥴을 풀기도 하죠. 꼭 매니저가 해야한다는 법은 없습니다. 지금처럼 회사 차원에서 관리하기도 합니다. 만약 계속일을 한다면, 세 가지의 결말이 있습니다."
    checkpoint_dir = "checkpoints/fish-speech-1.5"
    project_name = None
    
    inferencer = FishSpeechInference(
        checkpoint_dir=checkpoint_dir,
        project_name=project_name
    )
    inferencer.run_inference(text, reference_audio, prompt_text)

if __name__ == "__main__":
    main()
