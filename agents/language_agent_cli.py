import os
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure OpenRouter
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# Initialize token counter
def count_tokens(text: str) -> int:
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

# Cost tracking
class CostTracker:
    def __init__(self):
        self.total_tokens = 0
        self.total_cost = 0
        self.daily_usage = {}
        self.load_usage()

    def load_usage(self):
        if os.path.exists('language_usage.json'):
            with open('language_usage.json', 'r') as f:
                data = json.load(f)
                self.total_tokens = data.get('total_tokens', 0)
                self.total_cost = data.get('total_cost', 0)
                self.daily_usage = data.get('daily_usage', {})

    def save_usage(self):
        with open('language_usage.json', 'w') as f:
            json.dump({
                'total_tokens': self.total_tokens,
                'total_cost': self.total_cost,
                'daily_usage': self.daily_usage
            }, f)

    def track_usage(self, tokens: int):
        today = datetime.now().strftime('%Y-%m-%d')
        self.total_tokens += tokens
        self.daily_usage[today] = self.daily_usage.get(today, 0) + tokens
        # Calculate cost (OpenRouter GPT-3.5-turbo: $0.002 per 1K tokens)
        self.total_cost = (self.total_tokens / 1000) * 0.002
        self.save_usage()

    def get_usage_summary(self):
        return f"""
Usage Summary:
-------------
Total Tokens: {self.total_tokens:,}
Total Cost: ${self.total_cost:.4f}
Today's Usage: {self.daily_usage.get(datetime.now().strftime('%Y-%m-%d'), 0):,} tokens
"""

def detect_language(text):
    system_prompt = """You are a language detection expert. Your task is to identify the language of the given text.
    Respond with ONLY the language name in English (e.g., "English", "Spanish", "French", etc.).
    If you cannot determine the language, respond with "Unknown"."""
    
    try:
        response = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0,
            max_tokens=50
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def translate_text(text, target_language):
    system_prompt = f"""You are a professional translator. Translate the following text to {target_language}.
    Provide ONLY the translation without any explanations or additional text.
    If the text is already in {target_language}, respond with "Text is already in {target_language}."
    If you cannot translate to {target_language}, respond with "Cannot translate to {target_language}." """
    
    try:
        response = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def correct_text(text):
    system_prompt = """You are a language correction expert. Correct any spelling, grammar, or punctuation errors in the text.
    Provide ONLY the corrected text without any explanations or additional text."""
    
    try:
        response = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    print("Initializing Language Agent...")
    cost_tracker = CostTracker()
    
    print("\nLanguage Agent CLI")
    print("=================")
    print("Commands:")
    print("  detect <text> - Detect the language of the text")
    print("  translate <text> to <language> - Translate text to specified language")
    print("  correct <text> - Correct spelling and grammar in the text")
    print("  usage - Show usage statistics")
    print("  exit - Exit the program")
    print("===============================================\n")

    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'exit':
                print("\nFinal Usage Summary:")
                print(cost_tracker.get_usage_summary())
                break
                
            if user_input.lower() == 'usage':
                print(cost_tracker.get_usage_summary())
                continue

            if not user_input:
                continue

            # Track input tokens
            input_tokens = count_tokens(user_input)
            cost_tracker.track_usage(input_tokens)

            # Process the command
            if user_input.lower().startswith('detect '):
                text = user_input[7:].strip()
                if text:
                    print("\nDetecting language...")
                    response = detect_language(text)
                    print(f"Detected language: {response}")
                else:
                    print("\nPlease provide text to detect language.")
                    continue
                    
            elif user_input.lower().startswith('translate '):
                parts = user_input[10:].split(' to ', 1)
                if len(parts) == 2:
                    text, target_language = parts
                    if text and target_language:
                        print(f"\nTranslating to {target_language}...")
                        response = translate_text(text, target_language)
                        print(f"Translation: {response}")
                    else:
                        print("\nPlease provide both text and target language.")
                        continue
                else:
                    print("\nPlease use format: translate <text> to <language>")
                    continue
                    
            elif user_input.lower().startswith('correct '):
                text = user_input[8:].strip()
                if text:
                    print("\nCorrecting text...")
                    response = correct_text(text)
                    print(f"Corrected text: {response}")
                else:
                    print("\nPlease provide text to correct.")
                    continue
                    
            else:
                print("\nUnknown command. Available commands:")
                print("  detect <text> - Detect the language of the text")
                print("  translate <text> to <language> - Translate text to specified language")
                print("  correct <text> - Correct spelling and grammar in the text")
                print("  usage - Show usage statistics")
                print("  exit - Exit the program")
                continue

            # Track output tokens
            output_tokens = count_tokens(str(response))
            cost_tracker.track_usage(output_tokens)

            # Print usage info
            print(f"\nUsage: {input_tokens + output_tokens:,} tokens (${(input_tokens + output_tokens)/1000*0.002:.4f})")

        except KeyboardInterrupt:
            print("\n\nExiting...")
            print(cost_tracker.get_usage_summary())
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    main() 