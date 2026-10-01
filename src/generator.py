"""
Generator module using LoRA fine-tuning
Supports: Qwen 2.5, Llama 3.1, Mistral-7B, Phi-3.5
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from datasets import Dataset
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path


class GeneratorTrainer:
    """Trains LLM with LoRA for answer generation (supports Qwen, Llama, Mistral, Phi)"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-14B-Instruct",
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        quantization: str = "4bit",
        device: str = None
    ):
        """
        Initialize generator trainer
        
        Args:
            model_name: Pre-trained model name
                - "Qwen/Qwen2.5-14B-Instruct" (recommended, no auth needed)
                - "meta-llama/Meta-Llama-3.1-13B-Instruct" (requires auth)
                - "mistralai/Mistral-7B-Instruct-v0.3" (no auth needed)
                - "microsoft/Phi-3.5-mini-instruct" (no auth needed, fast)
            lora_rank: LoRA rank (r=8 is a good starting point)
            lora_alpha: LoRA alpha scaling parameter
            lora_dropout: LoRA dropout rate
            quantization: Quantization type
                - "4bit": 4-bit quantization (~9GB for 14B, fastest, 95-98% quality) [DEFAULT]
                - "8bit": 8-bit quantization (~14GB for 14B, 98-99% quality)
                - "fp16": Half precision (~28GB for 14B, 99.9% quality)
                - "none" or "fp32": Full precision (~56GB for 14B, 100% quality)
            device: Device to use
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.model_name = model_name
        self.quantization = quantization.lower()
        
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Load model with chosen quantization
        print(f"Loading model: {model_name} (quantization: {self.quantization})")
        
        if self.quantization == "4bit":
            # QLoRA: 4-bit quantization (~9GB for 14B)
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            self.model = prepare_model_for_kbit_training(self.model)
            
        elif self.quantization == "8bit":
            # 8-bit quantization (~14GB for 14B)
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            self.model = prepare_model_for_kbit_training(self.model)
            
        elif self.quantization == "fp16":
            # Half precision (~28GB for 14B)
            print("   Using FP16 (half precision)")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto" if self.device == 'cuda' else None,
                trust_remote_code=True
            )
            
        elif self.quantization in ["none", "fp32"]:
            # Full precision (~56GB for 14B)
            print("   Using FP32 (full precision) - requires large GPU!")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="auto" if self.device == 'cuda' else None,
                trust_remote_code=True
            )
            
        else:
            raise ValueError(f"Unknown quantization: {self.quantization}. Use: 4bit, 8bit, fp16, fp32, or none")
        
        # Configure LoRA
        print(f"Configuring LoRA (rank={lora_rank}, alpha={lora_alpha})")
        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
    
    def create_prompt(
        self,
        question: str,
        context: Optional[str] = None,
        include_sources: bool = True
    ) -> str:
        """
        Create prompt for the model
        
        Format:
        [INST] You are a teaching assistant for a deep learning course.
        Answer the student's question using only the provided context.
        
        Context: {context}
        
        Question: {question}
        
        Provide a clear answer and cite your sources (slide numbers, Piazza posts, etc.).
        [/INST]
        
        Args:
            question: Student question
            context: Retrieved context (optional)
            include_sources: Whether to ask for source citations
        
        Returns:
            Formatted prompt
        """
        base_instruction = "You are a teaching assistant for a deep learning course."
        
        if context:
            instruction = (
                f"{base_instruction}\n"
                f"Answer the student's question using only the provided context.\n\n"
                f"Context: {context}\n\n"
                f"Question: {question}\n\n"
            )
        else:
            instruction = (
                f"{base_instruction}\n"
                f"Answer the student's question.\n\n"
                f"Question: {question}\n\n"
            )
        
        if include_sources:
            instruction += "Provide a clear answer and cite your sources (slide numbers, Piazza posts, etc.)."
        else:
            instruction += "Provide a clear and helpful answer."
        
        # Use tokenizer's chat template (auto-handles Qwen/Llama/Mistral/Phi formats)
        messages = [
            {"role": "system", "content": base_instruction},
            {"role": "user", "content": instruction.replace(base_instruction + "\n", "")}
        ]
        
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            # Fallback if no chat template
            print(f"⚠️  No chat template, using simple format: {e}")
            prompt = f"{base_instruction}\n\n{instruction}"
        
        return prompt
    
    def prepare_training_data(
        self,
        train_df: pd.DataFrame,
        include_context: bool = True
    ) -> Dataset:
        """
        Prepare training data for fine-tuning
        
        Args:
            train_df: DataFrame with 'question', 'answer', and optionally 'context'
            include_context: Whether to include retrieved context in prompts
        
        Returns:
            HuggingFace Dataset
        """
        formatted_data = []
        
        for _, row in train_df.iterrows():
            question = row['question']
            answer = row['answer']
            context = row.get('context', None) if include_context else None
            
            prompt = self.create_prompt(question, context)
            full_text = f"{prompt} {answer}{self.tokenizer.eos_token}"
            
            formatted_data.append({
                'text': full_text,
                'prompt': prompt,
                'answer': answer
            })
        
        return Dataset.from_list(formatted_data)
    
    def tokenize_function(self, examples):
        """Tokenize text for training"""
        return self.tokenizer(
            examples['text'],
            truncation=True,
            max_length=512,
            padding='max_length'
        )
    
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset],
        output_dir: str,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 100,
        logging_steps: int = 10,
        save_steps: int = 100
    ):
        """
        Train model with LoRA
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            output_dir: Where to save checkpoints
            num_epochs: Number of training epochs
            batch_size: Per-device batch size
            learning_rate: Learning rate
            gradient_accumulation_steps: Gradient accumulation steps
            warmup_steps: Warmup steps
            logging_steps: Log every N steps
            save_steps: Save checkpoint every N steps
        """
        # Tokenize datasets
        print("Tokenizing datasets...")
        tokenized_train = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        tokenized_val = None
        if val_dataset:
            tokenized_val = val_dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=val_dataset.column_names
            )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            evaluation_strategy="steps" if val_dataset else "no",
            eval_steps=save_steps if val_dataset else None,
            save_total_limit=3,
            fp16=self.device == 'cuda',
            push_to_hub=False,
            report_to=["tensorboard"],
            load_best_model_at_end=True if val_dataset else False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=data_collator,
        )
        
        # Train
        print("\nStarting training...")
        trainer.train()
        
        # Save final model
        print(f"\nSaving model to {output_dir}")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print("✓ Training complete!")


class Generator:
    """Production generator for inference"""
    
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen2.5-14B-Instruct",
        lora_path: Optional[str] = None,
        device: str = None
    ):
        """
        Load model for inference
        
        Args:
            base_model_name: Base model name (should match training model!)
            lora_path: Path to LoRA adapter (if fine-tuned)
            device: Device to use
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Loading tokenizer: {base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Loading model: {base_model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            device_map="auto" if self.device == 'cuda' else None,
            trust_remote_code=True
        )
        
        # Load LoRA adapter if provided
        if lora_path:
            print(f"Loading LoRA adapter from {lora_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.model = self.model.merge_and_unload()  # Merge for faster inference
        
        self.model.eval()
    
    def generate(
        self,
        question: str,
        context: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate answer for a question
        
        Args:
            question: Student question
            context: Retrieved context (optional)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        
        Returns:
            Generated answer
        """
        # Create prompt
        prompt = self._create_prompt(question, context)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer (remove prompt)
        answer = full_output[len(prompt):].strip()
        
        return answer
    
    def _create_prompt(self, question: str, context: Optional[str] = None) -> str:
        """Create prompt using proper chat template (same as in trainer)"""
        system_message = "You are a teaching assistant for a deep learning course."
        
        if context:
            user_message = (
                f"Answer the student's question using only the provided context.\n\n"
                f"Context: {context}\n\n"
                f"Question: {question}\n\n"
                f"Provide a clear answer and cite your sources (slide numbers, Piazza posts, etc.)."
            )
        else:
            user_message = (
                f"Question: {question}\n\n"
                f"Provide a clear and helpful answer."
            )
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            # Fallback
            prompt = f"{system_message}\n\n{user_message}"
        
        return prompt


if __name__ == "__main__":
    print("=== Generator Module ===")
    print("\nNote: This module requires a GPU with sufficient memory.")
    print("For full training, use the Colab notebook.")
    print("\nExample usage:")
    print("""
    from generator import GeneratorTrainer, Generator
    from data_preparation import DataPreparer
    
    # Load data
    preparer = DataPreparer()
    train_df, val_df, test_df = preparer.load_splits(prefix="sample")
    
    # Initialize trainer
    trainer = GeneratorTrainer(
        lora_rank=8,
        use_4bit=True  # Use QLoRA for memory efficiency
    )
    
    # Prepare data
    train_dataset = trainer.prepare_training_data(train_df)
    val_dataset = trainer.prepare_training_data(val_df)
    
    # Train
    trainer.train(
        train_dataset,
        val_dataset,
        output_dir="models/generator-lora",
        num_epochs=3,
        batch_size=4
    )
    
    # Inference
    generator = Generator(lora_path="models/generator-lora")
    answer = generator.generate(
        "How do I implement backpropagation?",
        context="Backpropagation uses the chain rule..."
    )
    print(answer)
    """)

