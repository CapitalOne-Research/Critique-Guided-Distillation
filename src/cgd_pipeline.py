# pipeline.py

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Union

import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# # Import our new self-contained model definition
# from transformers import AutoConfig, AutoModelForCausalLM
# from modeling_custom_split_llama import CustomSplitLlamaConfig, CustomSplitLLamaForCausalLM

# # --- DEFINITIVE FIX: Register our custom Config and Model classes ---
# AutoConfig.register("custom_split_llama", CustomSplitLlamaConfig)
# AutoModelForCausalLM.register(CustomSplitLlamaConfig, CustomSplitLLamaForCausalLM)


# ---------------------------
# Utilities: model detection
# ---------------------------


def is_gpt_oss(model_path_or_name: str) -> bool:
    s = (model_path_or_name or "").lower()
    return ("gpt-oss" in s) or ("openai/gpt-oss" in s) or ("gpt_oss" in s)

def has_chat_template(tokenizer) -> bool:
        """Check if a tokenizer defines a chat template."""
        return hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None

# ---------------------------
# Harmony templates (GPT-OSS)
# ---------------------------

HARMONY_SYSTEM_HEADER = (
    "You are ChatGPT, a large language model trained by OpenAI.\n"
    "Knowledge cutoff: 2024-06\n"
    "Current date: 2025-08-29\n"
    "Reasoning: high\n"
    "# Valid channels: analysis, commentary, final. Channel must be included for every message."
)


def build_harmony_prompt(user_message: str, developer_message: str = "") -> str:
    """
    Build a single Harmony-formatted prompt string for GPT-OSS.
    The model will respond with assistant <analysis> ... </end> and <final> ... <return>.
    """
    parts = ["<|start|>system<|message|>", HARMONY_SYSTEM_HEADER, "<|end|>\n\n"]
    if developer_message:
        parts += ["<|start|>developer<|message|>", developer_message, "<|end|>\n\n"]
    parts += [
        "<|start|>user<|message|>",
        user_message,
        "<|end|>\n",
        "<|start|>assistant\n",  # allow model to produce <analysis> and then <final>
    ]
    return "".join(parts)


def parse_harmony_final(text: str) -> str:
    """
    Extract content between <|start|>assistant<|final|> and <|return|> or <|end|>.
    Falls back gracefully if tags are missing.
    """
    if not text:
        return text.strip()

    final_tag = "<|start|>assistant<|final|>"
    ret_tag = "<|return|>"
    end_tag = "<|end|>"

    try:
        if final_tag in text:
            after_final = text.split(final_tag, 1)[1]
            # Prefer <|return|>
            if ret_tag in after_final:
                return after_final.split(ret_tag, 1)[0].strip()
            # Fallback to <|end|>
            if end_tag in after_final:
                return after_final.split(end_tag, 1)[0].strip()
            return after_final.strip()
    except Exception:
        pass

    # Defensive fallback: handle compressed logs like "assistantfinal..."
    if "assistantfinal" in text:
        tail = text.split("assistantfinal", 1)[1]
        return tail.lstrip(":\n\r -").strip()

    # If nothing matched, just return the raw text
    return text.strip()


    
# ---------------------------
# Dataset Configuration
# ---------------------------

DATASET_MAPPING = {
    "webinstruct_sub": {
        "file_path": "webinstructsub_chunk1.json",
        "format": "json",
        "processor": "_process_webinstruct",
    },
    "webinstruct_filter": {
        "file_path": "000_WebInstructSub_filter_1_00179.jsonl",
        "format": "jsonl",
        "processor": "_process_webinstruct",
    },
    "nemotron_sft_math_v1.1_chunk_0": {
        "file_path": "nemotron_sft_math_v1.1_chunk_0.json",
        "format": "json",
        "processor": "_process_generic_chat_format",
    },
    "openmathreason_additional": {
        "file_path": "OpenMathReasoning/data/additional_problems-00000-of-00001.jsonl",
        "format": "jsonl",
        "processor": "_process_problem_answer_format",
    },
}


# ---------------------------
# Data Handling
# ---------------------------


class DataHandler:
    """Handles loading and processing of datasets for the pipeline."""

    def __init__(
        self, input_file: str = None, dataset_name: str = None, num_samples: int = None
    ):
        self.input_file = input_file
        self.dataset_name = dataset_name
        self.num_samples = num_samples

        if not self.input_file and not self.dataset_name:
            raise ValueError(
                "Either an --input_file or a --dataset_name must be provided."
            )

    def _process_webinstruct(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Processes the specific format of the webinstructsub dataset."""
        processed_data = []
        data_list = df.to_dict(orient="records")
        # Heuristic: unwrap nested dict-as-blob
        if (
            isinstance(data_list, list)
            and len(data_list) > 0
            and isinstance(data_list[0], dict)
            and len(data_list[0]) > 100
        ):
            data_list = list(data_list[0].values())

        for i, item in enumerate(data_list):
            try:
                conv = item.get("conversations", [])
                question = conv[0]["content"].split(" Answer:")[0]
                if question.startswith("Question: "):
                    question = question[len("Question: ") :]
                answer = conv[1]["content"]
                processed_data.append(
                    {"id": i, "prompt": question, "baseline_answer": answer}
                )
            except (IndexError, KeyError) as e:
                print(f"Skipping malformed record at index {i}: {item}. Error: {e}")
        return processed_data

    def _process_generic_chat_format(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Flexible processor for datasets in chat format.
        It checks for conversations under the keys 'input', 'messages', or 'conversations'.
        """
        processed_data = []
        data_list = df.to_dict(orient="records")
        conversation_keys = ["input", "messages", "conversations"]

        for i, record in enumerate(data_list):
            conversation = None
            for key in conversation_keys:
                if key in record and isinstance(record[key], list):
                    conversation = record[key]
                    break
            if not conversation:
                print(f"Skipping record {i} as no valid conversation key was found.")
                continue

            prompt, baseline_answer = None, None
            for message in conversation:
                if (
                    isinstance(message, dict)
                    and "role" in message
                    and "content" in message
                ):
                    if message["role"] == "user" and prompt is None:
                        prompt = message["content"]
                    if message["role"] == "assistant" and baseline_answer is None:
                        baseline_answer = message["content"]

            if prompt:
                processed_data.append(
                    {
                        "id": i,
                        "prompt": prompt,
                        "baseline_answer": baseline_answer or "",
                    }
                )
        return processed_data

    def _process_problem_answer_format(
        self,
        df: pd.DataFrame,
        problem_key: str = "problem",
        answer_key: str = "expected_answer",
        additional_keys: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        processed_data = []
        data_list = df.to_dict(orient="records")

        if additional_keys is None:
            additional_keys = ["problem_source", "problem_type", "used_in_kaggle"]

        for i, record in enumerate(data_list):
            if problem_key not in record:
                print(f"Skipping record {i}: missing '{problem_key}' key")
                continue

            problem = (record.get(problem_key) or "").strip()
            expected_answer = (record.get(answer_key) or "").strip()

            if not problem:
                print(f"Skipping record {i}: empty problem field")
                continue

            processed_record = {
                "id": i,
                "prompt": problem,
                "baseline_answer": expected_answer,
            }
            for key in additional_keys:
                if key in record:
                    processed_record[key] = record[key]
            processed_data.append(processed_record)

        print(
            f"Processed {len(processed_data)} records out of {len(data_list)} total records"
        )
        return processed_data

    def load_data(self) -> List[Dict[str, Any]]:
        if self.input_file:
            print(f"📖 Loading intermediate data from: {self.input_file}")
            with open(self.input_file, "r", encoding="utf-8") as f:
                return json.load(f)

        if self.dataset_name not in DATASET_MAPPING:
            raise ValueError(
                f"Dataset '{self.dataset_name}' not found in DATASET_MAPPING."
            )

        config = DATASET_MAPPING[self.dataset_name]
        file_path = config["file_path"]
        file_format = config.get("format", "json")
        processor_name = config.get("processor")

        if not processor_name:
            raise ValueError(
                f"No 'processor' specified for dataset '{self.dataset_name}' in DATASET_MAPPING."
            )

        print(f"📖 Loading initial data from: {file_path} (Format: {file_format})")

        if file_format == "jsonl":
            df = pd.read_json(file_path, lines=True)
        else:
            df = pd.read_json(file_path)

        processor: Callable[[pd.DataFrame], List[Dict[str, Any]]] = getattr(
            self, processor_name
        )
        processed_data = processor(df)

        if self.num_samples and self.num_samples < len(processed_data):
            print(f"🔪 Taking the first {self.num_samples} samples...")
            processed_data = processed_data[: self.num_samples]

        return processed_data


# ---------------------------
# Prompt Formatting
# ---------------------------


class PromptFormatter:
    """Creates prompts for each pipeline step.
    If use_harmony=True -> returns dict: {"developer": str, "user": str}
    If use_harmony=False -> returns list of chat dicts [{role, content}, ...]
    """
    @staticmethod
    def _parse_s1_1_response(text: str) -> str:
        """
        A helper function to parse the 'answer' section. It robustly finds a marker like
        'answer \nAnswer' by ignoring whitespace and returns the text that follows.
        """
        try:
            # Define a regular expression pattern.
            # \s+ matches one or more whitespace characters (space, newline, tab, etc.)
            # The 're.IGNORECASE' flag makes the search case-insensitive.
            marker_pattern = r"answer\s+Answer"
        
            # Split the text by the pattern, performing only one split.
            parts = re.split(marker_pattern, text, maxsplit=1, flags=re.IGNORECASE)
        
            # If the split results in more than one part, the marker was found.
            if len(parts) > 1:
                # The desired content is the second part of the split.
                return parts[1].lstrip(": ").strip()
        
        except Exception as e:
            # In case of any unexpected error, log it and fall back.
            print(f"Error during parsing: {e}")
            pass
        
        # Fallback: If the marker isn't found or an error occurs, return the original text.
        return text

        
    @staticmethod
    def for_probing(
        item: Dict[str, Any], use_harmony: bool = False
    ) -> Union[Dict[str, str], List[Dict[str, str]]]:
        # This function should ONLY create the prompt for the student model.
        user_content = item["prompt"]
        if use_harmony:
            developer = (
                "Please think step by step. Use the `analysis` channel for reasoning, "
                "and put only the final clean answer in the `final` channel."
            )
            return {"developer": developer, "user": user_content}
        return [{"role": "user", "content": user_content}]


    @staticmethod
    def for_critique(
        item: Dict[str, Any], student_model_name: str, use_harmony: bool = False
    ) -> Union[Dict[str, str], List[Dict[str, str]]]:
        task_instruction = (
            "You are a knowledgeable science and mathematics expert. A student has attempted to answer "
            "the following question. You are provided with the student's answer. "
            "Review the student's answer carefully, identify any inaccuracies, missing logical steps, or unclear explanations. "
            "Provide a detailed critique and then end with a line exactly in the format: 'Conclusion: right' or 'Conclusion: wrong'."
        )
        
        # ✅ PARSE THE STUDENT RESPONSE HERE
        student_response = item.get('student_response', '')
        # if "s1.1" in student_model_name.lower():
        #     student_response = PromptFormatter._parse_s1_1_response(student_response)

        user_content = (
            f"Question: {item['prompt']}\n\n"
            f"Student's Answer: {student_response}" # Now using the clean version
        )
        
        if use_harmony:
            return {"developer": task_instruction, "user": user_content}
        return [
            {"role": "system", "content": task_instruction},
            {"role": "user", "content": user_content},
        ]

    @staticmethod
    def for_refinement(
        item: Dict[str, Any], critique_model_name: str, use_harmony: bool = False
    ) -> Union[Dict[str, str], List[Dict[str, str]]]:
        task_instruction = (
            "You are a knowledgeable science expert. Based on the student's initial answer and your critique, "
            "write an improved, comprehensive answer that corrects inaccuracies and fills missing steps. "
            "Ensure the final answer is clear, accurate, and complete."
        )

        # PARSE BOTH THE STUDENT RESPONSE AND THE CRITIQUE HERE
        student_response = item.get('student_response', '')
        # student_model = item.get('probe_model', '') # Name of model from probe step
        # if "s1.1" in student_model.lower():
        #     student_response = PromptFormatter._parse_s1_1_response(student_response)

        critique = item.get('critique', '')
        # if "s1.1" in critique_model_name.lower():
        #     critique = PromptFormatter._parse_s1_1_response(critique)

        user_content = (
            f"Question: {item['prompt']}\n\n"
            f"Student's Initial Answer: {student_response}\n\n" # Now clean
            f"Critique: {critique}" # Now clean
        )
        
        if use_harmony:
            return {"developer": task_instruction, "user": user_content}
        return [
            {"role": "system", "content": task_instruction},
            {"role": "user", "content": user_content},
        ]

# ---------------------------
# vLLM Client
# ---------------------------


class VLLMClient:
    """A client to handle generation using a local vLLM model."""

    def __init__(
        self, model_path: str, tensor_parallel_size: int = 1, max_model_len: int = 8192
    ):
        print(f"🤖 Initializing vLLM model: {model_path}")
        self.model_path = model_path
        self.is_harmony = is_gpt_oss(model_path)

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            # trust_remote_code=True,
            gpu_memory_utilization=0.9,
            max_model_len=max_model_len,
            max_num_seqs=256,
            enforce_eager=False,
            dtype="bfloat16"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.has_chat_template = has_chat_template(self.tokenizer)
        if self.is_harmony:
            print("🔧 Using Harmony prompt format (GPT-OSS).")
        elif self.has_chat_template:
            print("💬 Using chat template from tokenizer.")
        else:
            print("📜 No chat template detected — falling back to manual prompt formatting.")

    
    def _to_prompt(self, conv: Union[List[Dict[str, str]], Dict[str, str]]) -> str:
        """Build a raw prompt string from either Harmony dict or role-based messages."""
        if self.is_harmony:
            # Expect conv as {"developer": str, "user": str}
            developer_msg = conv.get("developer", "") if isinstance(conv, dict) else ""
            user_msg = conv.get("user", "") if isinstance(conv, dict) else ""
            return build_harmony_prompt(user_msg, developer_msg)
    
        # Non-Harmony models
        if isinstance(conv, list):
            if self.has_chat_template:
                # Instruction-tuned chat model (e.g., Llama-3.1-8B-Instruct)
                # This part is correct and remains the same.
                return self.tokenizer.apply_chat_template(
                    conv, tokenize=False, add_generation_prompt=True
                )
            else:
                # --- START: MODIFIED SECTION FOR BASE MODELS ---
                # Base model fallback (e.g., Llama-3.1-8B-Base)
                # We will format it as a simple Q&A prompt.
                prompt = ""
                # Assuming the conversation starts with a user question.
                user_content = ""
                for msg in conv:
                    if msg.get("role") == "user":
                        user_content = msg.get("content", "")
                        # We'll just take the first user message for a simple Q&A.
                        break
                
                # Use a clear format that the base model can complete.
                prompt = f"Question: {user_content}\n\nAnswer:"
                return prompt
                # --- END: MODIFIED SECTION FOR BASE MODELS ---
    
        raise ValueError(
            "Non-Harmony conversation should be a list of {role, content} dicts."
        )


    def generate(
        self, conversations: List[Union[List[Dict[str, str]], Dict[str, str]]]
    ) -> List[str]:
        prompts = [self._to_prompt(conv) for conv in conversations]
        print(f"Example compiled prompt (first):\n{prompts[0][:800]}...\n")
        
        stop_token_ids = []
        if self.tokenizer.eos_token_id is not None:
            stop_token_ids.append(self.tokenizer.eos_token_id)
        
        # Llama 3 models also use a specific token ID for end-of-turn.
        # It's often the same as EOS but good to add explicitly if needed.
        # For Llama-3.1, 128009 is '<|eot_id|>'
        if 128009 not in stop_token_ids:
             stop_token_ids.append(128009)

        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=0.95,
            max_tokens=16384,
            stop_token_ids=stop_token_ids,
            # For Harmony, we let the model produce <|return|>; we parse it out later.
            # stop=[]  # optional: you can set stop=["<|return|>"] if you prefer truncation at boundary
        )

        outputs = self.llm.generate(prompts, sampling_params)
        raw_texts = [output.outputs[0].text for output in outputs]

        return [t.strip() for t in raw_texts]
        

# ---------------------------
# Pipeline
# ---------------------------


def run_pipeline_step(args: argparse.Namespace):
    vllm_client = VLLMClient(
        args.model_path, args.tensor_parallel_size, args.max_model_len
    )
    use_harmony = vllm_client.is_harmony

    # Load data
    data_handler = DataHandler(
        input_file=args.input_file,
        dataset_name=args.dataset_name,
        num_samples=args.num_samples,
    )
    data = data_handler.load_data()
    print(f"✅ Loaded {len(data)} items for step '{args.step}'")

    prompts: List[Union[List[Dict[str, str]], Dict[str, str]]] = []
    print(f"🔨 Building prompts for step: '{args.step}'")

    if args.step == "critique":
        # Assumes each 'item' has a 'student_model' key.
        prompts = [
            PromptFormatter.for_critique(
                item,
                student_model_name=item.get("student_model", ""),
                use_harmony=use_harmony
            )
            for item in data
        ]
    elif args.step == "refine":
        # Assumes each 'item' has a 'critique_model' key.
        prompts = [
            PromptFormatter.for_refinement(
                item,
                critique_model_name=item.get("critique_model", ""),
                use_harmony=use_harmony
            )
            for item in data
        ]
    elif args.step == "probe":
        prompts = [
            PromptFormatter.for_probing(item, use_harmony=use_harmony)
            for item in data
        ]
    else:
        raise ValueError(f"Unknown pipeline step provided: {args.step}")
        
    # with open(f"prompts_{args.step}.json", "w", encoding="utf-8") as f:
    #     json.dump(all_conversations, f, ensure_ascii=False, indent=2)

    # Generate
    print(f"Generating responses for {len(prompts)} prompts...")
    generated_texts = vllm_client.generate(prompts)
    print("Generation complete.")

    # Parse the generated text
    parsed_texts = []
    if use_harmony:
        parsed_texts = [parse_harmony_final(text) for text in generated_texts]
    else:
        # We only parse the output from the 'probe' step if it's an s1.1 model
        if "s1.1" in args.model_path.lower():
            print("Parsing 'answer Answer' section from s1.1 outputs...")
            for text in generated_texts:
                # The _parse_s1_1_response helper handles the 'answer Answer' logic
                parsed_texts.append(PromptFormatter._parse_s1_1_response(text))
        else:
            # For all other steps, no parsing is needed here
            parsed_texts = generated_texts


    # Merge results (This section is fine)
    output_key = {
        "probe": "student_response",
        "critique": "critique",
        "refine": "refined_response",
    }[args.step]

    # results = []
    # skipped_count = 0
    # for i, item in enumerate(data):
    #     final_text = parsed_texts[i]

    #     # 🗑️ FILTERING: Applied *after* parsing.
    #     # Now we check if the parsed text *still* starts with 'think', which means
    #     # the 'answer Answer' marker was never found.
    #     if final_text.strip().lower().startswith('think'):
    #         skipped_count += 1
    #         continue  # Discard this sample

    #     item_copy = dict(item)
    #     item_copy[f"{args.step}_model"] = args.model_path
    #     item_copy[output_key] = final_text
    #     results.append(item_copy)

    # if skipped_count > 0:
    #     print(f"🗑️ Discarded {skipped_count} out of {len(data)} samples due to truncated 'think' sections.")


    

    results = []
    for i, item in enumerate(data):
        item_copy = dict(item)
        # Add the model name that generated this output for the next step
        item_copy[f"{args.step}_model"] = args.model_path
        item_copy[output_key] = parsed_texts[i]
        results.append(item_copy)

    # If final refine step, reformat to SFT triples (This section needs a small fix)
    if args.step == "refine":
        print("Reformatting final output for SFT (instruction, input, output)...")
        task_instruction = (
            "You are a knowledgeable science expert. Based on the student's initial answer and the teacher's critique, "
            "write an improved, comprehensive answer that corrects inaccuracies and fills missing steps. "
            "Ensure the final answer is clear, accurate, and complete."
        )
        sft_formatted_data = []
        for item in results:

            # 1. Get the student response and the name of the model that created it.
            student_response = item.get('student_response', '')
            student_model_name = item.get('probe_model', '') # Get model name from probe step

            # 2. Parse the student response if it came from an s1.1 model.
            if "s1.1" in student_model_name.lower():
                student_response = PromptFormatter._parse_s1_1_response(student_response)

            # 3. Do the same for the critique as an extra safeguard.
            critique = item.get('critique', '')
            # critique_model_name = item.get('critique_model', '') # Get model name from critique step
            # if "s1.1" in critique_model_name.lower():
            #     critique = PromptFormatter._parse_s1_1_response(critique)

            input_prompt = (
                f"Question: {item['prompt']}\n\n"
                f"Student's Initial Answer: {student_response}\n\n"
                f"Critique: {critique}"
            )

                
            # input_prompt = (
            #     f"Question: {item['prompt']}\n\n"
            #     f"Student's Initial Answer: {item.get('student_response','')}\n\n"
            #     f"Critique: {item.get('critique','')}"
            # )

            sft_formatted_data.append(
                {
                    "instruction": task_instruction,
                    "input": input_prompt,
                    "output": item["refined_response"],
                }
            )
        results = sft_formatted_data

    # Save
    os.makedirs(Path(args.output_file).parent, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(
        f"🎉 Successfully completed step '{args.step}'. Results saved to {args.output_file}"
    )


# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a step in the data generation pipeline."
    )

    parser.add_argument(
        "--step",
        type=str,
        required=True,
        choices=["probe", "critique", "refine"],
        help="The pipeline step to execute.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the local vLLM model directory.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the output JSON file.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism.",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=8192,
        help="Max model length.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input_file",
        type=str,
        help="Path to an intermediate JSON dataset file from a previous step.",
    )
    group.add_argument(
        "--dataset_name",
        type=str,
        help="Name of the initial dataset to load (configured in DATASET_MAPPING).",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to process from the initial dataset.",
    )

    args = parser.parse_args()
    run_pipeline_step(args)
