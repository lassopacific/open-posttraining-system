## evaluating on the test data
import time
import json
from pathlib import Path
from evaluating_reasoning_models.final_candidate import extract_final_candidate
from evaluating_reasoning_models.render_prompt import render_prompt
from evaluating_reasoning_models.progress import progress
from evaluating_reasoning_models.text_generation_wrapper import text_generation_wrapper
from evaluating_reasoning_models.grade_answer import grade_answer


def eval_math500(model, tokenizer, device, math_data,max_token_count, out_path=None, verbose=False):
    

    if out_path is None:
        dev_name = str(device).replace(":", "-")
        out_path = Path(f"math500-{dev_name}.jsonl")

    num_examples = len(math_data)
    num_correct = 0 ## counter for, how many correct responses we got.  
    start_time = time.time()

    with open(out_path, "w", encoding="utf-8") as f: 
        for i, row in enumerate(math_data, start=1):
            try:
                prompt = render_prompt(row["problem"])  ## applying prompt template
                generated_text = text_generation_wrapper(
                    prompt,
                    model,
                    tokenizer,
                    device,
                    max_new_tokens=max_token_count,
                    verbose=verbose
                )   ## this is the main text generation function

                extracted = extract_final_candidate(generated_text)
                is_correct = grade_answer(extracted, row["answer"])  ## passing the predicted and groundtruth to grade function.
                num_correct += int(is_correct)  ## increamenting the num_correct 

                ## record---> to save iterations, that will be later be used for inspection..___> 
                record = {
                    "index": i,
                    "problem": row["problem"],
                    "gtruth_answer": row["answer"],
                    "generated_text": generated_text,
                    "extracted": extracted,
                    "is_correct": bool(is_correct)
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"\nError on sample {i}: {e}")

            ## ok, now using the eta_progress
            progress_message = progress(
                processed=i,
                total=num_examples,
                start_time=start_time,
                show_eta=True,
                label="Math-500"
            )
            
            print(progress_message, end="\r", flush=True)
            
            ## print responses during the generation
            if verbose: 
                print(
                    f"\n\n{'='*50}\n{progress_message}\n"
                    f"{'='*50}\nExtracted: {extracted}\n"
                    f"Expected: {row['answer']}\n"
                    f"Correct so far: {num_correct}\n{'-'*50}"
                )

            
        

    ## print summary information
    seconds_elapsed = time.time() - start_time
    acc = num_correct / num_examples if num_examples else 0.0

    print(f"\nAccuracy: {acc*100:.1f}% ({num_correct}/{num_examples})")
    print(f"Total time: {seconds_elapsed/60:.1f}min")
    print(f"Logs written to: {out_path}")

    return num_correct, num_examples, acc   ## this is the return function