from transformers import AutoModelForCausalLM, AutoTokenizer


def get_dataset():
    with open("code_completion_for_dataset.py") as file:
        code = file.read().split("\n\n")
    code = list(map(lambda part: part.strip(), code))
    code = list(filter(lambda part: part.startswith("def"), code))
    # Format:
    # Code before, Code after, Line missing
    dataset = []
    for function in code:
        lines = function.split("\n")
        if len(lines) < 3 or len(lines) > 10:
            continue
        line_to_remove = int(len(lines) // 2)
        removed_line = lines[line_to_remove]
        if "." not in removed_line:
            continue
        dataset.append(["\n".join(lines[:line_to_remove] + [removed_line[:removed_line.index(".")] + "."]),
                        "\n".join(lines[line_to_remove+1:]),
                        removed_line[removed_line.index(".")+1:]])
    return dataset


dataset = get_dataset()

checkpoint = "bigcode/tiny_starcoder_py"
device = "cuda" # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

good_matches = 0
errors = 0
total = len(dataset)
for example in dataset:
    input_text = f"<fim_prefix>{example[0]}<fim_suffix>\n{example[1]}"
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(inputs,  max_length=len(input_text)+50)
    model_result = tokenizer.decode(outputs[0])
    try:
        model_result = model_result[model_result.index("<fim_middle>")+12:]
        if "<|endoftext|>" in model_result:
            model_result = model_result[:model_result.index("<|endoftext|>")]
        if "\n" in model_result:
            model_result = model_result[:model_result.index("\n")]
    except:
        print("error")
        errors += 1
    print("Model output:")
    print(model_result)
    print("Correct output:")
    print(example[2])
    print("\n---------------------------------------\n")
    if model_result in example[2] or example[2] in model_result:
        good_matches += 1

print(f"Total: {total}, Error: {errors}, Correct: {good_matches}")
