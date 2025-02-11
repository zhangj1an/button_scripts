
import os 
import json
import _0_t2a_config
#from foundation_models.llamaindex_llama3 import load_model, load_documents, make_inference
from foundation_models.gpt_4o_model import GPT4O
from utils.create_or_replace_path import create_or_replace_path
from utils.load_string_from_file import load_string_from_file

def define_action_names(user_manual_filepath, visually_grounded_elements_filepath, propose_action_prompt_filepath, filter_action_prompt_filepath,proposed_actions_filepath):
    # load user manual file 
    user_manual_text = load_string_from_file(user_manual_filepath)

    # load grounded elements from filepath

    grounded_element_prompt = load_string_from_file(visually_grounded_elements_filepath)

    with open(visually_grounded_elements_filepath, "r") as f:
        json_list = json.load(f)
    # read all the element names from this list 
    grounded_element_prompt = [item['element_name'] for item in json_list]
    grounded_element_prompt = ", ".join(grounded_element_prompt)

    propose_action_prompt = load_string_from_file(propose_action_prompt_filepath)

    first_round_prompt = grounded_element_prompt + propose_action_prompt + user_manual_text

    model = GPT4O()
    response = model.chat_with_text(first_round_prompt)

    """
    # the effect of filtering redundant actions is minimal. Error is mostly caused by redundant element names. 
    model = GPT4O()
    filter_action_prompt = load_string_from_file(filter_action_prompt_filepath)
    filter_action_prompt = filter_action_prompt.replace("xxxxx", response)
    filter_action_prompt = filter_action_prompt.replace("yyyyy", user_manual_text)
    filter_action_prompt = filter_action_prompt.replace("zzzzz", grounded_element_prompt)
    response = model.chat_with_text(filter_action_prompt)
    """
    # save response into designated filepath
    with open(proposed_actions_filepath, "w") as f:
        f.write(response)
        print(f"Executable actions saved to {proposed_actions_filepath}")
    
    return response 


def batch_define_action_names(user_manual_root_dir, visually_grounded_elements_root_dir, propose_action_prompt_filepath, filter_action_prompt_filepath, proposed_actions_root_dir):
    # list all files in the user manual root dir
    machine_type_dirs = [os.path.join(user_manual_root_dir, d) for d in os.listdir(user_manual_root_dir) if os.path.isdir(os.path.join(user_manual_root_dir, d))]

    for dir1 in machine_type_dirs:
        machine_id_dirs = [os.path.join(dir1, d) for d in os.listdir(dir1) if d.endswith(".txt")]
        
        
        for user_manual_filepath in machine_id_dirs:
            #if not any(substring in user_manual_filepath for substring in ["_2_washing_machine/_2.txt"]):
            #    continue
            print("processing: ", user_manual_filepath)
            machine_type, machine_id = user_manual_filepath.split("/")[-2:]
            machine_id = machine_id.split(".")[0].split("_")[-1]
            visually_grounded_elements_filepath = os.path.join(visually_grounded_elements_root_dir, f"{machine_type}/{machine_id}.json")
            proposed_actions_filepath = os.path.join(proposed_actions_root_dir, f"{machine_type}/_{machine_id}.txt")
            create_or_replace_path(proposed_actions_filepath)
            define_action_names(user_manual_filepath, visually_grounded_elements_filepath, propose_action_prompt_filepath, filter_action_prompt_filepath, proposed_actions_filepath)
            #exit()
    pass

if __name__ == "__main__":

    # srun -u -o "log.out" --mem=20000 --gres=gpu:1 --cpus-per-task=8 --job-name “t2a” python3 

    #######################################
    #
    # Extract Action Names from User Manual
    #
    #######################################
    user_manual_filepath = os.path.expanduser('~/TextToActions/dataset/simulated/_1_user_manual/_2_text')

    visually_ground_elements_filepath = os.path.expanduser('~/TextToActions/dataset/simulated/_4_visual_grounding/_0_control_panel_element_bbox/_3_proposed_control_panel_element_bbox')

    propose_action_prompt_filepath = os.path.expanduser('~/TextToActions/code/simulated/prompts/_10_create_action_names.txt')

    filter_action_prompt_filepath = os.path.expanduser('~/TextToActions/code/simulated/prompts/_10_filter_action_names.txt')

    proposed_actions_root_dir = os.path.expanduser('~/TextToActions/dataset/simulated/_4_visual_grounding/_1_action_names/_1_proposed_action_names')

    batch_define_action_names(user_manual_filepath, visually_ground_elements_filepath, propose_action_prompt_filepath, filter_action_prompt_filepath, proposed_actions_root_dir)
