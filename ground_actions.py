# input an image and oracle list of actions, output a list of tuples that matches action names to bounding boxes. 

# then move the gripper to the first button, directly on top.


# for the offline version, write an official simulator first. 

import sys
import os 
sys.path.append(os.path.expanduser("~/TextToActions/code")) 
from tools.foundation_models.gpt_4o_model import GPT4O
import uvicorn 
from fastapi import FastAPI, HTTPException
from tools._3_detect_bbox_from_photos import make_query_images, process_image 
from tools._4_map_control_panel_element_names_to_bbox_indexes import ground_proposed_entities_in_batch, task_prompt as prompt_identify_control_panel_element_index 

from tools._5_json_map_control_panel_element_names_to_bbox_indexes import consolidate_mapping_results, meta_prompt as prompt_format_control_panel_element_index

from tools._6_remove_duplicate_bboxes import resolve_duplicate_bbox_id_for_one_instance, user_manual_prompt as remove_duplicate_bboxes_user_manual_prompt, task_prompt as remove_duplicate_bboxes_task_prompt

from tools._7_json_map_control_panel_element_names_to_bbox import map_element_to_bboxes as map_control_panel_element_names_to_bbox

from tools._8_visualise_grounding_control_element_name_result import visualise_entities as visualise_grounded_control_panel_elements

from tools._11_generated_grounded_action_json import prompt as prompt_map_actions_to_entities, generate_grounded_actions_for_proposed_world_model
# do all the things in one file 

from tools._13_visualisation_grounding_action_result import visualise_actions as visualise_grounded_actions


def ground_actions(image_filepath, unnamed_entity_bbox_filepath, unnamed_entity_visualisation_filepath, query_image_to_match_entity_to_names_folder,  oracle_control_panel_elements_list_filepath):
    # load the image
    # load the action list
    # run the model
    # return the list of tuples

    # identify the unnamed entities in the image,
    # only retain those likely to be a button/dial
    process_image(image_path=image_filepath, bbox_savepath = unnamed_entity_bbox_filepath, detected_image_path=unnamed_entity_visualisation_filepath)
    # prepare query images for GPT to match the entity to the action names
    make_query_images(image_path=image_filepath, bbox_savepath = unnamed_entity_bbox_filepath, query_image_root_dir = query_image_to_match_entity_to_names_folder) 
    # given a bbox, give it a control panel element name. might have duplicates
    ground_proposed_entities_in_batch(query_image_dir = query_image_to_match_entity_to_names_folder, control_panel_elements_from_user_manual = oracle_control_panel_elements_list_filepath, meta_prompt = prompt_identify_control_panel_element_index, visual_grounding_result_filepath=control_panel_element_index_filepath, raw_image_filepath = image_filepath)
    # format the result from txt to json
    consolidate_mapping_results(control_panel_element_index_filepath,
    control_panel_element_index_formatted_filepath, prompt_format_control_panel_element_index)

    # give button name unique bbox index
    resolve_duplicate_bbox_id_for_one_instance(input_json_filepath = control_panel_element_index_formatted_filepath, output_json_filepath=control_panel_element_index_unique_filepath, raw_image_filepath = image_filepath, query_image_save_filepath = query_image_for_unique_entity_id_folder, machine_type = machine_name, bbox_input_root_dir = unnamed_entity_bbox_filepath, user_manual_filepath = user_manual_filepath, task_prompt = remove_duplicate_bboxes_task_prompt, user_manual_prompt = remove_duplicate_bboxes_user_manual_prompt)

    # give button name the bbox 
    map_control_panel_element_names_to_bbox(index_mapping_filepath=control_panel_element_index_unique_filepath, located_bboxes_dir=unnamed_entity_bbox_filepath, output_filepath=proposed_control_panel_element_bbox_filepath)

    # visualise grounded entities
    visualise_grounded_control_panel_elements(grounded_element_bboxes_filepath= proposed_control_panel_element_bbox_filepath, control_panel_image_path= image_filepath, output_savepath = visualised_proposed_control_panel_element_bbox_filepath)

    # map action names to grounded entity names
    generate_grounded_actions_for_proposed_world_model(prompt=prompt_map_actions_to_entities, input_action_option_filepath= oracle_action_list_filepath, input_grounded_element_bboxes_filepath=proposed_control_panel_element_bbox_filepath, output_proposed_world_model_action_bboxes_filepath=grounded_action_bbox_from_oracle_names_filepath)

    visualise_grounded_actions(oracle_action_mapping_filepath=grounded_action_bbox_from_oracle_names_filepath, image_path=image_filepath, visualisation_output_machine_type_dir=visualised_proposed_actions_dir)


    pass

if __name__ == "__main__":

    #########################################################
    # right now use oracle button name and action name first. 
    # later add the generation process. 
    #########################################################
    machine_name = "water_dispenser"
    # Get the absolute path of the parent directory "button_script"
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Current script's directory
    parent_dir = os.path.join(script_dir, "..")  # Navigate one level up to "button_script"
    parent_dir = os.path.abspath(parent_dir)  # Normalize the path
    image_filepath = os.path.join(parent_dir, f"data/{machine_name}/_0_image/test_rgb.png")
    oracle_control_panel_elements_list_filepath = os.path.join(parent_dir, f"data/{machine_name}/_1_oracle/_1_control_panel_elements.txt")
    oracle_action_list_filepath = os.path.join(parent_dir, f"data/{machine_name}/_1_oracle/_2_action_names.txt")
    
    unnamed_entity_bbox_filepath = os.path.join(parent_dir, f"data/{machine_name}/_3_visual_grounding/_0_unnamed_entity_bbox.json")
    unnamed_entity_visualisation_filepath = os.path.join(parent_dir, f"data/{machine_name}/_3_visual_grounding/_1_unnamed_entity_visualisation.png")
    verify_entity_validity_folder = os.path.join(parent_dir, f"data/{machine_name}/_3_visual_grounding/_2_verify_entity_validity")
    query_image_to_match_entity_to_names_folder = os.path.join(parent_dir, f"data/{machine_name}/_3_visual_grounding/_3_query_image_to_match_entity_to_names")

    control_panel_element_index_filepath = os.path.join(parent_dir, f"data/{machine_name}/_3_visual_grounding/_4_control_panel_element_index.txt")

    control_panel_element_index_formatted_filepath = os.path.join(parent_dir, f"data/{machine_name}/_3_visual_grounding/_5_control_panel_element_index.json")

    control_panel_element_index_unique_filepath = os.path.join(parent_dir, f"data/{machine_name}/_3_visual_grounding/_6_control_panel_element_index_unique.json")

    query_image_for_unique_entity_id_folder = os.path.join(parent_dir, f"data/{machine_name}/_3_visual_grounding/_7_query_image_for_unique_entity_id")

    user_manual_filepath = os.path.join(parent_dir, f"data/{machine_name}/_1_oracle/_0_user_manual.txt")

    proposed_control_panel_element_bbox_filepath = os.path.join(parent_dir, f"data/{machine_name}/_3_visual_grounding/_8_proposed_control_panel_element_bbox.json")

    visualised_proposed_control_panel_element_bbox_filepath = os.path.join(parent_dir, f"data/{machine_name}/_3_visual_grounding/_9_visualised_proposed_control_panel_element_bbox")

    grounded_action_bbox_from_oracle_names_filepath = os.path.join(parent_dir, f"data/{machine_name}/_3_visual_grounding/_10_grounded_action_bbox_from_oracle_names.json")

    visualised_proposed_actions_dir = os.path.join(parent_dir, f"data/{machine_name}/_3_visual_grounding/_11_visualised_proposed_actions")
    








