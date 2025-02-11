# 1. grade the visual grounding results:
# a rough estimate, as the TP can be wrong,
# total score is the total number of oracle actions 

import os 
import TextToActions.code.simulated.samples._0_logic_units as _0_logic_units

import json
from utils.create_or_replace_path import create_or_replace_path
from PIL import Image, ImageDraw, ImageFont
from _3_detect_bbox_from_photos import group_boxes_within_distance, calculate_area
from foundation_models.owlv2_crane5_query import OWLViT, visualize_image

def get_precision_and_recall(TP, FP, FN):
    if TP + FP == 0:
        precision = 0
    else: 
        precision = TP / (TP + FP)
    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)

    return precision, recall
def evaluate_visual_grounding_action_result(proposed_action_to_oracle_action_mapping_filepath, oracle_actions_filepath):
    with open(proposed_action_to_oracle_action_mapping_filepath, "r") as f:
        proposed_action_to_oracle_action_mapping = json.load(f)
    with open(oracle_actions_filepath, "r") as f:
        oracle_actions = json.load(f)

    
    TP = 0
    FP = 0
    FN = 0
    fn_names = []
    total_action = len(oracle_actions)
    # TP: the oracle action is present in the first choice of the grounded actions (true value is lower)
    # FP: the actions are proposed but cannot be grounded to any oracle actions (true value is higher)
    # FN: the oracle action is not present in the first choice of the grounded actions (true value is higher)
    # precision: TP / (TP + FP) (true value is lower)
    # recall: TP / (TP + FN) (true value is lower)

    for oracle_action in oracle_actions:
        
        oracle_action_name = oracle_action["action"]
        found = False
        for proposed_action in proposed_action_to_oracle_action_mapping:
            
            if len(proposed_action_to_oracle_action_mapping[proposed_action])>0 and oracle_action_name in proposed_action_to_oracle_action_mapping[proposed_action][0]:
                TP += 1
                found = True 
        if not found:
            FN += 1
            fn_names += [oracle_action_name]

    
    for proposed_action in proposed_action_to_oracle_action_mapping:
        if len(proposed_action_to_oracle_action_mapping[proposed_action]) == 0:
            FP += 1

    precision, recall = get_precision_and_recall(TP, FP, FN)

    return total_action, TP, FP, FN, precision, recall, fn_names


def batch_evaluate_visual_grounding_action_result(oracle_actions_dir, proposed_actino_to_oracle_action_mapping_dir, visual_grounding_action_score_dir):

    create_or_replace_path(visual_grounding_action_score_dir)
    result = []
    machine_level_result = []
    overall_result = []
    machine_type_list = [f for f in os.listdir(oracle_actions_dir) if os.path.isdir(os.path.join(oracle_actions_dir, f))]
    overall_TP = 0
    overall_FP = 0
    overall_FN = 0
    overall_total_action = 0
    overall_fn_names = []
    for machine_type in machine_type_list:
        machine_level_TP = 0
        machine_level_FP = 0
        machine_level_FN = 0
        machine_level_total_action = 0
        machine_level_fn_names = []
        machine_id_files = [f for f in os.listdir(os.path.join(oracle_actions_dir, machine_type)) if f.endswith(".json")]
        for machine_id_file in machine_id_files:
            machine_id = machine_id_file.split(".")[0]
            proposed_action_to_oracle_action_mapping_filepath = os.path.join(proposed_action_to_oracle_action_mapping_dir, machine_type, machine_id + ".json")
            oracle_actions_filepath = os.path.join(oracle_actions_dir, machine_type, machine_id + ".json")

            
            print("Evaluating visual grounding action result for machine type: {}, machine id: {}".format(machine_type, machine_id))
            total_action, TP, FP, FN, precision, recall, fn_names = evaluate_visual_grounding_action_result(proposed_action_to_oracle_action_mapping_filepath, oracle_actions_filepath)
            machine_level_TP += TP
            machine_level_FP += FP
            machine_level_FN += FN
            machine_level_total_action += total_action
            machine_level_fn_names += fn_names
            overall_TP += TP
            overall_FP += FP
            overall_FN += FN
            overall_total_action += total_action
            overall_fn_names += fn_names
            datapoint_result = {"machine_type": machine_type,
                    "machine_id": machine_id,
                    "total_action": total_action,
                    "TP": TP,
                    "FP": FP,
                    "FN": FN,
                    "precision": precision,
                    "recall": recall,
                    "fn_actions": fn_names
                }
            result.append(datapoint_result)
        machine_level_precision, machine_level_recall = get_precision_and_recall(machine_level_TP, machine_level_FP, machine_level_FN)
        datapoint_result = {"machine_type": machine_type,
                "machine_level_total_action": machine_level_total_action,
                "machine_level_TP": machine_level_TP,
                "machine_level_FP": machine_level_FP,
                "machine_level_FN": machine_level_FN,
                "machine_level_precision": machine_level_precision,
                "machine_level_recall": machine_level_recall,
                "machine_level_fn_actions": machine_level_fn_names
            }
        machine_level_result.append(datapoint_result)

    overall_precision, overall_recall = get_precision_and_recall(overall_TP, overall_FP, overall_FN)
    overall_result = {"overall_TP": overall_TP,
                      "overall_total_action": overall_total_action,
            "overall_FP": overall_FP,
            "overall_FN": overall_FN,
            "overall_precision": overall_precision,
            "overall_recall": overall_recall,
            "overall_fn_actions": overall_fn_names
        }
    instance_level_result_filepath = os.path.join(visual_grounding_action_score_dir, "_1_instance_level.json")
    create_or_replace_path(instance_level_result_filepath)
    with open(instance_level_result_filepath, "w") as f:
        json.dump(result, f, indent=4)
    
    machine_level_result_filepath =  os.path.join(visual_grounding_action_score_dir, "_2_machine_type_level.json") 
    create_or_replace_path(machine_level_result_filepath)
    with open(machine_level_result_filepath, "w") as f:
        json.dump(machine_level_result, f, indent=4)
    
    overall_result_filepath = os.path.join(visual_grounding_action_score_dir, "_3_overall.json") 
    create_or_replace_path(overall_result_filepath)
    with open(overall_result_filepath, "w") as f:
        json.dump(overall_result, f, indent=4)

def add_or_update_item(items, new_item):
    """
    Adds a new item to the list or updates the label of an existing item if the bbox matches.

    :param items: List of dictionaries, each containing 'bbox' and 'label'.
    :param new_item: Dictionary with 'bbox' and 'label' to add or merge.
    :return: None (the list is modified in place)
    """
    for item in items:
        if item['bbox'] == new_item['bbox']:
            # If the bbox matches, concatenate the labels and return
            item['label'] += "\n" + new_item['label']
            return False
    # If no existing bbox matches, add the new item to the list
    items.append(new_item)
    return True

def batch_visualise_actions(oracle_action_mapping_dir, control_panel_images_dir, visualisation_output_dir):
    machine_type_list = [f for f in os.listdir(oracle_action_mapping_dir) if os.path.isdir(os.path.join(oracle_action_mapping_dir, f))]
    for machine_type in machine_type_list:
        machine_id_files = [f for f in os.listdir(os.path.join(oracle_action_mapping_dir, machine_type)) if f.endswith(".json")]
        for machine_id_file in machine_id_files:
            machine_id = machine_id_file.split(".")[0]
            oracle_action_mapping_filepath = os.path.join(oracle_action_mapping_dir, machine_type, machine_id + ".json")
            control_panel_image_machine_type_dir = os.path.join(control_panel_images_dir, machine_type)
            visualisation_output_machine_type_dir = os.path.join(visualisation_output_dir, machine_type)
            visualise_actions(oracle_action_mapping_filepath, control_panel_image_machine_type_dir, visualisation_output_machine_type_dir, machine_id)

    pass 


def add_text_to_the_right_of_image(image, text, text_width = 100):
    """
    Adds a chunk of text below an image.

    :param image: A PIL Image object.
    :param text: The text string to add below the image.
    :param text_height: The height of the text area in pixels.
    :return: A new PIL Image object with the text added.
    """
    width, height = image.size

    # Create a new image with extra space for the text
    new_image = Image.new('RGB', (width + text_width, height ), (255, 255, 255))
    new_image.paste(image, (0, 0))

    # Initialize ImageDraw
    draw = ImageDraw.Draw(new_image)

    # Load a default font or a specified font with size
    font = ImageFont.load_default()

    # Calculate text position for horizontal centering and padding from the bottom
    _, _, text_width, text_height = draw.textbbox((0, 0), text=text, font=font)

    text_x = width + 10 
    text_y = (height - text_height) / 2 

    # Draw the text
    draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0))

    return new_image

def visualise_actions(oracle_action_mapping_filepath, image_path, visualisation_output_machine_type_dir):
    # specific to each machine instance 

    # load the oracle actions
    with open(oracle_action_mapping_filepath, "r") as f:
        oracle_actions = json.load(f)
    
    # iterate over the images 
    # data structure:
    # { "bbox": [x1, y1, x2, y2], 
    #   "label": "action_name + action_name"
    # }

    annotation_info = []
    id = 0
    annotation_text = ""
    for oracle_action in oracle_actions:
        print("oracle_action: ", oracle_action)
        bboxes = oracle_action["bboxes"]
        action_name = oracle_action["action"]
        for item in bboxes:
            bbox_list = item["bbox"]
            for bbox in bbox_list:
                new_item = {"bbox": bbox, "label": action_name, "id": id}
                add_id = add_or_update_item(annotation_info, new_item)
                if add_id:
                    annotation_text += f"{id}: {action_name}\n"
                    id += 1
                        
    # save the annotation_info into a json file 
    annotation_info_filepath = os.path.join(visualisation_output_machine_type_dir, + "annotation.json")
    create_or_replace_path(annotation_info_filepath)
    with open(annotation_info_filepath, "w") as f:
        json.dump(annotation_info, f, indent=4)
    
    annotation_info = sorted(annotation_info, key=lambda x: calculate_area(x['bbox']))
    bbox_group = group_boxes_within_distance(annotation_info, distance_threshold=0.1)
    for i, group in enumerate(bbox_group):
        image = Image.open(image_path)
        annotated_image = visualize_image(image, bboxes = [obj['bbox'] for obj in group], labels = [obj['label'] for obj in group], show=False, return_img= True, rect_color="green", text_color="green")
        output_savepath = os.path.join(visualisation_output_machine_type_dir,str(i) + ".png")
        create_or_replace_path(output_savepath)
        print("output_savepath: ", output_savepath)
        annotated_image.convert('RGB').save(output_savepath)

    """
    example oracle file: a list of items below:
    {
        "action": "turn_power_selector_dial_clockwise",
        "bbox_label": [
            "0_power_dial"
        ],
        "action_type": "turn_dial_clockwise",
        "bboxes": [
            {
                "image_name": "0_1",
                "bbox": [
                    [
                        0.4,
                        0.2843501326259947,
                        0.6613207547169812,
                        0.41750663129973475
                    ]
                ]
            },
        ]
    }
    """
    pass 
if __name__ == "__main__":

    # srun -u -o "log.out" --mem=20000 --gres=gpu:1 --cpus-per-task=8 --job-name “t2a” python3 

    ############################################
    # 
    #  1. evaluation grounding action result
    # 
    ############################################
    proposed_action_to_oracle_action_mapping_dir = os.path.expanduser('~/TextToActions/dataset/_4_visual_grounding/_6_proposed_to_oracle_action_mapping')

    oracle_action_mapping_dir = os.path.expanduser('~/TextToActions/dataset/_3_simulators/_3_oracle_simulator_action_to_bbox_mapping')

    visual_grounding_action_score_dir = os.path.expanduser('~/TextToActions/dataset/_6_results/_1_visual_grounding_action_results')

    proposed_action_to_oracle_action_mapping_filepath = os.path.join(proposed_action_to_oracle_action_mapping_dir, "_1_microwave/0.json")

    oracle_actions_filepath = os.path.join(oracle_action_mapping_dir, "_1_microwave/0.json")


    batch_evaluate_visual_grounding_action_result(oracle_action_mapping_dir, proposed_action_to_oracle_action_mapping_dir, visual_grounding_action_score_dir)

    ############################################
    # 
    # 2. visualise oracle actions
    # 
    ############################################

    control_panel_images_dir = os.path.expanduser('~/TextToActions/dataset/_2_control_panel_images/_0_raw')
    visualisation_oracle_output_dir = os.path.expanduser('~/TextToActions/dataset/_6_results/_2_visualise_oracle_actions')
    #batch_visualise_actions(oracle_action_mapping_dir, control_panel_images_dir, visualisation_oracle_output_dir)

    ############################################
    #
    # 3. visualise generated actions
    #
    ############################################

    visualisation_proposed_action_output_dir = os.path.expanduser('~/TextToActions/dataset/_6_results/_3_visualise_proposed_actions')

    proposed_actions_bbox_dir = os.path.expanduser('~/TextToActions/dataset/_4_visual_grounding/_5_proposed_world_model_action_bbox')
    #batch_visualise_actions(proposed_actions_bbox_dir, control_panel_images_dir, visualisation_proposed_action_output_dir)

