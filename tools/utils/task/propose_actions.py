import os
import sys
import json
import textwrap
sys.path.append(os.path.expanduser("~/TextToActions/code"))
from foundation_models.gpt_4o_model import GPT4O
from simulated.utils.task.interact_with_simulator import interact_with_appliance_simulator, interact_with_user_manual_simulator, parse_action
from simulated.utils.extract_python_code import extract_python_code
from simulated.utils.task.matches_run_action_format import matches_run_action_format

def format_check_action_effect(action_tuples):
    # the same action can only have one effect 
    # the effect can only be either "prev" or "next"
    try:
        # create a set to check for duplicates of action name
        action_name_set = set()
        for action in action_tuples:
            action_name = action[0]
            # check if action name exists in the set 
            if action_name in action_name_set:
                return False, f"Duplicate action name: {action_name}"
            action_effect = action[1] 
            if action_effect not in ["prev", "next"]:
                return False, f"Wrong proposed action effect: {action_effect}" 
        return True, ""
    except Exception as e:
        print(f"Exception occurred: {e}")
        return False, f"Wrong proposed action effect: {e}"
    # return valid_flag, and error_message 

def generate_action_effects(prompt, user_manual_text, defined_variables_code, action_option_text, command_string, adjusting_variable_value, expected_feedback, past_actions, latest_feedback):

    replacement_dict = {
        "xxxxx": user_manual_text,
        "uuuuu": defined_variables_code,
        "hhhhh": action_option_text,
        "zzzzz": command_string,
        "yyyyy": adjusting_variable_value,
        "mmmmm": expected_feedback,
        "qqqqq": past_actions,
        "wwwww": latest_feedback
    }

    
    for key, value in replacement_dict.items():
        formatted_value = json.dumps(value, indent=4) if isinstance(value, dict) else str(value)
        prompt = prompt.replace(key, formatted_value)

    
    times = 0
    max_attempts = 3  # Adjust attempts based on setting
    action_tuples = []
    while times < max_attempts:
        model = GPT4O()
        
        response, conversation_history = model.chat_with_context(prompt, None, None)
        
        response = extract_python_code(response)
        try:
                action_proposal_ns = {}
                exec(response, action_proposal_ns)
                action_tuples = action_proposal_ns.get("action_tuples", [])
                # here need to add a format check, the second time must be either "next" or "prev" for each tuple. 
                # the same action cannot have two effects.
                valid_flag, error_message = format_check_action_effect(action_tuples) 
                if valid_flag:
                    return action_tuples
                else:
                    print(f"trial {times} fails when generating action effects: {error_message}") 
                
        except:
            print("Exception occurred")
            
        times += 1
    
    
    if len(action_tuples) == 0:
        print("No valid action generated")
    return action_tuples

def propose_action(
    appliance_simulator, 
    prompt_template, 
    replacement_dict, 
    grounding_mapping_filepath, 
    setting,  # "NM_NR_LP" or "M_UR_LP"
    initial_setup=False, 
    prompt_initialised_context=None, 
    conversation_history=None, 
    verbose=False
):
    

    prompt = prompt_template
    for key, value in replacement_dict.items():
        formatted_value = json.dumps(value, indent=4) if isinstance(value, dict) else str(value)
        prompt = prompt.replace(key, formatted_value)
    
    satisfied = False
    times = 0
    max_attempts = 3  # Adjust attempts based on setting
    variable_reason = ""
    variable_response_string = ""
    while not satisfied and times < max_attempts:
        model = GPT4O()
        
        if initial_setup:
            response, conversation_history = model.chat_with_context(prompt, None, None)
        else:
            response, conversation_history = model.chat_with_context(prompt_initialised_context, prompt, conversation_history)
        
        # if sr and initial setup, conversation history should be: the entire prompt 

        # if sr and not initial setup, conversation history should be: the current state simulator, the proposed action, the expected feedback.

        # required return: appliance simulator, conversation_history(prob no need), termination_flag, current_observation, prompt_initialised_context, proposed_action, grounded_action, expected_feedback 
        response = extract_python_code(response)
        print("Response: \n", response)
        with open("temp.txt", "w") as f:
            f.write(prompt)
        
        if verbose:
            print("Proposing action...")
            print(response)

        if setting == "NR":
            satisfied = matches_run_action_format(response)
        elif setting == "UR_LP":
            try:
                action_proposal_ns = {}
                exec(response, action_proposal_ns)
                variable_reason = action_proposal_ns.get("variable_reason", "")
                variable_response_string = action_proposal_ns.get("variable_response_string", "")
                reason = action_proposal_ns.get("reason", "")
                satisfied = matches_run_action_format(variable_response_string)
                
            except Exception as e:
                print("Exception occurred during action proposal: {e}")
                satisfied = False
        elif setting == "UR_MA":
            try: 
                action_proposal_ns = {}
                exec(response, action_proposal_ns)
                variable_next_action = action_proposal_ns.get("variable_next_action", "")
                adjusting_variable_name = action_proposal_ns.get("adjusting_variable_name", "")
                expected_feedback = action_proposal_ns.get("expected_feedback", "")
                reason = action_proposal_ns.get("reason", "")
                satisfied = matches_run_action_format(variable_next_action)

            except Exception as e:
                print(f"Exception occurred: {e}")
                satisfied = False
            pass 
        elif setting == "SR":
            try: 
                # variable_next_action
                action_proposal_ns = {}
                exec(response, action_proposal_ns)
                #pass_check = True
                proposed_action = action_proposal_ns.get("proposed_action", "")
                expected_feedback = action_proposal_ns.get("expected_feedback", "")
                reason = action_proposal_ns.get("reason", "")
                satisfied = matches_run_action_format(proposed_action)
                #print(f"Parsing response : \n Proposed action: {proposed_action}, expected_feedback: {expected_feedback} ")
                
            except Exception as e:
                print(f"Exception occurred: {e}")
                satisfied = False
                pass
        
        else:
            print("Unknown setting")

        if "end" in response:
            if setting == "NR":
                return (appliance_simulator, conversation_history, True, "", prompt, "end", "end") 
            elif setting == "UR_LP":
                return (appliance_simulator, conversation_history, True, "", prompt, "end", variable_reason, "end")
            elif setting == "UR_MA":
                return (appliance_simulator, True, "", prompt, "end", "end", adjusting_variable_name, expected_feedback, reason)
            elif setting == "SR":
                return (appliance_simulator, True, "", prompt, "end", "end", expected_feedback, reason)
            else:
                print("unknown task setting")
            
        
        times += 1
        del model


    if not satisfied:
        if setting == "NR":
            return (appliance_simulator, conversation_history, True, "", prompt, response, "")
        elif setting == "UR_LP":
            return (appliance_simulator, conversation_history, True, "", prompt, variable_response_string, variable_reason, "")
        elif setting == "UR_MA":
            return (appliance_simulator, True, "", prompt, variable_next_action, "", adjusting_variable_name, expected_feedback, reason)
        elif setting == "SR":
            #return (proposed_action, expected_feedback, True)
            return simulator, True, "", prompt, proposed_action, "", expected_feedback, reason 
            #parsed_actions = []
            #for action in proposed_actions:
            #    parsed_actions.append(parse_action(action))
            #return (parsed_actions, True)
        else:
            print("unknown task setting")

    
    if setting == "NR":
        proposed_action = response
    elif setting == "UR_LP":
        proposed_action = variable_response_string
        print("Reason in context: ", variable_reason)
    elif setting == "UR_MA":
        proposed_action = variable_next_action
        #print("Expected feedback in context: ", adjusting_variable_name, expected_feedback)
    elif setting == "SR":
        pass
        #print(f"Proposed action: {proposed_action}, expected_feedback: {expected_feedback} ")
        #return (proposed_action, expected_feedback, False)

    if verbose:
        print("Proposed action: ", proposed_action)
    
    #print("Proposed action in context : ", proposed_action)
    
    print("Proposed action in function propose_action: ", proposed_action)
    simulator, termination_flag, current_observation, grounded_action = interact_with_appliance_simulator(
        appliance_simulator, proposed_action, grounding_mapping_filepath, verbose
    )
    
    if setting == "NR":
        return simulator, conversation_history, termination_flag, current_observation, prompt, proposed_action, grounded_action
    elif setting == "UR_LP":
        return simulator, conversation_history, termination_flag, current_observation, prompt, proposed_action, variable_reason, grounded_action
    elif setting == "UR_MA":
        return simulator, termination_flag, current_observation, prompt, proposed_action, grounded_action, adjusting_variable_name, expected_feedback, reason
    elif setting == "SR":
        return simulator, termination_flag, current_observation, prompt, proposed_action, grounded_action, expected_feedback, reason
    else:
        print("unknown task setting")
