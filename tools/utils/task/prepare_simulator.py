import os
import sys
import textwrap
sys.path.append(os.path.expanduser("~/TextToActions/code"))
from simulated.utils.load_string_from_file import load_string_from_file, load_string_from_files

def load_generated_code(generated_files, setting = "agnostic"):
    with open("temp_generated.py", 'w') as f:
        code_content = load_string_from_files(generated_files)
        if setting == "agnostic":
            code_content = code_content.replace("goal_state = ExtendedSimulator()", "goal_state = copy.deepcopy(ExtendedSimulator())")
        elif setting == "specific":
            code_content = code_content.replace("goal_state = Simulator()", "goal_state = copy.deepcopy(Simulator())")
        import_statements = """
        import sys 
        import copy
        sys.path.append("/data/home/jian/TextToActions/code")
        from simulated.paper_exp.baselines_v0.temp_imports import *
        """
        import_statements = textwrap.dedent(import_statements)
        generated_appliance_code = import_statements + "\n" + code_content
        f.write(generated_appliance_code)
    return generated_appliance_code

def prepare_oracle_simulator(template_code_filepaths, oracle_appliance_code_filepath):
    template_code = load_string_from_files(template_code_filepaths)
    with open("temp_imports.py", 'w') as f:
        f.write(template_code)
    with open(oracle_appliance_code_filepath, 'r') as f:
        appliance_code = f.read()
        appliance_code = appliance_code.replace("Simulator", "Oracle")
    with open("temp_oracle.py", 'w') as f:
        import_statements = """
        import sys 
        import copy
        sys.path.append("/data/home/jian/TextToActions/code")
        from simulated.paper_exp.baselines_v0.temp_imports import *
        """
        import_statements = textwrap.dedent(import_statements)
        oracle_appliance_code = import_statements + "\n" + appliance_code
        f.write(oracle_appliance_code)
    return oracle_appliance_code

def prepare_generated_simulator(planner_filepath, generated_code_filepaths, output_filepaths, setting = "agnostic"):
    with open("temp_generated_variable.py", 'w') as f:
        f.write(load_string_from_file(generated_code_filepaths[0]))
        # include both var and feature 
    with open("temp_generated_feature.py", 'w') as f:
        f.write(load_string_from_file(generated_code_filepaths[1]))
    
    if setting == "agnostic":
        temp_world_model = load_string_from_file(generated_code_filepaths[2]) + "\n" +load_string_from_file(output_filepaths["world_model_specific_to_command"])
    elif setting == "specific":
        temp_world_model = load_string_from_file(generated_code_filepaths[2])
    
    with open("temp_generated_world_model.py", 'w') as f:
        f.write(temp_world_model)
    with open("temp_goal.py", 'w') as f:
        f.write(load_string_from_file(output_filepaths["goal_state"]))

    #? planner_filepath? 
    generated_files = ["temp_generated_variable.py", "temp_generated_feature.py", "temp_generated_world_model.py", "temp_goal.py", planner_filepath]
    generated_appliance_code = load_generated_code(generated_files, setting)
    return generated_appliance_code, generated_files

def load_oracle_generated_and_goal_simulator(oracle_appliance_code, generated_appliance_code, setting = "agnostic"):
    oracle_globals = {}
    generated_globals = {}

    # Execute oracle_appliance_code in its own dictionary
    exec(oracle_appliance_code, oracle_globals)
    ApplianceSimulator = oracle_globals.get('Oracle', None)
    # Execute generated_appliance_code in its own dictionary
    exec(generated_appliance_code, generated_globals)
    if setting == "agnostic":
        UserManualSimulator = generated_globals.get('ExtendedSimulator', None)
    elif setting == "specific":
        UserManualSimulator = generated_globals.get('Simulator', None)
    goal_state_simulator = generated_globals.get('goal_state', None)
    feature_sequence = generated_globals.get('feature_sequence', None)
    appliance_simulator = ApplianceSimulator()
    current_state_simulator = UserManualSimulator()
    AStarSearch = generated_globals.get('AStarSearch', None)
    return appliance_simulator, current_state_simulator, goal_state_simulator, feature_sequence, AStarSearch
