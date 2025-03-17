# Steps to Ground Actions on Appliances

#### Prerequisite: Launch OWLv2 for detecting control panel bboxes 
This file must be run before running the actual `ground_action.py` file.

navigate to `tools/foundation_models`, then run 

```
srun -u -o "api-owlv2-log.out" -w crane5 --mem=20000 --gres=gpu:1 --cpus-per-task=4 --time=03:00:00 --job-name "owlv2" uvicorn owlv2_crane5_api:app --host=0.0.0.0 --port=4229 --reload --loop asyncio
```

Currently the time is set to 3 hours.

Inside the file named `api-owlv2-log.out`, this print statement shows the OWLv2 is ready: "Application startup complete." 

#### Add API key to GPT-4o model 
open file `tools/foundation_models/gpt_4o_model.py`, fill in line 19:

```
os.environ["OPENAI_API_KEY"] = ""
```

`tools/foundation_models/claude_sonnet_model.py` is also called in `resolve_duplicate_bbox_id_for_one_instance()`, but is not used. Not sure if needs to add API key.

#### Required inputs
The appliance require user manual and an observation file with png extension. Put it in `data/{water_dispenser}/_0_input`. 

Sample user manual file: `_0_pdf.pdf` (can be any filename)
Sample observation file: `0.png` (should be an index)



#### Output formats

The output folder is at `data/{water_dispenser}/output_{0}` (0 is the number from observation file name).

The bounding box of the grounded actions is located at `{output_folder}/_3_visual_grounding/_1_action_names/_2_proposed_action_bbox.json`

#### Ground Actions 
at root directory, run 

```
srun -u -o "log.out" -w crane2 --mem=20000 --gres=gpu:1 --cpus-per-task=8 --job-name “vlm” python3 ground_actions.py
```

![Sample grounding result](data/water_dispenser/sample_output/_3_visual_grounding/_1_action_names/_3_visualised_proposed_actions.png)