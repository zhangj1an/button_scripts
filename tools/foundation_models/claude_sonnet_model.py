import anthropic
import base64
from io import BytesIO
from PIL import Image
import os
import requests


class ClaudeSonnet:
    def __init__(self,  model_name="claude-3-5-sonnet-20240620", max_tokens=1024, temperature=0.7):
        self.api_key = ""
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

    @staticmethod
    def image_to_base64(image_path, compressed_image):
        ext = os.path.splitext(image_path)[1].lower()
        format = 'JPEG' if ext in ['.jpg', '.jpeg'] else 'PNG' if ext == '.png' else 'GIF' if ext == '.gif' else 'WEBP' if ext == '.webp' else 'JPEG'
        image = compressed_image.convert('RGB')  # Convert image to RGB to ensure compatibility
        buffered = BytesIO()
        image.save(buffered, format=format)

        size_before = buffered.tell()
        base64_encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')
        size_after = len(base64_encoded.encode('utf-8'))
        return base64_encoded

    @staticmethod
    def get_image_type(path):
        ext = os.path.splitext(path)[1].lower()
        return {
            '.png': 'image/png',
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }.get(ext, 'image/jpeg')

    @staticmethod
    def resize_image_to_target_size(original_image, image_path, target_size_mb=3.0, step_percentage=10): # target mb = 3.4
        target_size = target_size_mb * 1024 * 1024  # Target size in bytes
        image_format = 'JPEG' if original_image.format == 'JPEG' else 'PNG'
        quality = 85  # Fixed quality for initial attempts
        
        # Check initial size of the image
        initial_buf = BytesIO()
        original_image.save(initial_buf, format=image_format, quality=quality)
        initial_size = initial_buf.tell()
        if initial_size <= target_size:
            return original_image  # Return original if already under or equal to the target size

        
        # Start with an initial scale of 100% and decrease gradually
        scale = 100
        buf = BytesIO()
        while True:
            # Calculate new dimensions
            new_width = int(original_image.width * scale / 100)
            new_height = int(original_image.height * scale / 100)
            
            # Resize from original and save to buffer
            resized_image = original_image.resize((new_width, new_height), Image.LANCZOS)
            buf.seek(0)
            buf.truncate()
            resized_image.save(buf, format=image_format, quality=quality)
            size = buf.tell()
            
            # Check if the size is within the desired range
            if size < target_size:
                break
            else:
                # Decrease scale by step_percentage
                scale -= step_percentage
        
        
        # If under target size, incrementally increase size to maximize dimensions
        increment = step_percentage / 2
        while size < target_size and increment >= 1:
            scale += increment
            new_width = int(original_image.width * scale / 100)
            new_height = int(original_image.height * scale / 100)
            resized_image = original_image.resize((new_width, new_height), Image.LANCZOS)
            buf.seek(0)
            buf.truncate()
            resized_image.save(buf, format=image_format, quality=quality)
            new_size = buf.tell()
            
            if new_size > target_size:
                break
            else:
                size = new_size
        
        image_name = image_path.split("/")[-1]
        print(f"resizing image {image_name} with scale {scale}%")
        # Load the final adjusted image from buffer
        buf.seek(0)
        final_image = Image.open(buf)
        return final_image

    def chat_with_text(self, prompt):
        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            )
            response = message.content[0].text
            return response
        except Exception as e:
            print(f"Error processing text: {str(e)}")
    
    def chat_with_context(self, context, new_message, conversation_history, context_window = 2):
        if conversation_history is None:
            conversation_history = []

        if new_message is not None:
            # Append new message to the conversation history
            conversation_history.append({"role": "assistant", "content": "ok."})
            conversation_history.append({"role": "user", "content": new_message})
        
        if len(conversation_history) > context_window:
            conversation_history.pop(0)
            conversation_history.pop(0)

        try:
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": context},
                ] + conversation_history
            )
            response = message.content[0].text
            return response, conversation_history
        except Exception as e:
            print(f"Error processing context: {str(e)}")
    def chat_with_multiple_images(self, image_filepaths, prompt):
        try:
            message_content = [{"type": "text", "text": prompt}]
            for image_path in image_filepaths:
                media_type = self.get_image_type(image_path)
                image = Image.open(image_path)
                
                compressed_image = self.resize_image_to_target_size(image, image_path)
                
                image_data = self.image_to_base64(image_path, compressed_image)
                message_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_data
                    }
                })

            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature, 
                messages=[{"role": "user", "content": message_content}]
            )
            response = message.content[0].text
            return response
        except Exception as e:
            print(f"Error processing image: {str(e)}")


if __name__ == "__main__":

    task_prompt = """
    The elements listed above are responsible for the operationd feedback of the appliance's control panel. They represent names of buttons, dials, switches and digital displays. Please ignore the indicators.

    I will input three images of the control panel, with a bounding box index: xxxxx. 

    The first image is the original photo. The second image contains a zoomed in version of a certain part of the appliance, and contain red bounding box and some green bounding boxes. The third image represents the same zoomed in region, but contains no bounding boxes. I want to firstly ask a question: whether the red bounding box circles out an object mentioned in the above list. If the answer is yes, then I want to ask a second question. If any green bounding box is selecting the same object as the red bounding box, then compared to the green bounding boxes, is the red bounding box a good choice to represent the object of your choice? As long as the red bounding box is a better choice than the green bounding boxes, then the answer is yes. It is okay for the red boxes to only circle out partial of the object.

    The criteria of best choice is as follows. For dials, only select the bounding box that covers the dial knob, and ignore bounding boxes that select the labellings. For buttons, if the button consists of both symbols and text, choose the bounding box that circles out the symbol. 

    If both answer is yes, then output the control panel name, followed by the bounding box index: xxxxx. For example:
    <control_element_name> : <index>

    If the red bounding box can be mapped to multiple element names, then output like this:
    <control_element_name> : <index>
    <control_element_name> : <index>
    ...

    Please copy the control element name exactly as listed.

    Otherwise, reply "None".

    Please do not return anything else.


    """

    image_raw_filepath = "/data/home/jian/RLS_microwave/benchmark_3/_2_control_panel_images/_1_selected/_1_microwave/1_0.jpeg"
    image_labelled_filepath = "/data/home/jian/RLS_microwave/benchmark_3/_2_control_panel_images/_4_query_images/_1_microwave/1/0/2_0.png"
    image_cropped_filepath = "/data/home/jian/RLS_microwave/benchmark_3/_2_control_panel_images/_4_query_images/_1_microwave/1/0/2_1.png"

    query_index = "2"

    control_element_list_filepath = "/data/home/jian/RLS_microwave/benchmark_3/_1_user_manual/_3_extracted_control_panel_element_names/_1_microwave/_1.txt"
    with open(control_element_list_filepath, "r") as f:
        control_element_list = f.read()

    prompt = control_element_list + task_prompt.replace("xxxxx", query_index)

    image_filepaths = [image_raw_filepath, image_labelled_filepath, image_cropped_filepath]
    chat_with_images(image_filepaths, prompt)