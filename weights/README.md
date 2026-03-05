---
library_name: transformers
license: mit
pipeline_tag: image-text-to-text
---
📢 [[Project Page](https://microsoft.github.io/OmniParser/)] [[Blog Post](https://www.microsoft.com/en-us/research/articles/omniparser-for-pure-vision-based-gui-agent/)] [[Demo](https://huggingface.co/spaces/microsoft/OmniParser/)] 

# Model Summary
OmniParser is a general screen parsing tool, which interprets/converts UI screenshot to structured format, to improve existing LLM based UI agent. 
Training Datasets include: 1) an interactable icon detection dataset, which was curated from popular web pages and automatically annotated to highlight clickable and actionable regions, and 2) an icon description dataset, designed to associate each UI element with its corresponding function. 

This model hub includes a finetuned version of YOLOv8 and a finetuned BLIP-2 model on the above dataset respectively. For more details of the models used and finetuning, please refer to the [paper](https://arxiv.org/abs/2408.00203).

# Responsible AI Considerations
## Intended Use
- OmniParser is designed to be able to convert unstructured screenshot image into structured list of elements including interactable regions location and captions of icons on its potential functionality. 
- OmniParser is intended to be used in settings where users are already trained on responsible analytic approaches and critical reasoning is expected. OmniParser is capable of providing extracted information from the screenshot, however human judgement is needed for the output of OmniParser. 
- OmniParser is intended to be used on various screenshots, which includes both PC and Phone, and also on various applications.  
## limitations
- OmniParser is designed to faithfully convert screenshot image into structured elements of interactable regions and semantics of the screen, while it does not detect harmful content in its input (like users have freedom to decide the input of any LLMs), users are expected to provide input to the OmniParser that is not harmful. 
- While OmniParser only converts screenshot image into texts, it can be used to construct an GUI agent based on LLMs that is actionable. When developing and operating the agent using OmniParser, the developers need to be responsible and follow common safety standard. 
- For OmniPaser-BLIP2, it may incorrectly infer the gender or other sensitive attribute (e.g., race, religion etc.) of individuals in icon images. Inference of sensitive attributes may rely upon stereotypes and generalizations rather than information about specific individuals and are more likely to be incorrect for marginalized people. Incorrect inferences may result in significant physical or psychological injury or restrict, infringe upon or undermine the ability to realize an individual’s human rights. We do not recommend use of OmniParser in any workplace-like use case scenario.

# License
Please note that icon_detect model is under AGPL license, and icon_caption_blip2 & icon_caption_florence is under MIT license. Please refer to the LICENSE file in the folder of each model. 


