OV_OUT_DIR = '/home/aistudio/openvino_notebooks/notebooks/paddleocr_vl/ov_paddleocr_vl_model'

from notebook_utils import device_widget

device = device_widget("CPU")
from gradio_helper import make_demo


from pathlib import Path



import openvino as ov
from ov_paddleocr_vl import OVPaddleOCRVLForCausalLM

# Parameters
ov_model_path = str(OV_OUT_DIR)
task = "ocr"
max_new_tokens = 512

llm_infer_list = []
vision_infer = []
core = ov.Core()

paddleocr_vl_model = OVPaddleOCRVLForCausalLM(
    core=core,
    ov_model_path=ov_model_path,
    device=device.value,
    llm_int4_compress=False,
    llm_int8_compress=True,
    vision_int8_quant=False,
    llm_int8_quant=True,
    llm_infer_list=llm_infer_list,
    vision_infer=vision_infer,
)


demo = make_demo(paddleocr_vl_model)

try:
    # demo.launch()
    demo.launch(debug=True, height=900,share=True)
    # demo.launch(debug=True, height=900)
    # demo.launch(debug=True)
except Exception:
    # demo.launch(debug=True,height=900)
    demo.launch(debug=True, share=True, height=900)
# if you are launching remotely, specify server_name and server_port
# demo.launch(server_name='your server name', server_port='server port in int')
# Read more in the docs: https://gradio.app/docs/