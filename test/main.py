import sys
sys.path.append("Module_2\\Chap_9")
from src.utils import *


df = create_dataframe("Module_2\\Chap_9\\data\\Dataset")
create_index(df)

query_image_path = "Module_2\\Chap_9\\data\\Dataset\\Avatar_Thuan_Duong.jpg"
display_query_and_top_matches(query_image_path, df)