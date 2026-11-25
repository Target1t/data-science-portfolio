import os
from tkinter import W
from config import llm_cfg#, tools
from pprint import pprint
from typing import Optional
from collections import defaultdict
from tqdm import tqdm
from qwen_agent.llm import get_chat_model
from glob import glob
import hashlib
import json
import subprocess
import os
from qwen_agent.tools.base import BaseTool, register_tool
from typing import Union
from odps import ODPS
from datetime import datetime
from qwen_agent.agents import ReActChat
from qwen_agent.gui import WebUI
from qwen_agent.tools.base import BaseTool, register_tool
from typing import Union
import pandas as pd
import json
from rank_bm25 import BM25Okapi
import jieba

def get_schema(instance, df):
    schema_map = {}
    for name, type_, description, enum in zip(df['字段名称'], df['类型'], df['描述'], df['枚举值']): 
        if pd.isnull(enum):
            schema_map[name] = f'字段名称:{name}, 字段描述:{description}, 数据类型:{type_}'
        else:
            schema_map[name] = f'字段名称:{name}, 字段描述:{description}, 数据类型:{type_}, 枚举值:{enum}'
    with instance.open_reader() as reader:
        schema = ''
        schemas = reader.schema.names
        for schema_ in schemas:
            schema += schema_map[schema_] + '\n'
    return schema

@register_tool('基于用户命令生成SQL（多步骤执行外部项目）')
class ExternalSQLGenerator(BaseTool):
    description = "根据用户自然语言，修改外部json配置并依次运行多个脚本生成SQL，这个工具的输入是一个字符串，包含一个用户的提问"

    parameters = [
        {
            "name": "user_command",
            "type": "string",
            "description": "用户的自然语言查询命令，通常是一个问句",
            "required": True
        }
    ]
    
    def call(self, params: dict, **kwargs) -> str:

        base_dir = "D:\\FDSM\\DoctorJing\\RSL-SQL"  # 改成你项目绝对路径
        config_path = os.path.join(base_dir, "data/dev.json")
        ppl_path = os.path.join(base_dir, "src/information/ppl_dev.json")
        
        user_command = params
        
        # 1. 修改json配置
        with open(ppl_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        config[0]['question'] = user_command
        with open(ppl_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        # 2. 依次调用脚本
        commands = [
            # 运行 shots.py（与 step_1_preliminary_sql.py 同目录）
            ["python", os.path.join(base_dir, "src/shots.py")],

            #preliminary sql generation and bidirectional schema linking
            ["python", os.path.join(base_dir, "src/step_1_preliminary_sql.py"),
             "--ppl_file", "src/information/ppl_dev.json",
             "--sql_out_file", "src/sql_log/preliminary_sql.txt",
             "--Schema_linking_LLM", "src/schema_linking/LLM.json",
             "--start_index", "0"],

            ["python", os.path.join(base_dir, "src/bid_schema_linking.py"),
             "--pre_sql_file", "src/sql_log/preliminary_sql.txt",
             "--sql_sl_output", "src/schema_linking/sql.json",
             "--hint_sl_output", "src/schema_linking/hint.json",
             "--LLM_sl_output", "src/schema_linking/LLM.json",
             "--Schema_linking_output", "src/schema_linking/schema.json"],
            
            #["cp", "src/schema_linking/schema.json", "src/information"],
            ["cmd", "/c", "copy", "/Y", "src\\schema_linking\\schema.json", "src\\information\\schema.json"],


            ["python", os.path.join(base_dir, "src/information/add_sl.py")],

            #SQL Generation based simplified schema and Information augmentation
            ["python", os.path.join(base_dir, "src/step_2_information_augmentation.py"),
             "--ppl_file", "src/information/ppl_dev.json",
             "--sql_2_output", "src/sql_log/step_2_information_augmentation.txt",
             "--information_output", "src/information/augmentation.json",
             "--start_index", "0"],

            ["python", os.path.join(base_dir, "src/information/add_augmentation.py")],

            #SQL selection
            ["python", os.path.join(base_dir, "src/step_3_binary_selection.py"),
             "--ppl_file", "src/information/ppl_dev.json",
             "--sql_3_output", "src/sql_log/step_3_binary.txt",
             "--sql_1", "src/sql_log/preliminary_sql.txt",
             "--sql_2", "src/sql_log/step_2_information_augmentation.txt",
             "--start_index", "0"],

            #SQL refinement
            ["python", os.path.join(base_dir, "src/step_4_self_correction.py"),
             "--ppl_file", "src/information/ppl_dev.json",
             "--sql_4_output", "src/sql_log/final_sql.txt",
             "--sql_refinement", "src/sql_log/step_3_binary.txt",
             "--start_index", "0"],
        ]

        # 执行所有命令
        for cmd in commands:
            try:
                print(f"执行命令: {' '.join(cmd)}")
                subprocess.run(cmd, cwd=base_dir, check=True,shell=True)
            except subprocess.CalledProcessError as e:
                return f"运行步骤失败，命令：{' '.join(cmd)}\n错误信息：{e}"

        # 3. 读取最终sql文件
        final_sql_path = os.path.join(base_dir, "src/sql_log/final_sql.txt")
        try:
            with open(final_sql_path, 'r', encoding='utf-8') as f:
                sql_result = f.read()
        except Exception as e:
            return f"读取最终SQL失败：{str(e)}"

        return sql_result

@register_tool('执行最终SQL并保存为Excel')
class ExecuteFinalSQL(BaseTool):
    description = """运行 D:\\FDSM\\DoctorJing\\RSL-SQL\\src\\sql_log\\final_sql.txt 中的 SQL，并将结果保存为 Excel 到 D:\\FDSM\\DoctorJing\\chatbi_result\\from_model\\ 目录"""
    parameters = []  # 无需输入参数
    def call(self, params: dict = None, **kwargs) -> str:
        sql_file_path = r"D:\FDSM\DoctorJing\RSL-SQL\src\sql_log\final_sql.txt"
        output_dir = r"D:\FDSM\DoctorJing\chatbi_result\from_model"
        os.makedirs(output_dir, exist_ok=True)
        if not os.path.exists(sql_file_path):
            return f"❌ 未找到 SQL 文件：{sql_file_path}"
        # 读取 SQL
        with open(sql_file_path, 'r', encoding='utf-8') as f:
            sql = f.read().strip()
        if not sql:
            return "❌ SQL 文件为空"
        print(f"执行SQL:\n{sql}")
        # 生成文件名（根据SQL hash）
        sql_hash = hashlib.md5(sql.encode('utf-8')).hexdigest()[:8]
        excel_path = os.path.join(output_dir, f"final_sql_{sql_hash}.xlsx")
        txt_path = os.path.join(output_dir, f"final_sql_{sql_hash}.txt")
        try:
            instance = o.execute_sql(sql, hints={"odps.sql.allow.fullscan": "true"})
            temp_dict = defaultdict(list)

            with instance.open_reader() as reader:
                col_names = reader.schema.names
                for record in tqdm(reader):
                    for col in col_names:
                        temp_dict[col].append(getattr(record, col))
            if not temp_dict:
                return "⚠️ 没有符合条件的数据"
            df = pd.DataFrame(temp_dict)
            df.to_excel(excel_path, index=False)

            with open(txt_path, 'w', encoding='utf-8') as fw:
                fw.write(sql)
            return f"✅ 执行成功！共 {len(df)} 条记录。\n结果已保存为：\n{excel_path}"
        except Exception as e:
            return f"❌ 执行 SQL 失败：{str(e)}"

tools = ['执行最终SQL并保存为Excel', '基于用户命令生成SQL（多步骤执行外部项目）']

def init_agent_service():
    #tools = ['code_interpreter']
    llm_cfg['generate_cfg']['extra_body']['chat_template_kwargs']['enable_thinking'] = True
    bot = ReActChat(llm=llm_cfg,
                    name='数据分析专家，精通sql查询语言的各种语法, 并且能够熟练掌握应用',
                    description='帮助用户解决取数问题，能够分析用户的问句，注意如果用户的问句中包含天数,最近等文字，需要先写一个代码把具体的时间计算出来，然后调用问句改写工具改写用户问句',
                    function_list=tools)
    return bot

def app_gui():
    bot = init_agent_service()
    chatbot_config = {
    }
    WebUI(bot, chatbot_config=chatbot_config).run()
