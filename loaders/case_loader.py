import hashlib
import json
from pathlib import Path

current_dir = Path(__file__).resolve().parent

full_dir = current_dir.parent / "data" / "raw" / "agentclinic_medqa_extended.jsonl"

class medQA_scenario:
    def __init__(self, path=None):
        self.path = path
        if self.path is None:
            self.path = full_dir
        with open(self.path, "r") as age_medQA:
            data_hash = {} 
            data_id = {}
            id_hash = {}
            data = []

            for i in age_medQA:
                case = json.loads(i)
                data.append(case)
                data_hash.update({ self.unique_hash(case): case})
                data_id.update({len(data)-1 : case})
                id_hash.update({len(data)-1 : self.unique_hash(case)})
                
            self.data_hash = data_hash
            self.data_id = data_id
            self.data = data
            self.id_hash = id_hash

    def get_case_id(self, id):
        try:
            val = self.data_id[id]
        except KeyError:
            raise KeyError("Key not found")

        return val


    def get_case_hash(self, hash):
        try:
            val = self.data_hash[hash]
        except KeyError:
            raise KeyError("Key not found")

        return val

    @staticmethod
    def unique_hash(case):
        diagnosis = case["OSCE_Examination"]["Correct_Diagnosis"]
        patient_actor = json.dumps(case["OSCE_Examination"]["Patient_Actor"], sort_keys=True)
    
        return hashlib.sha256((diagnosis + patient_actor).encode()).hexdigest()[:16]

    def static_vignette(data, indent=0):
        pad = "  " * indent
        lines = []
        if type(data) is dict:
            for k, v in data.items():
                label = k.replace("_", " ")
                if label != "Correct Diagnosis":
                    if isinstance(v, (dict, list)):
                            lines.append(f"{pad}{label}: \n")
                            lines.append(medQA_scenario.static_vignette(v, indent + 1))
                    else:
                            lines.append(f"{pad}{label}: {v} \n")
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    lines.append(medQA_scenario.static_vignette(item, indent))
                else:
                    lines.append(f"{pad}- {item} \n")
        else:
            lines.append(f"{pad}{data}")

        return "".join(lines)
'''
    def static_vignette(self):
        prompts = []
        for i in self.data:
            comp = i["OSCE_Examination"]
            prompts.append(
                "\n\n".join([
                    self._format_section({"Objective": comp["Objective_for_Doctor"]}),
                    self._format_section({"Patient": comp["Patient_Actor"]}),
                    self._format_section({"Physical Exam": comp["Physical_Examination_Findings"]}),
                    self._format_section({"Test Results": comp["Test_Results"]}),
                ])
            )
        return "\n".join(prompts)
'''





'''
        output = {}
        def flatten(data, start=""):
            if type(data) is dict:
                for key in data:
                    flatten(data[key], start+key+"_")

            elif type(data) is list:
                for index, value in enumerate(data):
                    flatten(value, start+str(index)+"_")

            else:
                output[start[:-1]] = data

            return output

        fin_output = []
        for i in info:
            fin_output.append(flatten(i))
        return fin_output
'''



medQA = medQA_scenario()
print(medQA_scenario.static_vignette(medQA.data).split("OSCE Examination:")[-1])
   