"""AgentClinic case loading and normalization.

This adapter will convert AgentClinic records into the project's common case
representation. Dataset-specific logic belongs here, not in experiment runners.
"""

import hashlib
import json
from pathlib import Path

path = Path(__file__).resolve().parents[3] / "data" / "agentclinic_medqa_extended.jsonl"

class medQA:
    data: list
    hashed: dict
    idhash: dict

    def __init__(self, ):
        with open(path, "r") as age_medQA:
            hashed = {} 
            idhash = {}
            data = []

            for i in age_medQA:
                case = json.loads(i)
                data.append(case)
                hashed.update({ self.findHash(case): case})
                idhash.update({len(data)-1 : self.findHash(case)})
                
            self.hashed = hashed
            self.data = data
            self.idhash = idhash

    def getID(self, id):
       return self.hashed[self.idhash[id]] if id < len(self.data) else None


    def getHash(self, hash):
        return self.hashed[hash] if hash in self.idhash.values() else None

    @staticmethod
    def findHash(case):
        diagnosis = case["OSCE_Examination"]["Correct_Diagnosis"]
        patient_actor = json.dumps(case["OSCE_Examination"]["Patient_Actor"], sort_keys=True)
    
        return hashlib.sha256((diagnosis + patient_actor).encode()).hexdigest()[:16]

    @staticmethod
    def extract_evidence(case):
        """Flatten the three clinical evidence sections into atomic pieces.

        Each piece is one labeled finding (history, symptom, exam, or test
        result). This is the unit of evidence revealed per turn in the
        randomized-order condition.
        """
        exam = case["OSCE_Examination"]
        items = []
        for section in ("Patient_Actor", "Physical_Examination_Findings", "Test_Results"):
            medQA._walk(exam[section], section, items)
        return items

    @staticmethod
    def _walk(node, path, items):
        if isinstance(node, dict):
            for key, value in node.items():
                medQA._walk(value, f"{path} > {key}", items)
        elif isinstance(node, list):
            items.append(f"{path}: " + "; ".join(str(item) for item in node))
        else:
            items.append(f"{path}: {node}")

    @staticmethod
    def vignette(data, indent=0):
        pad = "  " * indent
        lines = []
        if type(data) is dict:
            for k, v in data.items():
                label = k.replace("_", " ")
                if label != "Correct Diagnosis":
                    if isinstance(v, (dict, list)):
                            lines.append(f"{pad}{label}: \n")
                            lines.append(medQA.vignette(v, indent + 1))
                    else:
                            lines.append(f"{pad}{label}: {v} \n")
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    lines.append(medQA.vignette(item, indent))
                else:
                    lines.append(f"{pad}- {item} \n")
        else:
            lines.append(f"{pad}{data}")

        return "".join(lines)



if __name__ == "__main__":
    sample = medQA()
    print(sample.vignette(sample.data).split("OSCE Examination")[-1])