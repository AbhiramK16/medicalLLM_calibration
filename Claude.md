# Calibration Collapse in Sequential Clinical Dialogue Agents Turn-Level Confidence vs. Accuracy in Multi-Turn Medical LLM Diagnosis

Question: Does the models confidence track its accuracy as a clinical convo goes on, or stay high while accuracy tanks?

Models: TBD

Dataset: AgentClinic - MedQA


Full Proposal: Calibration Collapse in Sequential Clinical Dialogue Agents

Proposals must be specific and detailed so the direction of the research is clear. If you’re not sure about a given item, please mark it as such and we can discuss.
Add text using your assigned color for all tabs in this doc:
mentor, person1, person2, person3, Kishore
Relevant Past Papers
How is it done today, and what are the limits of current practice?

Schmidgall et al., 2024
https://arxiv.org/abs/2405.07960
Summary:
This paper creates a simulated clinical setting where an LLM acts like a doctor and has to figure out a diagnosis through a back-and-forth conversation with a simulated patient. Instead of just answering one medical question, the model has to ask questions, collect information, and update its thinking over multiple turns. The paper shows that model accuracy drops a lot when the task becomes interactive instead of being a normal static medical question.
Gap/Limitation:
The paper mainly looks at diagnostic accuracy, but it does not check whether the model’s confidence also changes as its accuracy drops. So we do not really know if the model realizes when it is becoming less reliable during the conversation.
Benchmarking Multi-turn Medical Diagnosis: Hold, Lure, and Self-Correction (MINT)
Fang et al., 2026
https://arxiv.org/abs/2604.04325
Summary:
MINT is a multi-turn medical diagnosis benchmark with 1,035 cases. It breaks medical evidence into different turns and tests whether models wait for enough information before making a diagnosis. One major finding is that LLMs often answer too early, with more than 55% committing to a diagnosis within the first two turns.
Gap/Limitation:
MINT shows that models often commit too early, but it does not connect that behavior to confidence. It does not answer whether the model is highly confident when it answers early, or whether its confidence actually matches the amount of evidence it has seen.
On Calibration of Modern Neural Networks
Guo et al., 2017
https://arxiv.org/abs/1706.04599
Summary:
This paper is important because it explains how to measure whether a model’s confidence is trustworthy. It introduces Expected Calibration Error, or ECE, which checks whether a model’s confidence matches its actual accuracy.
Gap/Limitation:
This calibration method was made mainly for single-shot classification tasks, like image classification. It has not really been tested for multi-turn medical conversations where the model’s answer and confidence can change over time.



Motivation	
The problem I want to study is that medical LLM agents may get less accurate as a clinical conversation goes on, but their confidence may not show that drop. This matters because in a medical setting, confidence is a big deal. If an AI gives a diagnosis with high confidence, a doctor or patient may trust it more, even if the model is actually becoming less reliable.
We already know this could be a real issue because AgentClinic shows that accuracy can drop during multi-turn clinical conversations, and MINT shows that models often commit to answers before they have enough evidence. But neither paper really checks whether the model’s confidence changes with its accuracy.
My idea is to measure confidence and accuracy together at every turn of the conversation. Instead of only asking whether the final diagnosis is right, I want to ask: at each point in the conversation, is the model’s confidence actually matching how correct it is?
This should be possible because calibration methods like ECE are already commonly used in machine learning. The new part is applying that idea across turns in a medical dialogue, not just once at the end.


Key Ideas/Contributions/Novelty
The main new idea is measuring calibration over time in a clinical dialogue. AgentClinic looks at accuracy drops. MINT looks at early commitment. Guo et al. gives the calibration method. But no paper combines these ideas to ask whether an LLM knows when its diagnosis is becoming less reliable during a conversation.
The contribution of this project is a turn-by-turn calibration analysis. I want to track whether confidence and accuracy move together or separate as the conversation continues. I also want to compare different types of models, such as reasoning-tuned models and normal instruction-tuned models, to see if one type stays better calibrated than the other.
The main question is simple: when a medical LLM becomes less reliable during a conversation, does its confidence reflect that, or does it keep sounding sure?
Clean question: Does the models confidence track its accuracy as a clinical convo goes on, or stay high while accuracy tanks?
Methods
How does your idea work? Describe the way you will get your results from the initial step. Make a diagram. 
For every turn in an AgentClinic dialogue, the doctor-agent will be asked to give two things:
Its current leading diagnosis
A confidence score from 0–100%
After the full conversation ends, the correct diagnosis will be revealed. Then I will compare the model’s diagnosis at each turn to the ground truth. This lets me check whether the model’s confidence at each turn matched whether it was actually correct.
I will calculate Expected Calibration Error separately for early, middle, and late turns. This will show whether calibration gets worse as the conversation goes on. I will also run the same setup across at least two different LLMs so I can compare model types.
Experimental Setup

How are you going to test your idea to prove that it works?
Think about social science experiments where researchers have a plan of what to make participants do, what to ask them and how to calculate results based on the responses.
What are your baselines for comparison? i.e control group
What models? What datasets? What metrics (if not accuracy)?

What additional analysis do you plan on doing?
How are you going to prove that any improvements in the system are because of your idea alone, i.e. reduce confounding factors and do ablation testing
How does your idea impact the system? What metrics are you quantifying with? 
Visualizations? Statistics?
The main testbed will be AgentClinic-MedQA, using the dialogue-only environment. I will run a fixed set of cases across at least two LLM backbones. Ideally, one will be a reasoning-tuned model and one will be a standard instruction-tuned model.
The baseline will be the same MedQA cases shown in a single-turn format, where the model gets all the information at once. This helps separate normal model calibration from calibration problems caused by the multi-turn conversation itself.
I will also include a control condition where the model gets the same information, but the order of evidence is randomized. This helps test whether calibration drift is caused by conversation length in general, or by the specific way evidence builds up over time.
The main visualizations will be:
ECE by turn group
Accuracy by turn group
Stated confidence vs. actual accuracy
Accuracy-confidence divergence over time

Datasets and Evaluation
Which datasets are you going to use to evaluate your method? Or, are you creating your own?
Reference relevant previous papers here.
If you’re training, what dataset will you use for that?
What is the evaluation metric(s)? 
Some tasks are straightforward to measure (e.g accuracy, for mathematical reasoning). Some are much harder (e.g LLM persuasive ability - think about how you would do this).
AgentClinic GitHub:
https://github.com/SamuelSchmidgall/AgentClinic
This will be the main multi-turn dialogue environment. No training is needed because this is an evaluation-only study.
MedQA GitHub:
https://github.com/jind11/MedQA
MedQA will be used for the static single-turn comparison condition.
Evaluation metrics:
Expected Calibration Error by turn group
Diagnostic accuracy by turn group
Correlation between stated confidence and correctness
Accuracy-confidence gap across the conversation
No model training is required. This is a behavioral evaluation using existing models.


Benchmarks/Evaluation Sets
Evaluation and comparison with existing systems.
What are your baselines? 
Baseline 1:
Single-turn static MedQA cases where the model gets all information at once.
Baseline 2:
Randomized evidence order, where the same information is given but not in the normal clinical sequence.
Main comparison:
Sequential AgentClinic dialogue vs. both baselines across at least two LLM backbones.


Ideal Results
What’s the best case scenario/what do you hope to demonstrate?
What is your hypothesis? What results would prove it to be true?
The best result would be finding that ECE gets worse across turns in the sequential dialogue setting compared to the static baseline. That would show that models are not just becoming less accurate, but also becoming less calibrated.
My hypothesis is that confidence will stay flat or high even when accuracy drops. This would mean the model keeps sounding confident even when it is getting less reliable. If the gap between confidence and accuracy becomes larger in later turns, that would support the main idea.
A strong result would also show that this effect is smaller in the static or randomized-order baselines. That would suggest the issue is specifically related to sequential clinical dialogue.


Potential Limitations
Computation limits, generalization limits, dataset limits, ethical limits?
Computation/API cost:
Running enough cases across multiple models could get expensive. I would probably need to scope the study to around 50–100 cases depending on available credits.
Generalization limit:
AgentClinic-MedQA uses simulated clinical conversations, so the results may not fully apply to real clinical conversations.
Dataset limit:
MedQA cases are exam-style questions, so they may not capture all the uncertainty and messiness of real diagnosis.
Ethical limit:
This study uses simulated benchmark data, not real patients, so privacy risk is low. But the results should not be overstated as direct evidence about real-world hospital deployment.



# Full task set:

1.1: Calibration Collapse in Sequential Clinical Dialogue Agents Turn-Level Confidence vs. Accuracy in Multi-Turn Medical LLM Diagnosis


Kishore: Oversee’s entire project and all tasks. 
Abhiram: Oversee’s entire project and all tasks. 
Aarav: 
Keyi: 




Week 1: Environment Setup & Instrumentation

Kishore: 

Clone the AgentClinic repo, install dependencies, and get the dialogue-only MedQA environment running end-to-end on 5 scratch cases with both model backbones. Confirm the doctor-agent and patient-agent loop terminates cleanly and the full transcript is recoverable.
Set up API keys, per-model cost logging, and a hard spend cap. Record cost-per-case for each backbone so Week 3 can be budgeted from real numbers, not estimates.
Freeze the two model backbones with exact version strings  one reasoning-tuned, one standard instruction-tuned  and pin them in a config file so every later run is reproducible.
Abhiram:
Download and preprocess MedQA. Build the case loader that yields the same case IDs used by AgentClinic so the sequential and static conditions are matched case-for-case.
Build Baseline 1: the static single-turn harness that hands the model the full case vignette at once and asks for one diagnosis + one confidence score.
Build Baseline 2: the randomized-evidence-order generator that shuffles the order in which evidence is revealed while holding total information constant. Verify the shuffle preserves all evidence and only changes sequence. Keyi: 


Aarav:
Write the per-turn elicitation prompt that asks the doctor-agent for its current leading diagnosis and a 0–100 confidence at every turn. Test 3 prompt variants and pick the one with the lowest refusal / malformed-output rate.
Write the structured-output parser and validate it on 20 hand-checked turns. Log parse failures explicitly rather than dropping them silently.
Verify the probe does not change the dialogue itself run 5 cases with and without the confidence probe and compare final diagnoses and turn counts. Keyi: 
Hold the patient-agent fixed across all conditions and backbones Keyi: 

Keyi: 

Lock the scope: number of cases (50–100), which two backbones, how many turns per case, and the early / middle / late turn-bin definition. 
Set up the shared repo, branch conventions, and the canonical turn-level logging schema that all three pipelines must write to.


Week 2: Pilot Runs & Grading Pipeline

Kishore:

Run the pilot: 10 cases × 2 models in the sequential condition. Confirm every turn produces a parseable diagnosis and confidence, and that turn indices align across cases of differing length.
Add retry and failure handling for API timeouts, truncated generations, and dialogues that terminate early. Log every failure with its cause.
Case exclusion rule, written before the runs
Abhiram:
Build the diagnosis grader. Implement three matching methods exact string, synonym/ontology list, and LLM-judge  and measure agreement between them on 100 pilot turns.
Hand-label 50 pilot turns as ground truth and report grader accuracy against those labels. Pick the primary grader and document why; keep the others as robustness checks.
Precision check will 50 vs 100 cases give usable ECE intervals?

Aarav:
Implement Expected Calibration Error with configurable binning, plus reliability diagrams. Unit-test it on synthetic data with known calibration so a bug cannot masquerade as a finding.
Implement the supporting metrics: accuracy per turn bin, mean confidence per turn bin, confidence accuracy gap, and point-biserial correlation between confidence and correctness.
Keyi: 
Read 10 full pilot transcripts by hand. Check for prompt leakage, the patient-agent revealing the answer, degenerate confidence values (all 80s), and refusals.
Make the go / no-go call on the full run and sign off on the final case count given measured cost-per-case.


Week 3: Full Experimental Runs + Stats + Ablations

Kishore:
Run the full sequential AgentClinic condition: all cases × 2 models, with 3 seeds per case if budget allows. Checkpoint after every case so a crash never costs a full run.
Produce the master turn-level dataset and run integrity checks  no missing turns, no duplicate case IDs, no cases present in one condition but absent in another.
Compute bootstrap confidence intervals on ECE for every bin and condition. Run paired tests comparing sequential vs. static and sequential vs. randomized-order on the same cases.
Report the confidence correctness correlation per turn bin and test whether it degrades over the course of the dialogue.
Abhiram:
Run Baseline 1 (static single-turn) and Baseline 2 (randomized evidence order) on the identical case set and identical backbones.
Grade every diagnosis across all three conditions with the primary grader, and re-grade a 10% random sample with the secondary graders as a robustness check.
Run the core ablation: does calibration drift track conversation length or evidence ordering? Compare late-turn ECE in the sequential condition against late-turn ECE in the randomized-order condition at matched turn counts.
Cross-reference early commitment with confidence  for cases where the model locks a diagnosis in the first two turns, does confidence stay high even when that early diagnosis is wrong?
Aarav:
Bin every turn into early / middle / late and compute ECE, accuracy, and mean confidence per bin, per model, per condition.
Produce the first pass of the four core figures: ECE by turn group, accuracy by turn group, stated confidence vs. actual accuracy, and confidence–accuracy divergence over time.
Run the reasoning-tuned vs. instruction-tuned comparison on every metric and check whether the calibration gap between model types widens in later turns.
Run the sensitivity checks: ECE under different bin counts, results under each of the three graders, and results with and without parse-failure turns included.
Keyi: 
Monitor spend against the cap daily and spot-check transcripts from the full run for anything the pilot did not surface.
Confirm case sets are matched across all three conditions before any analysis is trusted.
Decide which findings survive the robustness checks and which are too fragile to claim. Write the one-sentence headline result.
Review every figure for misleading axes, unequal bin sizes, and unlabelled uncertainty.


Paper Writing Tasks Due Each Week

Kishore:
Introduction
Methods
Abhiram:
Abstract
Results
Aarav:
Discussion
Conclusion


Keyi: 

Related Works
Results Figures only


Entire Team:
Finalize all figures with consistent styling and captions that state the sample size and uncertainty for each panel. (Ask mentor Kiran for color palette she will walk you through how to create figures. )  
1 methods pipeline figure (workflow)
4 results figures


Additional Notes:

Our project has 3 conditions, and they all use the same underlying medical cases, therefore only the presentation format changes:
Sequential the AgentClinic condition. The doctor-agent has to ask for information over ~20 turns, and evidence dribbles out in clinical order. This is Kishore's .
Static your Baseline 1. Same case, but the whole vignette is handed over at once and you get one diagnosis + one confidence. This is basically the original MedQA format.
Randomized-order your Baseline 2. Same evidence, revealed over turns, but scrambled out of clinical sequence. So sequential-vs-static is what isolates "does the multi-turn format itself hurt calibration." Sequential-vs-randomized isolates "is it conversation length, or specifically the way clinical evidence accumulates in order."

The case IDS must match because the AgentClinic-MedQA is 215 cases, each with history, symptoms, labs, and a gold-standard diagnosis. They can be found in agentclinic_medqa.jsonl, one JSON object per line, and the code loads them with get_scenario(id) returning self.scenarios[id] so a "case ID" is literally the line index in that file. Each record has an OSCE_Examination block with Patient_Actor, Objective_for_Doctor, Physical_Examination_Findings, Test_Results, and Correct_Diagnosis.

The reason your task says "matched case-for-case": if the sequential run uses cases 0–49 and your static run uses a different 50 MedQA questions, then any ECE difference you find could just be that one set of cases is harder. You'd have no way to tell. Matching means case 37 appears in all three conditions, so you can run paired tests, the same patient, three formats , and the case difficulty cancels out.

Your task says "download and preprocess MedQA," which makes it sound like you should pull the raw MedQA files and match them up to AgentClinic's cases. Don't do it that way. AgentClinic's cases were built by sampling MedQA questions and populating a structured JSON case file, so the mapping back to raw MedQA line numbers isn't guaranteed to be clean, and text-matching 215 rewritten vignettes is a miserable week. GitHub

Instead, make agentclinic_medqa.jsonl the single source of truth:
Load that file once. case_id = line index (or better, a stable hash of Correct_Diagnosis + patient text, so it survives an upstream file update — note the repo went from 107 → 215 cases at one point).

Build your static vignette by concatenating fields from that same record: patient history, physical exam findings, test results — into one prompt block. Same information, one shot.
Build Baseline 2 by taking the same evidence fields and shuffling their reveal order.
Emit a cases.json manifest: case_id, correct_diagnosis, text_hash, and which fields went into the static vignette. Everyone's pipeline logs case_id into Keyi's shared schema.
Then the integrity check is one line: assert the set of case IDs is identical across all three condition logs, and that correct_diagnosis per ID matches. Run that before anyone computes ECE.



