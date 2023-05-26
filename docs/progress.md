## 2022/10/16-2022/10/23
### srcIndex < srcSelectDimsize
- The input/output may be too long (PRIMERA: `input: 4096`, `output: 1024`).
- Before fixing the spaces problem, the input and output size only work with less then 512 tokens.
- The define special token `<KEEP>`, `<ADD>`, `<SUB>`.
- Too many spaces in contents (replace '.\n\n' with '.\c\c')
- Replace '.\\c\\c' with '.\n\n' for later edit actions calculation.

### Dataset Construction
- Key: 'document', Value: 'summary'
- document:  Maximum 10 paragraphs in a list
```
[' <KEEP> The 2021 Taliban offensive is an ongoing military offensive by the Taliban and allied militant groups, including al-Qaeda, against the government of Afghanistan and its allies that began on 1 May 2021, simultaneous with the withdrawal of most U.S. troops from Afghanistan. <KEEP> As of 15 July, over a third of Afghanistans 421 districts were controlled by the Taliban, and by 21 July, half of Afghanistan was under Taliban control <KEEP>',
 'During the Afghan Civil War (1996–2001), resistance to the Taliban was strongest in northern Afghanistan, the base of the Northern Alliance. <ADD> According to the Afghanistan Analysts Network, the Talibans concentration of its forces in the north may be an attempt to forestall the creation of a second Northern Alliance after the withdrawal of U.S. forces <KEEP>',
 'In May, the Taliban captured 15 districts from the Afghan government, including Nirkh and Jalrez districts in Maidan Wardak Province. <KEEP> Among the locations captured was the Dahla Dam in Kandahar Province, Afghanistans second largest dam. <KEEP> During the month, 405 Afghan National Security Forces and 260 civilians were killed during the clashes with the Taliban, while the Afghan Ministry of Defense claimed killing 2,146 Taliban fighters <KEEP>',
 'By the end of May, Portugal, Slovenia, Spain, and Sweden had completely withdrawn their forces from Afghanistan <KEEP>',
 'In June, the Taliban captured 69 districts from the Afghan government and entered the cities of Kunduz and Puli Khumri. <KEEP> The city of Mazar-i-Sharif was besieged by Taliban. <KEEP> Among the locations captured by Taliban was Afghanistans main border crossing with Tajikistan and the Saydabad District in Maidan Wardak Province, which is called the gateway of Afghanistans capital city Kabul. <KEEP> In terms of equipment the Taliban captured 700 trucks and Humvees from the Afghan security forces as well as dozens of armored vehicles and artillery systems <KEEP>',
 'An Afghan Air Force Mil Mi-17 was shot down by the Taliban, killing three pilots, while a UH-60 Black Hawk was damaged on ground after an outpost belonging to the Afghan Armed Forces was shelled by the Taliban in the same month. <KEEP> On 16 June, Taliban militants executed 22 surrendering Afghan Army commandoes in the town of Dawlat Abad. <KEEP> During the month, 703 Afghan National Security Forces and 208 civilians were killed during the clashes with the Taliban, while the Afghan Ministry of Defense claimed killing 1,535 Taliban fighters. <KEEP> On 19 June, Afghan National Army chief of staff, defense and interior ministers were replaced by President Ashraf Ghani. <KEEP> By the end of June, all Resolute Support Missions member countries had withdrawn their troops, except for the Britain, Turkey, and the U.S <KEEP>',
 'On 22 June, the Taliban captured Shir Khan Bandar, Afghanistans main Tajikistan border crossing. <KEEP> 13 districts fell to the Taliban within 24 hours. <KEEP> On the same day, heavy fighting was also occurring in Baghlan Province after Afghan forces launched a military operation on the outskirts of Pul-e-Khumri, the provincial capital, killing 17 Taliban militants including Qari Khalid, a Taliban divisional commander. <KEEP> Simultaneously, Taliban forces took control of Balkh and encircled Mazar-i-Sharif, the capital of Balkh Province. <KEEP> On 23 June, the Taliban and Afghan forces clashed inside Pul-e Khumri <KEEP>',
 'On 25 June, the Taliban took control of the Shinwari District and the Ghorband District in Parwan province north of Kabul. <ADD> That same day NBC News reported that the Taliban "were surprised at the speed of their advance and had avoided capturing some targets so as not to run afoul of the U.S.", and the Afghan government launched a program called National Mobilization that aimed to arm militia groups to fight the Taliban. <KEEP> Meanwhile, Taliban deputy emir Sirajuddin Haqqani issued a series of instructions on Voice of Jihad for the governance of territories seized in the offensive. <ADD> FDDs Long War Journal researcher Thomas Joscelyn argued that Haqqanis statements "read like those that would be issued by the head of a nation" <KEEP>',
 'On 27 June, Chaki Wardak District and Saydabad District fell to the Taliban after at least 50 Afghan troops surrendered and were captured by the Taliban. <KEEP> On the same day Rustaq District, Shortepa District and the Arghistan District fell to the Taliban. <KEEP> ToloNews reported that 108 districts fell to the Taliban in the last two months and the Afghan army had only managed to re-take 10. <KEEP> On 29 June, the Taliban launched an offensive on Ghazni, causing violent clashes within the city <KEEP>',
 'In July, the Taliban captured 64 districts from the Afghan government and entered the second and third largest cities of Afghanistan, Kandahar and Herat respectively. <KEEP> During the month, 335 Afghan National Security Forces and 189 civilians were killed during the clashes with the Taliban, while the Afghan Ministry of Defense claimed killing 3,159 Taliban fighters. <KEEP> Around 1,500 Afghan soldiers deserted into Tajikistan, according to its CSTO envoy. <KEEP> Iranian media reported that around 300 Afghan soldiers and civilians had crossed the border and entered into Iran to escape the Taliban <KEEP>']
```
- summary: `str` format 

### Hardware problem (ECC error)
- Create new instance.

### CUDA Error
- Downgrade the CUDA version to meet Pytorch's requirements.

### Add special token to tokenizer/vocab
- Add the additional special token utilizing Huggingface's built-in function.
- Add our special token to tokenizer, then update the `vocab.json`. Then use the updated tokenizer to do truncation and encoding.

### Applied method and Next step
- Replace '.\n\n' with '.\c\c' to fix the srcIndex problem.
- Special tokens are not added successfully, the built-in function may not work. *Sol: Try to utilize the updated tokenizer in the main model*
- The #outupts is more then given #instances, still needed to debug. 

### Methods of adding special tokens to tokenizer
- Huggingface's built-in function

```Python
self.docsep_token_id = self.tokenizer.convert_tokens_to_ids("<doc-sep>")
self.tokenizer.add_special_tokens({'additional_special_tokens': ["<KEEP>", "<ADD>", "<SUB>"]})
self.model.resize_token_embeddings(len(self.tokenizer))
self.keep_token_id = self.tokenizer.convert_tokens_to_ids("<KEEP>")
self.add_token_id = self.tokenizer.convert_tokens_to_ids("<ADD>")
self.sub_token_id = self.tokenizer.convert_tokens_to_ids("<SUB>")

attention_mask[input_ids == self.docsep_token_id] = 2
attention_mask[input_ids == self.keep_token_id] = 2
attention_mask[input_ids == self.add_token_id] = 2
attention_mask[input_ids == self.sub_token_id] = 2
```
- Update existed tokenizer, and `vocab.json`

```Python
tokenizer = AutoTokenizer.from_pretrained('../PRIMER_wcep')
model = LongformerEncoderDecoderForConditionalGeneration.from_pretrained('../PRIMER_wcep')
tokenizer.add_tokens(['<KEEP>', '<ADD>', '<SUB>'])
model.resize_token_embeddings(len(tokenizer))
model.save_pretrained('../PRIMER_wcep/new')
tokenizer.save_pretrained('../PRIMER_wcep/new')
```

## 2022/11/23-2022/11/30
### Find the indices of paragraphs which needed to be updated (reconstruct new dataset)
- Calculate the edit actions for each paragraphs (the `top-3` or `top-5` paragraphs).
- Construct new dataset to fine-tune PRIMERA. (modify the sliding window is optional)

## Solution to special token
- By construncting a new dataset, the sepcial tokens are no longer to be encoded.

### Length Problem
- If we extract the `top-3`/`top-5` paragraphs, the lengths are short enough for model compared with the full-content scale.

### Goal
- The original goal is unchanged, but we divide the task into 3 steps:
1. Apply the labeling algorithm on sentence-level annotation.
2. Re-construnct our datset (consists of the paragraphs which more needed to be updated).
3. Merge the output with our original dataset, then prove our method is capable to update an full-article given its old version and the triggered news event.

### TODO
- Extract the `top-3` most-edited paragraphs from training, testing, and dev set. (Extract from `labeled_fixed`, and the last paragraph of train/test/val pt file to construct )
#### May meet the alignment problem between the position from old and new paragraphs. **Make sure to check if the `#papagraphs` is same between versions.**
- Combine the `top-3` paragraphs from `content` and the triggered news as our new input (`content` of our wcep-format dataset), the `summary` would be the updated information.
- Since we extract the `top-3` paragraphs, the amount of our extended dataset will be 3-times larger.
- Create the baseline after the extraction. (So, we will have new baseline)
- Apply `PRIMERA-WCEP` to fine-tune our new-constructed dataset.
- Examine the result with the baseline.

### Discussions
- Since we labeled our data given the non-updated and updated version, `top-3` or `top-5` are easy to extract.
- **#Paragraphs in the updated version is more than non-updated version after solving the alignment problems, make those instances hard to align the indices. Calculate how many instances with this problem existed in our training data. We may need to truncate the updated version?**
- **Among the `1602` instances in our training data, there are `468` instances having more than 10 paragraphs differ from non-updated and updated contents. (29.21%)**
- Mapping method: `Updated version` -> `Non-updated version`. We have `[KEEP]`, `[SUB]`, `[ADD]` conditions. In our data, we have the `updated version with labels + news trigger` as src, and `updated version without labels` as tgt.
- Extract the triggers from the last paragraph of NetKu dataset(full-content) to form our extended dataset.
- Our fine-tuned model may only performs well on our NetKu dataset.
- Maybe a method to find the paragraphs which needed to be updated is still important without the updated content.


## 2022/11/30-2022/12/07
### Change the main idea to summary generation
- Given an old summary and the news event trigger, the model will be capable of how to modify the src (add labels to sentence-level).
- `src`: The un-labeled non-updated summary + news trigger event knowledge. `tgt`: **Labeled** updated summary.
- Labeling Direction: `Updated version` -> `Non-updated version`, the direction is unchanged.
- The `src` may have to merge with bi-directional diffs.
- Obtain the model generation, do the sentence editing according to the predicted labels.
- Calculate the #instances those have the same non-updated, and updated summary.

### TODO
- Re-labeled the `summary`, drop the `top-k` discussions. (DONE)
- Calculate the #paragarphs between non-updated and updated version of summary. (Dropped)
- We may have to drop the instances with high-similarity and low-similarity between non-updated and updated summaries.
- Check if the last paragraph of summary is same as the last paragraph in train/test/val pt file. (DONE)
- Re-construct the dataset with summary revisions, the `util/` from `edit_data` may be still work.
- Truncation is no longer needed.
- Check the baseline after the new dataset is constructed.


## 2022/12/07-2022/12/14
### Discussions
- Bi-directional sentence labeling is done.
- High similarity between non-updated and updated summaries.
- If update part of the full-content, we may have to find a rule to check which paragraphs are needed to be updated.
- Although the raio of `SequenceMatcher` is high, still have to check the `ROUGE-L` in our summaries (without trigger).

### TODO
- The clean data is updated. (DONE)
- The `ROUGE`, `Bert-Score` calculation on train set without trigger existed is done. (summary-level)
- Re-build the pt file for train set without trigger (we already have the `src` and `tgt` in local) (Done)
- Upload the local `src` and `tgt` to VM. Then, construct the upated pt file and do sentence-labeling. (Done)
- After labeling, calculate the sum of #RM, #ADD, #SUB. (Calculate the averaged edit-actions) (Done)
- If problems occur, we may have to have more reseaerch on finding the paragraphs those needed to be udpated. (Running)
- Re-label the train set (bs3) after truncating the input length into `4096`. `util/edit_data.py`.
- Post text-cleaning after labeling (truncation, calculate the #edits). 


## 2022/12/14-2022/12/21
### Discussions
- Since the sequence similarity and ROUGE between non-updated and updated summaries is high, we may have to focus on updating partial article.
- Papers: `EditEvel`, `Attention Temperature Matters in Abstractive Summarization Distillation`, `Leveraging Locality in Abstractive Text Summarization`
- Construct the relationship between paragraphs.
- Find the core paragraphs when trigger occurs (Tree, ROUGE)?

### TODO
- Check the #paragraph splitted with '\n\n' and '\\c\\c' (Done)
- Check the details of data collection.
- Truncate the train set (bs3) after labeling. `util/edit_data.py`. (Done)
- Post text-cleaning after labeling (truncation, calculate the #edits). (Done)
- Find "paragraphs relation related" papers.(Long-term)
- Survey the BartGraphSum paper and the the citation papers. (Long-term)
- Find more reference papers from ACL2022, EMNLP2022, NAACL2022. (Long-term)
- Fix the labeling scripts, and re-labeling. (Running)
- Prepare the code for pre-training(Done)
- Pretrain: BART encoder + softmax (Inputs: trigger + section name + summary), (Outputs: 0/1, means edit or not )
- Edit Actions (Pretrain): Paragraph-level, calculate the `#[ADD]+#[SUB]+#[RM]`.
- Generation: Given the trigger representaiton and the old section contents, output the generated summaries.
- Extract the triggers, find the section name, and define the edition threshold for each section.
- If we don't have the section name, may have to generate the names with GPT.

## 2022/12/21-2022/12/28
- Do not filter the duplicated sentences. (Done)
- Re-extract the docs and summaries with the raw data. (Running)
    - Extract the section names of each instance (Done)
    - Extract the texts exclude `summary` from instances, for section-pairing. (Running)

## 2022/12/28-2023/01/04
- Align section names to each paragraph (Running)
    - Remove summaries from each full-contnent
    - Align the rest paragraphs to sections
        - Extract section names except "See also", "References" (Done) 
        - Extract and count the numbers of paragraphs under each section (Done)
        - Align the "Section name" + "Number of paragraphs" (Done)
    - Add summaries back to full-content with section name `summary` (Running-3)
- The counts of paragraphs from each section is based on counting only "Contents", no summaries involved. But, total #paragraphs from sections equals to total #paragraphs of contents with summary included. index0 (Running-2)
- If the contents include <li>, the "non_update_all_paragraph" counts, but our program doesn"t count. (Done)
- In some instances, "See also." may not exist. We have to find another method to define the end_idx. (Done) 
- Concate the main section name and its sub-section. (Running)
- Define the column names of csv. (Running)
- Run unit tests, and fix the big block. (Done)
- Concate the main section name with sub-section name. (Done)
- The #paragraphs between non-updated and updated are not same, it's hard to align the section/sub-section with paragraphs.
    - New to Old or Old-to-New, but bidirectional may be optimal.
- Re-labeling the sentences.

## 2023/01/04-2023/01/11
- We meet the problems on hard to align the labeled data to respective section/subsection.
    - Construct the hash map to store the #paragraphs under each section/subsection.
    - Merge the section/subsection from non-updated and updated version.
    - The hash map store the larger #paragraphs of each section between non-updated and updated articles.
- Merge the sections:
    - Some sections are no longer existed in new version. (Not confirmed yet)
    - Some sections are added in new version.(Not confirmed yet)
    - #instances with different sections between non-updated and updated articles is 1361/1602 in train set, 174/201 in test set.
    - We may apply section name from updated articles, and remove the paragraphs with [RM] after labeling.
- Sentence labeling: remain bidirectional labeling.
- Then, running our later steps: Paragraphs labeling -> train BART encoder.

## 2023/01/11-2023/01/18
### See the instances with same sections
- Train set: We have `1334/1602` instances have same sections (names), `911` instances with same section names and #paragraphs.
    - Train set: Extract the `1334` instances, do sentence-labeling and paragraph-labeling. (Running)
    - Train set: Prepare tmp baselines. (Running)
    - Train set: Extract triggers from `1334` instances. (Done)
    - Train set: Extract the updated version of section-#paragraphs pairs (csv format).
        - Problems: Updated #paragraphs cannot capture both added and removed paragraphs.
        - Solution: Each paragraph should be labeled with corresponded section name at tail.
        - The solution may fix the different sections problems? (Thinking)
- Test set: We have `168/201` instances have same sections (names), `168` instances with same sections and #paragraphs.
    - Test set: Extract triggers from `168` instances. (Done)
- Val set: We have `152/192` instances have same sections (names), `152` instances with same sections and #paragraphs.
    - Val set: Extract triggers from `152` instances. (Done)

## 2023/01/18-2023/02/01
- Sentence re-labeling. (check if the #instances is the same) (Running)
    - In train set, 1 of the instances is abandoned. The local id is `290`. (Done)
    - In test set and val set, #instances remain the same.
- Check if the #paragarphs from data equals to our extraction.
    - The first instance return true, check all instances. (Done)
    - Create the titles as keys to pair with each paragraph. (Done)
 - `same_secs_old`: Text with un-matched #paragraphs.
- `same_secs_new`: Text with prunning according to #paragraphs under sections.
- `same_secs_insert`: Text with section name inserted.
- Re-calculating the baseline. (merge the test and val set?) (Running)
- Data Preparation (Done)
- Data Labeling (Running)

## 2023/02/01-2023/02/08
- `same_secs_insert_labeled`: Text with section name inserted, and sentences are labeled. `csv` formats include "paragraphs without [ADD]", triggers and targets.
- `merged_updated*`: Merge the paragraphs and trigger in the source, then construct csv file.
- Take test, val as different experiments, and calculate the baselines.
- Data Labeling (Running) (already removed [ADD] sentences)
- We need another algorithm to match paragraphs into its respective instance, record the num. (Done)
- Train the encoder. (Running)
- Re-calculate the baseline for test, val set. (Running)

## 2023/02/08-2023/02/15
- The classification result does not meet our expectation.
    - Use the tuned model to predict on train set (in order to check the training process) (1:2)
    - Resample the train set to (1:1). 

## 2023/02/15-2023/02/22
- `BART` is fine-tuned for our task, and the articles are updated.
- File names:
- `encoder.py`: the encoder script.
- `classification.py`: classification process with `encoder.py` trained model.
- `submission.csv`: the classification result.
- `eval_classification.py`: extract the classification report.
- `classification_report.csv`: the classification report.
- `generate.py`: generate the summaries with trained model.
- `generated_paragraphs.csv`: generated summaries.
- `pos_classification.csv`: extract the id with positive label.
- `updated_file.csv`: merge the updated and non-updated paragraphs.
- `final_merged.csv`: re-construct the instances. (`hyp` and `ref`)
- `final_hyp.src`: the splitted hyp file for calculate rouge score.
- `final_ref.tgt`: the splitted ref file for calculate rouge score.
- `decoder.py`: the decoder script.

### Other files
- `merged_numpar_train/test/val.csv`: the numbers of paragraphs from each instance.
- `merged_train/test/val.csv`: concate the paragraph and its section.
- `merged_updated_train/test/val.csv`: concate the paragraph, section, and its trigger.
- `samesecs_triggers_train/test/val.txt`: the triggers of instances.

### TODO 
1. The ROUGE is strange, so the baseline is re-calculating...if the baseline is fine, re-construct the `final_ref.tgt` (Done)
2. ROUGE is fixed already, calculate with other metrics.
3. More Experiments with ChatGPT.
4. Decoder has to be fine-tuned with `same_secs_insert/train_text.txt.src` and `same_secs_insert/train_text.txt.tgt`

## 2023/02/22-2023/03/01
- Zero-shot and fine-tune with PRIMERA
- PRIMERA as decoder (fine-tuned)

## 2023/03/01-2023/03/08
- Expriments with ChatGPT api.

## 2023/03/08-2023/03/15
- Experiments on PRIMERA 
- few-shots learning with ChatGPT api. (choose the highest, medium, lowest instances from two-staged BART)

## 2023/03/22-2023/03/29
- To run few-shots learning, we have to calculate the length of prompts first, then pick the one with the most editions. (only 70 instances fit this condition)
- We already fed the prompt (idx=91) to ChatGPT(3.5) api, and re-calculate the input length if we only give the non-updated andthe triggred news.
- Feed the 87 instances to api, then model has to generate the proper responses. (FSL in MAX Mode)
- We have to extract the references of the 87 instances according to the indices, then contruct the `hyps` and `refs` for evaluation. (mark as MAX Mode, and this is NOT DONE yet.) (DONE)
- `chatgpt_max_2`, `chatgpt_mean`, `chatgpt_min`: FSL with BART outputs (from our experiments).
- `chatgpt_max_3`, `chatgpt_mean_2`, `chatgpt_min_2`: FSL with ground truths.

## 2023/03/29-2023/04/05
- Running few-shots with GPT-4. (few-shots and prompts in single message) (few-shots with 3 or 5 examples)
- Check the `87` indices is same on one-shot GPT-3.5.
- Re-evaluate the `87` instances in our experiments.

## 2023/04/05-2023/04/12
- The GPT-4 is current in problem, cause timeout in our experiments, have to wait.
- Sampling the `87` instances from our experiments, and re-evaluate. (DONE)
- Write down more details of the differences between `NetKu` and the dataset we currently applied- In order to feed FSL, we have to exclude the fixed prompts, and arrange the max input length for each shot (total 5 shots). -> max input length for each shot is `1600`
- Means the length of old content + trigger + updated content has to be smaller than `1600`.
- To get the shots, we concate the old content, trigger, and updated content from `87` instances, than get the indices which has lengths < `1600`. -> Turns out that there is `43` instances fit our conditions.
- We have to access these `43` instances, than get the *max*, *mean*, and *min* instances to feed with FSL.

## 2023/04/12-2023/04/19
- One-shot learning: experiment with `GPT-3.5` and `GPT-4`.
- Prompts for one-shot learning: The first prompt has to tell GPT to update according with the triggered news, so that would fit our goal. Another prompt to test: Prompt has to tell GPT to update contents according to important person/time/places...
- The duration time of our experiments has to be noted.
- Lastly, do few-shot learning: sorting order has to be rearrange randomly 3 times, then calculate mean scores.

## 2023/05/18-2023/05/25
- Prepare the prompt files for experiments (decoder, inference) (Done) (./for_decoder_exp)
- Use Vicuna-13B as decoder to produce paragraphs. (ing)
    - Merge updated and non-updated paragraphs to article.
- Fine-tune Vicuna-13B in paragraph-level?
- Prepare the lengths statistics for decoder inputs, both the ground truths and decoder outputs (paragraph-level). (Done)
- Clean the code for non-updated, updated merging.
- More try on GPT-4. (trying)
- Use the GPT as decoder. (trying)
- If conduct fine-tune on LLaMA based models, LoRA or Adapter would be in consideration.
- Vicuna + LoRA? (as decoder) (zero-shot)
    - prepare json file (paragraph-level) for LoRA?
- Compare the code between `finetune.py` and `finetune_lora.py` from lit-llama repo
- Prepare the finetune code for Vicuna+LoRA.
    - Prepare the script to convert our data to meet the `./sample/lora_chinese_repo.jsonl`
    - Install packages according to the requirements during finetuning Vicuna with LoRA approach (./requirements_finetune_vicuna.txt)
- Prepare the merging code (for merging the non-upated and updated paragraphs into full articles)
- After completing Vicuna+LoRA, we have to merge Vicuna with our fit (LoRA)? [Ref](https://github.com/Facico/Chinese-Vicuna/tree/master/tools)
