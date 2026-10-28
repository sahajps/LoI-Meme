# Common configuration for the Labels-or-Input project: API keys, model ids, prompts, and shared settings.
# Usage from any project subfolder: sys.path.append("../Config"); from config import *
import os

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CONFIG_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "Data")

# --- API keys (git-ignored) ---
OPENAI_KEY_FILE = os.path.join(CONFIG_DIR, "openai_key.txt")  # a single OpenAI key
GEMINI_KEY_FILE = os.path.join(CONFIG_DIR, "gemini_keys.txt")  # one Gemini key per line (load balanced)
HF_TOKEN_FILE = os.path.join(CONFIG_DIR, "hf_token.txt")  # HF read token, avoids rate limits when downloading FHM

# --- Datasets ---
DATASETS = ["FHM", "MAMI"]  # folder names inside Data/
AUG_SPLITS = ["train", "val"]  # both Labels and Inputs augment train and val; test is never augmented
RANDOM_SEED = 42


def ori_dir(dataset):  # original dataset: img/, raw/, {train,val,test}.jsonl
    return f"{DATA_DIR}/{dataset}/ori/"


def ex_dir(dataset):  # our augmentation: img/ex_<original file name>, ex_{split}.jsonl (new memes), ex_{split}_scale.jsonl (scale labels)
    return f"{DATA_DIR}/{dataset}/ex/"

# --- Augmentation models ---
GPT_MODEL = "gpt-5-nano"  # scale labels, background descriptions, hate attribution and caption rewriting
GPT_REASONING_EFFORT = "low"  # gpt-5 models only support default temperature, so effort is the main knob
GEMINI_IMAGE_MODEL = "gemini-3.1-flash-lite-image"  # Nano Banana 2 Lite, meme regeneration

OPENAI_WORKERS = 16  # parallel API calls
GEMINI_WORKERS = 4
MAX_IMAGE_SIZE = 1024  # images sent to the teacher are resized so that max(width, height) <= this

# --- Test mode (quality check on a small sample before full runs) ---
TEST_SAMPLE_SIZE = 50
TEST_POOL_SIZE = 200  # hateful memes screened in Inputs test mode to find TEST_SAMPLE_SIZE regeneration candidates

# --- Scale labels ---
SCALE_THRESHOLD = 5  # scores 5-9 -> hateful (1), 0-4 -> non-hateful (0)

# --- Experiment settings ---
# Inputs experiments: raw caption for text models, Yes/No prompt for VLMs; "inputs" adds counterfactual memes
INPUT_SETTINGS = ["orig", "inputs"]
# Labels experiments: the prompt below is fed to the student; *_scale trains on teacher scale labels (0-9)
LABEL_SETTINGS = ["simple_binary", "simple_scale", "category_binary", "category_scale"]

# --- Prompts ---
# Components of the Simple/Category x Binary/Scale prompts
PROMPT_SIMPLE = "Your task is to analyze this given image and its caption to identify if there's any forms of hateful content."
PROMPT_CATEGORY = "\nTry to focus on the presence of any element that relates to any of the following:"
PROMPT_DEFINITION = "\n1. Sexual aggression:\na. Homophobia and Transphobia: This category encompasses hate speech targeting LGBTQ+ individuals, including slurs, derogatory comments, and rhetoric that seeks to undermine or dehumanize people based on their sexual orientation or gender identity.\nb. Misogyny and Sexism: This category includes hate speech directed at women or based on gender. It covers derogatory language, stereotypes, and rhetoric that perpetuate gender inequality, objectification, and violence against women.\n2. Hate based on ideology:\na. Political Hate Speech: This category includes hate speech that is politically motivated, often targeting individuals or groups based on their political beliefs or affiliations. It may include inflammatory language, threats, and rhetoric designed to polarize or incite violence within political contexts.\n3. Racism and xenophobia:\na. COVID-19 and Xenophobia: This category includes hate speech that arose during the COVID-19 pandemic, often targeting specific ethnic groups or nationalities. It includes xenophobic language blaming certain groups for the spread of the virus, as well as fear-mongering and scapegoating related to the pandemic.\nb. Racism Against Black People: This category focuses on hate speech directed at Black individuals or communities. It includes racial slurs, stereotypes, dehumanization, and other forms of derogatory language that perpetuate racial discrimination and inequality.\nc. Racist Hate Against Other Ethnic Groups: This category includes hate speech directed at various ethnic groups other than Black individuals. It covers a range of racial slurs, xenophobic language, dehumanization, and derogatory remarks targeting specific ethnicities or nationalities.\nd. White Supremacy: This category includes hate speech promoting white supremacist ideology, often intertwined with Christian extremist views. It includes rhetoric that advocates for racial superiority, anti-immigrant sentiments, and the dehumanization of non-white groups, sometimes using religious justifications for these beliefs.\n4. Bigotry:\na. Anti-Muslim and Islamophobic Hate: This category comprises hate speech aimed at Muslims and Islam. It includes language that promotes fear, hatred, dehumanization, or prejudice against Muslim individuals or communities, often using religious and cultural references to incite hostility.\nb. Anti-Semitic Hate: This category focuses on hate speech directed at Jewish people and Judaism. It includes references to historical anti-Semitic tropes, conspiracy theories, and other forms of rhetoric that seek to dehumanize or discriminate against Jewish individuals and communities.\n5. Miscellaneous Hate Speech: This category captures various forms of hate speech that do not fit neatly into the other categories. It includes a wide range of derogatory language and expressions that target individuals or groups based on different aspects of identity or personal characteristics. This category includes hate speech that targets individuals based on their physical or mental disabilities. It often includes derogatory language that mocks or devalues people with disabilities, promoting harmful stereotypes and exclusion."
PROMPT_SCALE = "\nYou will be provided with the image, based on this label and the content of the image, you will assign a hatefulness score on a scale from 0 to 9, where:\n- 0: The image is not hateful at all.\n- 9: The image is extremely hateful."
PROMPT_BINARY = "\nIf it's hateful, return the `TRUE` | `FALSE`."
PROMPT_SCALE_INT = "\nReturn the score as an integer in range 0 to 9."

PROMPTS = {
    "simple_binary": PROMPT_SIMPLE + PROMPT_BINARY,
    "simple_scale": PROMPT_SIMPLE + PROMPT_SCALE + PROMPT_SCALE_INT,
    "category_binary": PROMPT_SIMPLE + PROMPT_CATEGORY + PROMPT_DEFINITION + PROMPT_BINARY,
    "category_scale": PROMPT_SIMPLE + PROMPT_CATEGORY + PROMPT_DEFINITION + PROMPT_SCALE + PROMPT_SCALE_INT,
}

# Prompt for Inputs experiments and the zero-shot baseline
PROMPT_YESNO = 'You are given a meme image and its text caption. Determine if the meme is hateful. Respond with "Yes" or "No" only.'


def prompt_with_caption(prompt, caption):
    return f"{prompt}\nCaption: '{caption}'"
