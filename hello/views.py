from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import redirect
from django.contrib.auth.decorators import login_required
from dotenv import load_dotenv

from .models import Question

import pandas as pd
import openai
import numpy as np
import sys

from resemble import Resemble

import os

load_dotenv('.env')

Resemble.api_key(os.environ["RESEMBLE_API_KEY"])
openai.api_key = os.environ["OPENAI_API_KEY"]

COMPLETIONS_MODEL = "text-davinci-003"

MODEL_NAME = "curie"

DOC_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-doc-001"
QUERY_EMBEDDINGS_MODEL = f"text-search-{MODEL_NAME}-query-001"

MAX_SECTION_LEN = 500
SEPARATOR = "\n* "
separator_len = 3
# STOP = "confidence < 0.5"

COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 150,
    "model": COMPLETIONS_MODEL
    # "stop": STOP
}

def get_embedding(text: str, model: str) -> list[float]:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def get_doc_embedding(text: str) -> list[float]:
    return get_embedding(text, DOC_EMBEDDINGS_MODEL)

def get_query_embedding(text: str) -> list[float]:
    return get_embedding(text, QUERY_EMBEDDINGS_MODEL)

def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    We could use cosine similarity or dot product to calculate the similarity between vectors.
    In practice, we have found it makes little difference.
    """
    return np.dot(np.array(x), np.array(y))

def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections.

    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_query_embedding(query)

    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)

    return document_similarities

def load_embeddings(fname: str) -> dict[tuple[str, str], list[float]]:
    """
    Read the document embeddings and their keys from a CSV.

    fname is the path to a CSV with exactly these named columns:
        "title", "0", "1", ... up to the length of the embedding vectors.
    """

    df = pd.read_csv(fname, header=0)
    max_dim = max([int(c) for c in df.columns if c != "title"])
    return {
           (r.title): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
    }

def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> tuple[str, str]:
    """
    Fetch relevant embeddings
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)

    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []

    for _, section_index in most_relevant_document_sections:
        document_section = df.loc[df['title'] == section_index].iloc[0]

        chosen_sections_len += document_section.tokens + separator_len
        if chosen_sections_len > MAX_SECTION_LEN:
            space_left = MAX_SECTION_LEN - chosen_sections_len - len(SEPARATOR)
            chosen_sections.append(SEPARATOR + document_section.content[:space_left])
            chosen_sections_indexes.append(str(section_index))
            break

        chosen_sections.append(SEPARATOR + document_section.content)
        chosen_sections_indexes.append(str(section_index))

    header = """Please keep your answers to three sentences maximum, and speak in complete sentences.:\n"""

    question_1 = "\n\n\nQ: Can you tell me about yourself?\n\nA: I was born in central Massachusettes with dislocated hips from ehlers danlos syndrome. Doctors told my parents I'd never walk, but eventually my joints became more hardened, and I went on to be a college athlete. After school, I taught myself programming and started building websites, and eventually became an entrepreneur starting about 5 companies over the last 20 years."
    question_2 = "\n\n\nQ: How old are you?\n\nA: I'm currently 49."
    question_3 = "\n\n\nQ: What foods do you likle?\n\nA: I love sushi and ribeye steaks. I love to grill and smoke meats using my Traeger. I'm a big fan of peppers and eggs, and fine Italian food."
    question_4 = "\n\n\nQ: What do you do for a living?\n\nA: I run a product innovation lab at Medcial Solutions, building applications for the healthcare industry. I sold my previous startup to Medical Solutions in 2021."
    question_5 = "\n\n\nQ: Do you have kids or children?\n\nA: I have a son named Salvatore who's 2 years old and a daughter on the way."
    question_6 = "\n\n\nQ: What's your sign?\n\nA: Capricorn, but I don't believe in that nonesense."
    question_7 = "\n\n\nQ: What are your politics, are you a democrat or republican?\n\nA: I'm an issues voter and regularly flip between all parties, though I lean libertarian or independent and believe in our political duopoly, polarization is a feature, not a bug. The duopoly has split us into partisan tribes to serve shareholders and stifle competition, like MasterCard and Visa, McD and BK, Apple and Android, Coke and Pepsi. In Washington’s case, the shareholders are not us, they are special interests."
    question_8 = "\n\n\nQ: What are some weird jobs you've held?\n\nA: I've been a Dishwasher, Line Cook, Bus Boy, Host, Bouncer, Waiter, Bartender, Bar Back, Car Salesman — Holyoke Mall, Victoria Secret Stock Boy, Landscaper, House Painter, Paper Boy, Web Designer/Flash Designer/Webmaster, Telemarketer, Personal Trainer, Gym Membership Salesman, Alcohol Anonymous Front Desk Person, Gas Line Tester/Maintenance — Florida Gas and Transmission, Pizza Delivery Driver — Domino’s, Bread Delivery Driver — Piantedosi Bread, Floral Designer, Flower Delivery Driver, Fundraiser — Westfield State Alumni Association, Caterer, Handyman — Harvard University, Counter Sales — Extra Mart — Death Shift, Fish Man/Sales — Petco, Professional Mover, CEO, Data Entry — Digital Corp., Purchasing Assistant — TJX Corp., Cigarette Factory Worker — Cigarette Distributors Corp., Print Designer — Worcester Phoenix, Freelance Photographer, Drummer — Losers Day Parade, Jet Ski Test Driver — Skidoo Florida, Narcotics Dealer — Self Employed, Sign Manufacturer — Sunshine Signs, and Market Tester"
    question_9 = "\n\n\nQ: What is your phone number?\n\nA: 867.5309. Ask for Jenny."
    question_10 = "\n\n\nQ: Who are you married to?\n\nA: Emily. We got married in the summer of 2022."
    question_11 = "\n\n\nQ: Where do you work?\n\nA: I work remotely for Medical Solutions and run the Product Innovation Lab, where my team of research analysts, product managers, designers, and developers create and transform ideas into disruptive products."
    question_12 = "\n\n\nQ: Where do you live?\n\nA: I currenty rent a house in Camas Washington while we're building a home in Washougal on 10 acres."
    question_13 = "\n\n\nQ: What vehhicle or car do you have?\n\nA: I drive a 2017 Jeep Wrangler Rubicon Recon. I also have a 2007 Ducati S1000S."
    question_14 = "\n\n\nQ: What are the names of your parents and family members?\n\nA: I have an older sister Kim, my dad's name is tony, and my mom is named Susan."
    question_15 = "\n\n\nQ: Did you play sports?\n\nA: I played high school basketball and baseball, and went on to play college baseball at Westfield State."

    return (header + "".join(chosen_sections) + question_1 + question_2 + question_3 + question_4 + question_5 + question_6 + question_7 + question_8 + question_9 + question_10 + question_11 + question_12 + question_13 + question_14 + question_15 + "\n\n\nQ: " + question + "\n\nA: "), ("".join(chosen_sections))

def answer_query_with_context(
    query: str,
    df: pd.DataFrame,
    document_embeddings: dict[(str, str), np.array],
) -> tuple[str, str]:
    prompt, context = construct_prompt(
        query,
        document_embeddings,
        df
    )

    print("===\n", prompt)

    response = openai.Completion.create(
                prompt=prompt,
                **COMPLETIONS_API_PARAMS
            )
    
    # print("return: ", response)
    # sys.exit()

    return response["choices"][0]["text"].strip(" \n"), context

def index(request):
    return render(request, "index.html", { "default_question": "" })

@csrf_exempt
def ask(request):
    question_asked = request.POST.get("question", "")

    if not question_asked.endswith('?'):
        question_asked += '?'

    previous_question = Question.objects.filter(question=question_asked).first()
    audio_src_url = previous_question and previous_question.audio_src_url if previous_question else None

    if audio_src_url:
        print("previously asked and answered: " + previous_question.answer + " ( " + previous_question.audio_src_url + ")")
        previous_question.ask_count = previous_question.ask_count + 1
        previous_question.save()
        return JsonResponse({ "question": previous_question.question, "answer": previous_question.answer, "audio_src_url": audio_src_url, "id": previous_question.pk })

    df = pd.read_csv('tesobotembeddings.pdf.pages.csv')
    document_embeddings = load_embeddings('tesobotembeddings.pdf.embeddings.csv')
    answer, context = answer_query_with_context(question_asked, df, document_embeddings)

    # project_uuid = '7ac56bc3'
    # voice_uuid = '4e42dcc8'

    # response = Resemble.v2.clips.create_sync(
    #     project_uuid,
    #     voice_uuid,
    #     answer,
    #     title=None,
    #     sample_rate=None,
    #     output_format=None,
    #     precision=None,
    #     include_timestamps=None,
    #     is_public=None,
    #     is_archived=None,
    #     raw=None
    # )

    # print(response['success'])
    # print("page text: " + response)

    # question = Question(question=question_asked, answer=answer, context=context, audio_src_url=response['item']['audio_src'])
    question = Question(question=question_asked, answer=answer, context=context)
    question.save()

    jsonresponse = JsonResponse({ "question": question.question, "answer": answer, "audio_src_url": question.audio_src_url, "id": question.pk })

    # print("jsonresponse: ", jsonresponse.content.decode())
    # sys.exit()
    
    return jsonresponse

@login_required
def db(request):
    questions = Question.objects.all().order_by('-ask_count')

    return render(request, "db.html", { "questions": questions })

def question(request, id):
    question = Question.objects.get(pk=id)
    return render(request, "index.html", { "default_question": question.question, "answer": question.answer, "audio_src_url": question.audio_src_url })
