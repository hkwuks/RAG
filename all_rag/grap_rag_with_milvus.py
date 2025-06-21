import os
import numpy as np
from collections import defaultdict
from scipy.sparse import csc_matrix
from pymilvus import MilvusClient
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from tqdm import tqdm
from transformers import BertTokenizer, BertForTokenClassification
from transformers import pipeline

os.environ['OPENAI_API_KEY'] = 'OPENAI_API_KEY'

milvus_client = MilvusClient(
    uri='YOUR_ENDPOINT',
    token='YOUR_TOKEN'
)

llm = ChatOpenAI(
    model='gpt-4o',
    temperature=0
)

embedding_model = OpenAIEmbeddings(model='text_embedding-3-small')

nano_dataset = [
    {
        "passage": "Jakob Bernoulli (1654–1705): Jakob was one of the earliest members of the Bernoulli family to gain prominence in mathematics. He made significant contributions to calculus, particularly in the development of the theory of probability. He is known for the Bernoulli numbers and the Bernoulli theorem, a precursor to the law of large numbers. He was the older brother of Johann Bernoulli, another influential mathematician, and the two had a complex relationship that involved both collaboration and rivalry.",
        "triplets": [
            ["Jakob Bernoulli", "made significant contributions to", "calculus"],
            [
                "Jakob Bernoulli",
                "made significant contributions to",
                "the theory of probability",
            ],
            ["Jakob Bernoulli", "is known for", "the Bernoulli numbers"],
            ["Jakob Bernoulli", "is known for", "the Bernoulli theorem"],
            ["The Bernoulli theorem", "is a precursor to", "the law of large numbers"],
            ["Jakob Bernoulli", "was the older brother of", "Johann Bernoulli"],
        ],
    },
    {
        "passage": "Johann Bernoulli (1667–1748): Johann, Jakob’s younger brother, was also a major figure in the development of calculus. He worked on infinitesimal calculus and was instrumental in spreading the ideas of Leibniz across Europe. Johann also contributed to the calculus of variations and was known for his work on the brachistochrone problem, which is the curve of fastest descent between two points.",
        "triplets": [
            [
                "Johann Bernoulli",
                "was a major figure of",
                "the development of calculus",
            ],
            ["Johann Bernoulli", "was", "Jakob's younger brother"],
            ["Johann Bernoulli", "worked on", "infinitesimal calculus"],
            ["Johann Bernoulli", "was instrumental in spreading", "Leibniz's ideas"],
            ["Johann Bernoulli", "contributed to", "the calculus of variations"],
            ["Johann Bernoulli", "was known for", "the brachistochrone problem"],
        ],
    },
    {
        "passage": "Daniel Bernoulli (1700–1782): The son of Johann Bernoulli, Daniel made major contributions to fluid dynamics, probability, and statistics. He is most famous for Bernoulli’s principle, which describes the behavior of fluid flow and is fundamental to the understanding of aerodynamics.",
        "triplets": [
            ["Daniel Bernoulli", "was the son of", "Johann Bernoulli"],
            ["Daniel Bernoulli", "made major contributions to", "fluid dynamics"],
            ["Daniel Bernoulli", "made major contributions to", "probability"],
            ["Daniel Bernoulli", "made major contributions to", "statistics"],
            ["Daniel Bernoulli", "is most famous for", "Bernoulli’s principle"],
            [
                "Bernoulli’s principle",
                "is fundamental to",
                "the understanding of aerodynamics",
            ],
        ],
    },
    {
        "passage": "Leonhard Euler (1707–1783) was one of the greatest mathematicians of all time, and his relationship with the Bernoulli family was significant. Euler was born in Basel and was a student of Johann Bernoulli, who recognized his exceptional talent and mentored him in mathematics. Johann Bernoulli’s influence on Euler was profound, and Euler later expanded upon many of the ideas and methods he learned from the Bernoullis.",
        "triplets": [
            [
                "Leonhard Euler",
                "had a significant relationship with",
                "the Bernoulli family",
            ],
            ["leonhard Euler", "was born in", "Basel"],
            ["Leonhard Euler", "was a student of", "Johann Bernoulli"],
            ["Johann Bernoulli's influence", "was profound on", "Euler"],
        ],
    },
]

entityid_2_relationids = defaultdict(list)
relationid_2_passageids = defaultdict(list)

entities = []
relations = []
passages = []
for passage_id, dataset_info in enumerate(nano_dataset):
    passage, triplets = dataset_info["passage"], dataset_info["triplets"]
    passages.append(passage)
    for triplet in triplets:
        if triplet[0] not in entities:
            entities.append(triplet[0])
        if triplet[2] not in entities:
            entities.append(triplet[2])
        relation = " ".join(triplet)
        if relation not in relations:
            relations.append(relation)
            entityid_2_relationids[entities.index(triplet[0])].append(
                len(relations) - 1
            )
            entityid_2_relationids[entities.index(triplet[2])].append(
                len(relations) - 1
            )
        relationid_2_passageids[relations.index(relation)].append(passage_id)

embedding_dim = len(embedding_model.embed_query('foo'))


def create_milvus_collection(collection_name: str):
    '''
    Create a new Milvus collection with specified configuration.
    Args:
        collection_name: The name of the collection to create.
    '''
    if milvus_client.has_collection(collection_name=collection_name):
        milvus_client.drop_collection(collection_name=collection_name)
    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=embedding_dim,
        consistency_level='Strong'
    )


entity_col_name = "entity_collection"
relation_col_name = "relation_collection"
passage_col_name = "passage_collection"
create_milvus_collection(entity_col_name)
create_milvus_collection(relation_col_name)
create_milvus_collection(passage_col_name)


def milvus_insert(collection_name: str, text_lst: list[str]):
    '''
    Insert text data with embeddings into a Milvus collection in batches.

    This function processes a list of text strings, generates embeddings for them, and inserts the data into the
    specified Milvus collection in batches for efficient processing.
    Args:
        collection_name: The name of the Milvus collection to insert data into.
        text_lst: A list of text strings to be embedded and inserted.
    '''
    batch_size = 512
    for row_id in tqdm(range(0, len(text_lst), batch_size), desc="Inserting"):
        batch_texts = text_lst[row_id: row_id + batch_size]
        batch_embeddings = embedding_model.embed_documents(batch_texts)

        batch_ids = [row_id + j for j in range(len(batch_texts))]
        batch_data = [
            {'id': id_,
             'text': text,
             'vector': vector}
            for id_, text, vector in zip(batch_ids, batch_texts, batch_embeddings)
        ]
        milvus_client.insert(collection_name=collection_name, data=batch_data)


milvus_insert(collection_name=relation_col_name, text_lst=relations)
milvus_insert(collection_name=entity_col_name, text_lst=entities)
milvus_insert(collection_name=passage_col_name, text_lst=passages)

query = "What contribution did the son of Euler's teacher make?"

tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
model = BertForTokenClassification.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')

nlp_ner = pipeline('ner', model=model, tokenizer=tokenizer)
query_ner_list = [item['word'] for item in nlp_ner(query)]

query_ner_embeddings = [embedding_model.embed_query(query_ner) for query_ner in query_ner_list]

top_k = 3

entity_search_res = milvus_client.search(collection_name=entity_col_name, data=query_ner_embeddings, limit=top_k,
                                         output_fields=['id'])

query_embedding = embedding_model.embed_query(query)

relation_search_res = \
    milvus_client.search(collection_name=relation_col_name, data=[query_embedding], limit=top_k, output_fields=['id'])[
        0]

entity_relation_adj = np.zeros((len(entities), len(relations)))
for entity_id, entity in enumerate(entities):
    entity_relation_adj[entity_id, entityid_2_relationids[entity_id]] = 1

entity_relation_adj = csc_matrix(entity_relation_adj)

entity_adj_1_degree = entity_relation_adj @ entity_relation_adj.T
relation_adj_1_degree = entity_relation_adj.T @ entity_relation_adj

target_degree = 1

entity_adj_target_degree = entity_adj_1_degree
for _ in range(target_degree):
    entity_adj_target_degree @= entity_adj_1_degree

relation_adj_target_degree = relation_adj_1_degree
for _ in range(target_degree):
    relation_adj_target_degree @= relation_adj_1_degree

entity_relation_adj_target_degree = entity_adj_target_degree @ relation_adj_target_degree.T

expanded_relations_from_relation = {}
expanded_relations_from_entity = {}

filtered_hit_relation_ids = [relation_res['entity']['id'] for relation_res in relation_search_res]
for hit_relation_id in filtered_hit_relation_ids:
    expanded_relations_from_relation.update(relation_adj_target_degree[hit_relation_id].nonzero()[1].tolist())

filtered_hit_entity_ids = [
    one_entity_res["entity"]["id"]
    for one_entity_search_res in entity_search_res
    for one_entity_res in one_entity_search_res
]

for filtered_hit_entity_id in filtered_hit_entity_ids:
    expanded_relations_from_entity.update(
        entity_relation_adj_target_degree[filtered_hit_entity_id].nonzero()[1].tolist())

relation_candidate_ids = list(
    expanded_relations_from_relation | expanded_relations_from_entity
)

relation_candidate_texts = [
    relations[relation_id] for relation_id in relation_candidate_ids
]

query_prompt_one_shot_input = '''
I will provide you with a list of relationship descriptions. Your task is to select 3 relationships that may be useful to answer the given question. Please return a JSON object containing your thought process and a list of the selected relationships in order of their relevance.

Question:
When was the mother of the leader of the Third Crusade born?

Relationship descriptions:
[1] Eleanor was born in 1122.
[2] Eleanor married King Louis VII of France.
[3] Eleanor was the Duchess of Aquitaine.
[4] Eleanor participated in the Second Crusade.
[5] Eleanor had eight children.
[6] Eleanor was married to Henry II of England.
[7] Eleanor was the mother of Richard the Lionheart.
[8] Richard the Lionheart was the King of England.
[9] Henry II was the father of Richard the Lionheart.
[10] Henry II was the King of England.
[11] Richard the Lionheart led the Third Crusade.
'''

query_prompt_one_shot_output = '''{"thought_process": "To answer the question about the birth of the mother of the leader of the Third Crusade, I first need to identify who led the Third Crusade and then determine who his mother was. After identifying his mother, I can look for the relationship that mentions her birth.", "useful_relationships": ["[11] Richard the Lionheart led the Third Crusade", "[7] Eleanor was the mother of Richard the Lionheart", "[1] Eleanor was born in 1122"]}'''

query_prompt_template = '''Question:
{question}

Relationship descriptions:
{relation_des_str}

'''


def rerank_relations(query: str, relation_candidate_texts: list[str], relation_candidate_ids: list[str]) -> list[int]:
    '''
    Rerank candidate relations using LLM to select the most relevant ones for answering a query.

    This function uses a LLM with Chain-of-Thought pormpting to analyze candidate relationships and select the most
    useful ones for answering the query. It employs a one-shot learning approach with a predefined example to guide the
    LLM's reasoning process.

    Args:
        query: The input question that needs to be answered.
        relation_candidate_texts: List of candidate relationship descriptions.
        relation_candidate_ids: List of IDs corresponding to the candidate relations.

    Returns:
        A List of relation IDs ranked by their relevance to the query.
    '''

    relation_des_str = '\n'.join(
        map(lambda item: f'[{item[0]}] {item[1]}', zip(relation_candidate_ids, relation_candidate_texts))).strip()

    rerank_prompts = ChatPromptTemplate.from_messages(
        [
            HumanMessage(query_prompt_one_shot_input),
            AIMessage(query_prompt_one_shot_output),
            HumanMessagePromptTemplate.from_messages(query_prompt_template)
        ]
    )

    rerank_chain = (rerank_prompts | llm.bind(response_format={'type': 'json_object'}) | JsonOutputParser())

    rerank_res = rerank_chain.invoke({'question': query, 'relation_des_str': relation_des_str})

    rerank_relation_ids = []
    rerank_relation_lines = rerank_res['useful_relationships']
    id2lines = {}
    for line in rerank_relation_lines:
        id_ = int(line[line.find("[") + 1: line.find("]")])
        id2lines[id_] = line.strip()
        rerank_relation_ids.append(id_)
    return rerank_relation_ids
