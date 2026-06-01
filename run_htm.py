from pathlib import Path

from tqdm import tqdm
from time import time

import os

from transformers import AutoTokenizer

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import torch

import xutils.dl.pytorch.utils as pyu
import xutils.core.file_utils as fu

from htm.htm_network import HtmNetwork
from htm.text.tokenizer import Tokenizer
from htm.input.iterable_sparse_encoder import IterableSparseEncoder
from htm.output.learning_decoder_trainer import LearningDecoderTrainer
from htm.output.learning_decoder import LearningDecoder
from htm.spatial_pooler import SpatialPooler
from htm.temporal_memory import TemporalMemory

torch.set_printoptions(profile="full")  # Print all elements of tensors

if __name__ == '__main__':
    tokenizer_name = "gpt2"

    tokenizer_path = Path(f"./tokenizer/{tokenizer_name}").resolve()
    if not tokenizer_path.exists():
        print(f"Tokenizer not found, downloading {tokenizer_name}...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.save_pretrained(tokenizer_path)

    start_time = time()
    print("start_time", start_time)
    pyu.set_random_seed(1234)
    # device = pyu.get_device()
    device = "cpu"
    COLUMN_COUNT = 1024
    SPATIAL_POOL_ACTIVE = 32
    CELLS_PER_COLUMN = 32  # 16

    train_data = [
        # "the cute brown cow jumped over the lazy dog"
        """All my troubles seemed so far away Now it looks as though they're here to stay Oh, I believe in yesterday Suddenly I'm not half the man I used to be There's a shadow hangin' over me Oh, yesterday came suddenly Why she had to go, I don't know, she wouldn't say I said something wrong, now I long for yesterday Yesterday Love was such an easy game to play Now I need a place to hide away Oh, I believe in yesterday Why she had to go, I don't know, she wouldn't say I said something wrong, now I long for yesterday Yesterday Love was such an easy game to play Now I need a place to hide away Oh, I believe in yesterday""",
        """Hey Jude, don't make it bad.  Take a sad song and make it better.  Remember to let her into your heart,  Then you can start to make it better.  Hey Jude, don't be afraid.  You were made to go out and get her.  The minute you let her under your skin,  Then you begin to make it better.  And anytime you feel the pain, hey Jude, refrain,  Don't carry the world upon your shoulders.  For well you know that it's a fool who plays it cool  By making his world a little colder.  Hey Jude, don't let me down.  You have found her, now go and get her.  Remember to let her into your heart,  Then you can start to make it better.  So let it out and let it in, hey Jude, begin,  You're waiting for someone to perform with.  And don't you know that it's just you, hey Jude, you'll do,  The movement you need is on your shoulder.  Hey Jude, don't make it bad.  Take a sad song and make it better.  Remember to let her under your skin,  Then you'll begin to make it  Better better better better better better, oh.  Na na na nananana, nannana, hey Jude."""
        """The brain is the part of the body which lets living beings think. It does some bodily functions, such as telling the rest of the body what to do. Almost all animals have a brain: the exceptions are sponges, cnidarians, and lancelets. Plants and fungi do not have brains, although they do react to changes in their environment. The brain is made up of special cells called nerves, which are connected with each other and with other nerves in our body. The brain gets input from sense organs, and changes behavior in response to this information. In humans, the brain also controls our use of language, and is capable of abstract thought. The brain is the main control centre of the whole body. In all animals, the brain is protected in some way. In ourselves, and all vertebrates, it is protected by the bones of the skull. This is generally true but some activity is caused by the spinal cord directly. For example, reflex actions do not involve the brain. In lower animals, a lot is done without their brain being involved. A neuron (American English), neurone (British English), or nerve cell, is an excitable cell that fires electric signals called action potentials across a neural network in the nervous system, mainly in the central nervous system and help to receive and conduct impulses.""",
        # """The brain is an organ that serves as the center of the nervous system in all vertebrate and most invertebrate animals. It consists of nervous tissue and is typically located in the head (cephalization), usually near organs for special senses such as vision, hearing, and olfaction. Being the most specialized organ, it is responsible for receiving information from the sensory nervous system, processing that information (thought, cognition, and intelligence) and the coordination of motor control (muscle activity and endocrine system). While invertebrate brains arise from paired segmental ganglia (each of which is only responsible for the respective body segment) of the ventral nerve cord, vertebrate brains develop axially from the midline dorsal nerve cord as a vesicular enlargement at the rostral end of the neural tube, with centralized control over all body segments. All vertebrate brains can be embryonically divided into three parts: the forebrain (prosencephalon, subdivided into telencephalon and diencephalon), midbrain (mesencephalon) and hindbrain (rhombencephalon, subdivided into metencephalon and myelencephalon). The spinal cord, which directly interacts with somatic functions below the head, can be considered a caudal extension of the myelencephalon enclosed inside the vertebral column. Together, the brain and spinal cord constitute the central nervous system in all vertebrates. In humans, the cerebral cortex contains approximately 14–16 billion neurons,[1] and the estimated number of neurons in the cerebellum is 55–70 billion.[2] Each neuron is connected by synapses to several thousand other neurons, typically communicating with one another via cytoplasmic processes known as dendrites and axons. Axons are usually myelinated and carry trains of rapid micro-electric signal pulses called action potentials to target specific recipient cells in other areas of the brain or distant parts of the body. The prefrontal cortex, which controls executive functions, is particularly well developed in humans. Physiologically, brains exert centralized control over a body's other organs. They act on the rest of the body both by generating patterns of muscle activity and by driving the secretion of chemicals called hormones. This centralized control allows rapid and coordinated responses to changes in the environment. Some basic types of responsiveness such as reflexes can be mediated by the spinal cord or peripheral ganglia, but sophisticated purposeful control of behavior based on complex sensory input requires the information-integrating capabilities of a centralized brain. The operations of individual brain cells are now understood in considerable detail but the way they cooperate in ensembles of millions is yet to be solved.[3] Recent models in modern neuroscience treat the brain as a biological computer, very different in mechanism from a digital computer, but similar in the sense that it acquires information from the surrounding world, stores it, and processes it in a variety of ways. This article compares the properties of brains across the entire range of animal species, with the greatest attention to vertebrates. It deals with the human brain insofar as it shares the properties of other brains. The ways in which the human brain differs from other brains are covered in the human brain article. Several topics that might be covered here are instead covered there because much more can be said about them in a human context. The most important that are covered in the human brain article are brain disease and the effects of brain damage."""
        # """GREEN EGGS AND HAM. I AM SAM. I AM SAM. SAM I AM. THAT SAM-I-AM! THAT SAM-I-AM! I DO NOT LIKE THAT SAM-I-AM! DO WOULD YOU LIKE GREEN EGGS AND HAM? I DO NOT LIKE THEM, SAM-I-AM. I DO NOT LIKE GREEN EGGS AND HAM. WOULD YOU LIKE THEM HERE OR THERE? I WOULD NOT LIKE THEM HERE OR THERE. I WOULD NOT LIKE THEM ANYWHERE. I DO NOT LIKE GREEN EGGS AND HAM. I DO NOT LIKE THEM, SAM-I-AM. WOULD YOU LIKE THEM IN A HOUSE? WOULD YOU LIKE THEN WITH A MOUSE? I DO NOT LIKE THEM IN A HOUSE. I DO NOT LIKE THEM WITH A MOUSE. I DO NOT LIKE THEM HERE OR THERE. I DO NOT LIKE THEM ANYWHERE. I DO NOT LIKE GREEN EGGS AND HAM. I DO NOT LIKE THEM, SAM-I-AM. WOULD YOU EAT THEM IN A BOX? WOULD YOU EAT THEM WITH A FOX? NOT IN A BOX. NOT WITH A FOX. NOT IN A HOUSE. NOT WITH A MOUSE. I WOULD NOT EAT THEM HERE OR THERE. I WOULD NOT EAT THEM ANYWHERE. I WOULD NOT EAT GREEN EGGS AND HAM. I DO NOT LIKE THEM, SAM-I-AM. WOULD YOU? COULD YOU? IN A CAR? EAT THEM! EAT THEM! HERE THEY ARE. I WOULD NOT, COULD NOT, IN A CAR. YOU MAY LIKE THEM. YOU WILL SEE. YOU MAY LIKE THEM IN A TREE! I WOULD NOT, COULD NOT IN A TREE. NOT IN A CAR! YOU LET ME BE. I DO NOT LIKE THEM IN A BOX. I DO NOT LIKE THEM WITH A FOX. I DO NOT LIKE THEM IN A HOUSE. I DO NOT LIKE THEM WITH A MOUSE. I DO NOT LIKE THEM HERE OR THERE. I DO NOT LIKE THEM ANYWHERE. I DO NOT LIKE GREEN EGGS AND HAM. I DO NOT LIKE THEM, SAM-I-AM. A TRAIN! A TRAIN! A TRAIN! A TRAIN! COULD YOU, WOULD YOU ON A TRAIN? NOT ON TRAIN! NOT IN A TREE! NOT IN A CAR! SAM! LET ME BE! I WOULD NOT, COULD NOT, IN A BOX. I WOULD NOT, COULD NOT, WITH A FOX. I WILL NOT EAT THEM IN A HOUSE. I WILL NOT EAT THEM HERE OR THERE. I WILL NOT EAT THEM ANYWHERE. I DO NOT EAT GREEM EGGS AND HAM. I DO NOT LIKE THEM, SAM-I-AM. SAY! IN THE DARK? HERE IN THE DARK! WOULD YOU, COULD YOU, IN THE DARK? I WOULD NOT, COULD NOT, IN THE DARK. WOULD YOU COULD YOU IN THE RAIN? I WOULD NOT, COULD NOT IN THE RAIN. NOT IN THE DARK. NOT ON A TRAIN. NOT IN A CAR. NOT IN A TREE. I DO NOT LIKE THEM, SAM, YOU SEE. NOT IN A HOUSE. NOT IN A BOX. NOT WITH A MOUSE. NOT WITH A FOX. I WILL NOT EAT THEM HERE OR THERE. I DO NOT LIKE THEM ANYWHERE! YOU DO NOT LIKE GREEN EGGS AND HAM? I DO NOT LIKE THEM, SAM-I-AM. COULD YOU, WOULD YOU, WITH A GOAT? I WOULD NOT, COULD NOT WITH A GOAT! WOULD YOU, COULD YOU, ON A BOAT? I COULD NOT, WOULD NOT, ON A BOAT. I WILL NOT, WILL NOT, WITH A GOAT. I WILL NOT EAT THEM IN THE RAIN. NOT IN THE DARK! NOT IN A TREE! NOT IN A CAR! YOU LET ME BE! I DO NOT LIKE THEM IN A BOX. I DO NOT LIKE THEM WITH A FOX. I WILL NOT EAT THEM IN A HOUSE. I DO NOT LIKE THEM WITH A MOUSE. I DO NOT LIKE THEM HERE OR THERE. I DO NOT LIKE THEM ANYWHERE! I DO NOT LIKE GREEN EGGS AND HAM! I DO NOT LIKE THEM, SAM-I-AM. YOU DO NOT LIKE THEM. SO YOU SAY. TRY THEM! TRY THEM! AND YOU MAY. TRY THEM AND YOU MAY, I SAY. sAM! IF YOU LET ME BE, I WILL TRY THEM. YOU WILL SEE. (... and he tries them ...) SAY! I LIKE GREEN EGGS AND HAM! I DO! I LIKE THEM, SAM-I-AM! AND I WOULD EAT THEM IN A BOAT. AND I WOULD EAT THEM WITH A GOAT... AND I WILL EAT THEM, IN THE RAIN. AND IN THE DARK. AND ON A TRAIN. AND IN A CAR. AND IN A TREE. THEY ARE SO GOOD, SO GOOD, YOU SEE! SO I WILL EAT THEM IN A BOX. AND I WILL EAT THEM WITH A FOX. AND I WILL EAT THEM IN A HOUSE. AND I WILL EAT THEM WITH A MOUSE. AND I WILL EAT THEM HERE AND THERE. SAY! I WILL EAT THEM ANYWHERE! I DO SO LIKE GREEN EGGS AND HAM! THANK YOU! THANK YOU, SAM I AM."""
        # "hello world"
    ]

    tokenizer = Tokenizer(str(tokenizer_path.resolve()))
    train_data = [tokenizer.encode(train_data_item) for train_data_item in train_data]
    train_data_unique_items = list({item for sublist in train_data for item in sublist})

    encoder = IterableSparseEncoder(items=train_data_unique_items,  # list(vocab.keys())
                                    output_size=COLUMN_COUNT,
                                    device=device)
    decoder = LearningDecoder(num_cells=COLUMN_COUNT,  # * CELLS_PER_COLUMN,
                              num_inputs=len(train_data_unique_items),
                              device=device)
    decoder_trainer = LearningDecoderTrainer(model=decoder,
                                             device=device,
                                             numeric_encoder=lambda x: train_data_unique_items.index(x),
                                             numeric_decoder=lambda x: train_data_unique_items[x]
                                             )
    spatial_pooler = SpatialPooler(
        column_count=COLUMN_COUNT,
        output_size=SPATIAL_POOL_ACTIVE,
        potential_pct=0.2,
        permanence_inc=0.1,
        permanence_dec=0.1,
        device=device
    )
    temporal_memory = TemporalMemory(
        num_columns=COLUMN_COUNT,
        cells_per_column=CELLS_PER_COLUMN,
        input_size=SPATIAL_POOL_ACTIVE,
        segment_size=16,
        segment_threshold=12,
        permanence_threshold=0.5,
        initial_permanence=0.21,
        permanence_inc=0.1,
        permanence_dec=0.02,
        max_segments=COLUMN_COUNT * CELLS_PER_COLUMN * 16,  # 2 ** 14,
        device=device
    )

    htm_model = HtmNetwork(
        spatial_pooler=spatial_pooler,
        temporal_memory=temporal_memory,
        encoder=encoder,
        decoder_trainer=decoder_trainer,
        device=device
    ).to(device)

    # --- Training loop ---
    prime_spatial_pooler = True
    # prime_spatial_pooler = False
    checkpoint_directory = "checkpoint/"
    fu.ensure_parent_exists(checkpoint_directory)
    if prime_spatial_pooler:
        print("\n--- Priming Spatial Pooler ", "-" * 40)
        sp_output: list[torch.Tensor] = []
        sp_stability: list[float] = []
        spatial_pooler.train()
        for epoch in (pbar := tqdm(range(50), desc="Priming Spatial Pooler")):
            for i, sequence_item in enumerate(train_data_unique_items):
                pbar.set_postfix({"processing char": sequence_item})
                active_columns = htm_model(sequence_item,
                                           run_temporal_memory=False)
                if len(sp_output) > i:
                    sp_stability[i] = (sp_output[i] == active_columns).sum() / len(active_columns)
                    sp_output[i] = active_columns
                else:
                    sp_output.append(active_columns)
                    sp_stability.append(0)

                pbar.set_postfix({"Stability": sum(sp_stability) / len(sp_stability)})
        spatial_pooler.eval()
        print(f"SP Stability: {sum(sp_stability) / len(sp_stability)}")
        pyu.save_state_dict(spatial_pooler, f"{checkpoint_directory}spatial_pooler.pth")
    else:
        pyu.load_state_dict(spatial_pooler, f"{checkpoint_directory}spatial_pooler.pth")

    print("\n--- Training ", "-" * 50)
    wrong = {}
    for epoch in (pbar := tqdm(range(20), desc="Training")):
        pbar.set_postfix({"Epoch": epoch})
        for train_item in train_data:
            htm_model.reset()
            for sequence_item in train_item:
                htm_model(sequence_item)

    print("\n--- Inference ", "-" * 40)
    htm_model.reset()
    num_right = 0
    wrong = {}
    temporal_memory.eval()
    decoder_trainer.eval()
    for train_item in (pbar := tqdm(train_data, desc="Inference")):
        pbar.set_postfix({"Sequence": train_item})
        htm_model.reset()

        for i, sequence_item in enumerate(train_item):
            sequence_item_decoded = tokenizer.decode(sequence_item)
            pbar.set_postfix({"Item": sequence_item_decoded})
            predicted = htm_model(sequence_item)
            predicted_decoded = tokenizer.decode(predicted)

            if i + 1 < len(train_item):
                if predicted == train_item[i + 1]:
                    num_right += 1
                else:
                    if sequence_item not in wrong:
                        wrong[sequence_item_decoded] = {"count": 0, "wrong": set()}

                    next_sequence_item_decoded = tokenizer.decode(train_item[i + 1])
                    wrong[sequence_item_decoded]["count"] += 1
                    wrong[sequence_item_decoded]["wrong"].add(f'{predicted_decoded} != {next_sequence_item_decoded}')

    num_items = sum(map(len, train_data))
    print(f"Inference accuracy: {(num_items - sum(item["count"] for item in wrong.values())) / num_items}",
          wrong)

    print("Inference ", num_right / len(train_data))

    print("\n--- Auto Generation ---")
    htm_model.reset()
    gen_data = train_data[0]
    start_index = 0
    generate_index = 6
    predicted = None
    predicted_list = []
    num_right = 0

    predicted_list.append(gen_data[start_index])
    for i, sequence_item in enumerate(gen_data[start_index:]):
        input_item = sequence_item if i < generate_index else predicted
        predicted = htm_model(input_item)

        predicted_list.append(predicted)
        if i + 1 < len(gen_data):
            if predicted == gen_data[i + 1]:
                num_right += 1
            else:
                print(f"Wrong at {i}: {tokenizer.decode(predicted)} != {tokenizer.decode(gen_data[i + 1])}")

    print(f"Auto generation accuracy: {num_right / len(gen_data)}")
    print("Predicted: ", tokenizer.decode(predicted_list))
    print(f"--- Total time: {(time() - start_time):.2f} seconds ---")
