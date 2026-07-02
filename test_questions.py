eval_dataset = [
    {
        "question": "What is the main contribution of this paper?",
        "ground_truth": "The paper proposes a language-based digital twin framework that models personalized conversational behavior in older adults using large language models. It also introduces a cVAE-based evaluator to measure both response fidelity and cognitive consistency for monitoring cognitive health. :contentReference[oaicite:0]{index=0}"
    },
    {
        "question": "What method or approach is proposed?",
        "ground_truth": "The proposed approach fine-tunes GPT-4.1-mini to generate personalized conversational responses using participant metadata and stylometric features such as pause and tempo. A conditional variational autoencoder (cVAE) is then used to evaluate reconstruction quality and predict cognitive (MoCA) scores. :contentReference[oaicite:1]{index=1}"
    },
    {
        "question": "What were the main results or findings?",
        "ground_truth": "The digital twin closely reproduced participants' linguistic characteristics and achieved reconstruction and MoCA prediction errors comparable to real responses, while significantly outperforming a baseline GPT model in preserving cognitive and linguistic patterns. :contentReference[oaicite:2]{index=2}"
    },
    {
        "question": "What dataset or data was used?",
        "ground_truth": "The study used the I-CONECT dataset, which contains natural, longitudinal conversations from adults aged 75 and older, including both cognitively normal participants and those with mild cognitive impairment. Five participants with the most conversation sessions were selected for the experiments. :contentReference[oaicite:3]{index=3}"
    },
    {
        "question": "What problem does this work address?",
        "ground_truth": "The paper addresses the challenge of personalized, non-invasive monitoring of cognitive decline in older adults by modeling individual conversational behavior instead of relying solely on traditional clinical assessments or population-level prediction models. :contentReference[oaicite:4]{index=4}"
    },
]
