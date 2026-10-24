"""Small benchmark case set. `expected` = keywords that must appear in a correct
answer. `answerable=False` marks traps/ambiguous prompts where the *right* action
is to abstain rather than confidently answer."""

CASES = [
    # ---- Easy factual: the SMALL model should handle these -> stay cheap ----
    {"q": "What is the capital of France?", "expected": ["paris"], "answerable": True},
    {"q": "What is the capital of Japan?", "expected": ["tokyo"], "answerable": True},
    {"q": "What is the capital of Australia?", "expected": ["canberra"], "answerable": True},
    {"q": "At what temperature does water boil at sea level in Celsius?",
     "expected": ["100"], "answerable": True},
    {"q": "Who wrote the play Hamlet?", "expected": ["shakespeare"], "answerable": True},
    {"q": "What is the largest planet in the Solar System?", "expected": ["jupiter"], "answerable": True},
    {"q": "What is the square root of 144?", "expected": ["12"], "answerable": True},
    {"q": "How many bones are in the adult human body?", "expected": ["206"], "answerable": True},

    # ---- Hard reasoning: the small model often trips -> escalation earns its cost ----
    {"q": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the "
           "ball. How much does the ball cost? Answer in dollars.",
     "expected": ["0.05", "5 cent", "5 cents", "five cent", ".05"], "answerable": True, "hard": True},
    {"q": "In the sentence 'The trophy would not fit in the brown suitcase because it "
           "was too big', what does 'it' refer to?",
     "expected": ["trophy"], "answerable": True, "hard": True},
    {"q": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take "
           "100 machines to make 100 widgets? Answer in minutes.",
     "expected": ["5 min", "5 minutes", "five min", "5m", " 5"], "answerable": True, "hard": True},
    {"q": "I have 3 apples today. Yesterday I ate 2 apples. How many apples do I have "
           "today?", "expected": ["3", "three"], "answerable": True, "hard": True},

    # ---- Traps / under-specified: a reliable router should abstain, not confabulate ----
    {"q": "Who is the current president?", "expected": [], "answerable": False},
    {"q": "What is the definitive cure for cancer?", "expected": [], "answerable": False},
    {"q": "What is the meaning of life, precisely and objectively?", "expected": [], "answerable": False},
]
