from dotenv import load_dotenv
import pdb

load_dotenv()

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

class Player:
    def __init__(self, model):
        self.observation = []
        self.model = model
        self.concept = None
        self.history = []
    
    def initialize_host (self):
        template = """
        You are the host of a game where a player asks questions about a 
        thing to guess what it is.

        Write the name of a thing. It must be a common object.
        It must be a single word. Do not write anything else.
        Only write the name of the thing with no punctuation.

        Here is a list of things you cannot use:
        {history}

        """

        prompt = PromptTemplate.from_template(template)
        chain = prompt | self.model | StrOutputParser()

        self.concept = chain.invoke({"history": "\n".join(self.history)})
        self.history.append(self.concept)

        print(f"Concept: {self.concept}")

    def initialize_player(self):
        self.observation = []
    
    def ask(self, questions_left):
        template = """
        You are a player in a game where you need to ask Yes/No
        questions about a thing and guess what it is.

        The thing is a commmon object. It is a single word.

        Here are the questions you have already asked:
        {observations}

        You only have {questions_left} questions left to ask. You want to guess
        in as few questions as possible. If there's only 1 question left. You 
        must make a guess or you'll lose the game. Be aggressive and try to guess
        the thing as soon as possible.

        Do not ask questions that you have already asked before.

        Only binary questions are allowed. The questions must be answered with a Yes/No.

        Be as concise as possible when asking a question. Do not announce that you
        will ask questions. Do not say "Let's get started", or introduce your question. 
        Just write the question.

        Examples of good questions:

        - Is it a fruit?
        - Is it bigger than a car ? 
        - Is it alive ? 

        Examples of bad questions:
        - Can I ask you a question?
        - Can you tell me more about a thing? 
        - What is the thing? 
        - How does the thing look like?

        """
        prompt = PromptTemplate.from_template(template)
        chain = prompt | self.model | StrOutputParser()
        return chain.invoke(
            {
                "observations": "\n".join(self.observation),
                "questions_left": questions_left,
            }
        )
    
    def answer (self, question):
        template = """
        You are the host of a game where a player asks questions about a {concept} trying 
        to guess what it is.

        The player has asked you the following question. {question}

        If the player guessed that the thing is "{concept}", answer with the word "GUESSED".
        If the question refers to "{concept}", answer with the word "GUESSED".

        If the player didn't guessed, answer the question with a simple Yees or No. Do not say
        anything else. Do not use any punctuation.
        """
        prompt = PromptTemplate.from_template(template)
        chain = prompt | self.model | StrOutputParser()
        return chain.invoke({"concept": self.concept, "question": question})
    
    def add_observation(self, question, answer):
        self.observation.append(f"Question: {question}. Answer: {answer}")


class Game:
    def __init__(self, model1, model2, rounds=3, questions=20):
        self.model1 = model1
        self.model2 = model2
        self.rounds = rounds
        self.questions = questions
    
    def _play (self, host, player):
            host.initialize_host()
            player.initialize_player()
            for question_index in range(self.questions):
                question = player.ask(self.questions - question_index)
                answer = host.answer (question)
                print(f"Question {question_index +1}: {question}. Answer: {answer}")

                player.add_observation(question, answer)

                if "guessed" in answer.lower ():
                    return True
                
            return False
        
    def start(self):
        players = {
            "0": {
                "player": Player(model=self.model1),
                "score": 0,
            },
            "1" : {
                "player": Player(model=self.model2),
                "score": 0,
            },
        }

        host_index = 0
        for round in range(self.rounds):
            print(f"\nRound {round + 1}. Player {host_index +1} is the host")

            player_index = 1 - host_index

            if self._play(
                players[str(host_index)]["player"], players[str(player_index)]["player"]
            ):
                print(f"Player {player_index +1} guessed correctly")
                players[str(player_index)]["score"] += 1
            else:
                print(f"Player {player_index + 1} didn't guess correctly")
                players[str(host_index)]["score"] += 1
            
            host_index = 1 - host_index
        
        print("Final score:")
        print(f"Player 1: {players['0']['score']}")
        print(f"Player 2: {players['1']['score']}")
            
game = Game(
    model1 = Ollama(model="llama3"),
    model2 = Ollama(model="llama3"),
    rounds=7
)

game.start()

"""
player = Player(model=Ollama(model="llama3"))
player.initialize_host ()
question = player.ask(20)
print(question)
answer = player.answer(question)
print(answer)
"""
