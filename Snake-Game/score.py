from turtle import Turtle


class Score(Turtle):

    def __init__(self):
        super().__init__()
        self.score = 0
        self.high_score = 0
        self.color('yellow')
        self.hideturtle()
        self.penup()
        self.goto(0, 270)
        self.redundant()

    def redundant(self):
        with open('max_score.txt') as file:
            score = file.read()
            self.write(arg='score is {}  High score is {}'.format(self.score, score), align='center',
                       font=('arial', 22, 'normal'))

    def count(self):
        self.clear()
        self.score += 1
        self.redundant()

    def reset(self):
        self.clear()
        if self.high_score < self.score:
            with open('max_score.txt', 'w') as file:
                file.write(f'{self.score}')
        self.score = 0
        self.redundant()
