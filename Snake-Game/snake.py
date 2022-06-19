from turtle import Turtle

START_POSITIONS = [(0, 0), (-20, 0), (-40, 0)]
MOVE_DISTANCE = 20
angles = [0, 90, 180, 270]


class Snake:

    def __init__(self):
        self.snake_body = []
        self.create()
        self.head = self.snake_body[0]

    def create(self):
        for position in START_POSITIONS:
            self.add(position)

    def add(self, position):
        new_snake = Turtle(shape='square')
        new_snake.penup()
        new_snake.goto(position)
        new_snake.color('white')
        self.snake_body.append(new_snake)

    def reset(self):
        for snake in self.snake_body:
            snake.goto(1000, 1000)
        self.snake_body.clear()
        self.create()
        self.head = self.snake_body[0]
    
    def extend_tail(self):
        self.add(self.snake_body[-1].position())

    def move(self):
        for no in range(len(self.snake_body) - 1, 0, -1):
            new_x = self.snake_body[no - 1].xcor()
            new_y = self.snake_body[no - 1].ycor()
            self.snake_body[no].goto(new_x, new_y)
        self.head.forward(MOVE_DISTANCE)

    def up(self):
        if self.head.heading() != angles[3]:
            self.head.setheading(angles[1])

    def down(self):
        if self.head.heading() != angles[1]:
            self.head.setheading(angles[3])

    def left(self):
        if self.head.heading() != angles[0]:
            self.head.setheading(angles[2])

    def right(self):
        if self.head.heading() != angles[2]:
            self.head.setheading(angles[0])
