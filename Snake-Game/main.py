import time
from snake import Snake
from detect import Food
from turtle import Screen
from score import Score

sa = Screen()
sa.setup(width=600, height=600)
sa.bgcolor('black')
sa.title('Snake Game')
sa.tracer(0)
snake = Snake()
food = Food()
score = Score()
sa.listen()
sa.onkey(fun=snake.up, key="w")
sa.onkey(fun=snake.down, key="s")
sa.onkey(fun=snake.left, key='a')
sa.onkey(fun=snake.right, key='d')
game_on = True
while game_on:
    sa.update()
    time.sleep(0.1)
    snake.move()
    if snake.head.distance(food) < 10:
        food.move_food()
        snake.extend_tail()
        score.count()
    if snake.head.xcor() > 280 or snake.head.xcor() < -300 or snake.head.ycor() > 305 or snake.head.ycor() < -295:
        score.reset()
        snake.reset()
    for body in snake.snake_body[1:]:
        if snake.head.distance(body) < 10:
            score.reset()
            snake.reset()
sa.exitonclick()
