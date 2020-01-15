# War-Card-Game-Analysis
Using Python (Anaconda-Spyder) 1000s of hands of war are analyzed and graphed.

When I was younger one of my brothers and two of my cousins would always play the game of war. My brother, being older, would change the rules by playing with only four cards while my cousins and I split the rest of the deck. However, the four cards he played with were the four aces. My cousins and I never seemed to win. I wanted to see how often, under these circumstances, we would win.

I decided to use Python since I have just recently learned the language from a course on LinkedIn. Writing this application would enable me to delve deeper into some of the more complex aspects of this language. This program will take the classic war game and play a defined number of hands while saving the results of this to a dataFrame. This file can later be saved to a file. 

Note: The code and classes are heavily commented.

As this code is written, it will perform two tests:

First, a performance test. It will print out and write the results of this performance test to a file called: War_Stats_Total.csv
1000 games are attempted per set. If a game exceeds 5000 rounds then the game is aborted and the results are ignored. This only ever occured when discard piles were not shuffled.
These results include:

1.	hands - The number and size of all the player hands for this set.

2.	cards - Total cards in play, the deck as defined has 52 cards and needs to be evenly dispersed between players. Extra cards are discarded at random.

3.	Players - Total number of players for the set of games.

4.	Shuffling/cheats before - Percentage of times player 1 won, where discards are shuffled, and aces are given to player 1 before dealing out hands.

5.	Games (SCB) - Total number of games for number 4.

6.	Time (SCB) - Amount of time this set took for number 4.

7.	Shuffling/cheats after - Percentage of times player 1 won, where discards are shuffled, and aces are given to player 1 after dealing out hands. (If there are less than 52 cards in these rounds, there is a chance an ace will be removed before it can be given to player 1.)

8.	Games (SCB) - Total number of games for number 7.

9.	Time (SCB) - Amount of time this set took for number 7.

10.	No shuffling/cheats before - Percentage of times player 1 won, where discards are NOT shuffled, and aces are given to player 1 before dealing out hands.

11.	Games (SCB) - Total number of games for number 10.

12.	Time (SCB) - Amount of time this set took for number 10.

13.	No shuffling/cheats after - Percentage of times player 1 won, where discards are NOT shuffled, and aces are given to player 1 after dealing out hands. (As with number 5, there is a chance an ace will be removed before it can be given to player 1.)

14.	Games (SCB) - Total number of games for number 13.

15.	Time (SCB) - Amount of time this set took for number 13.

16.	Total Elapsed Time - Total amount of time all 4 sets took. Note: for the final row this will be the total elapsed time for all the sets.


Second, a series of 4 graphs are generated using the following scenarios:

1.	A value-count graph comparing the number of rounds to win a game to the number of occurrences.

2.	A scatter plot breaking the wins for each player and apply a kernel density estimate to color shift the colors for each player to more clearly display the number of rounds it took to win.

3.	A 3d scatter plot comparing the number of rounds to win a game to the number of occurrences broken out by players.

4.	A pie graph that displays the percentage of wins each player had.
