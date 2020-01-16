# Copyright @ 2020 Rick Oldenburger
# Version Date       Description
# .5      1/1/2020   Initial class card_games written
# .6      1/4/2020   Added to play and playSet for multiple sets, 
#                     and gave option to not print assesment of game play
# .7      1/7/2020   Added ability to set deck size and apply specific cards
#                     to a player (addCheats).
# .8      1/10/2020  Added line, scatter, and pie graphs
# .9      1/11/2020  Added 3d scatter graph
# 1.0     1/12/2020  Made the 3d graph and pie graph colors match,
#                     had to darken (_darken) the 3d graph slightly since it
#                     was too difficult to view otherwise
# 1.1     1/13/2020  Added flush = True sto print statements since Komodo no
#                     longer flushed the prints after the latest update
#                    Added code to clear intial graphs: 
#                     plt.figure(figsize=(10, 8)) since the DataFrame plot
#                     since pandas Series object did not reset certain settings
#                     from previous plots when executing the command: plot
#                     if the previous plot was still available
# 1.2     1/15/2020   Moved to GitHub

import random
from collections import defaultdict
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as kde
from matplotlib.colors import Normalize
from matplotlib import cm
import matplotlib.transforms
# Even though this generates a warning, we need it for the 3d graph
from mpl_toolkits.mplot3d import Axes3D
import time
from datetime import timedelta


class War_card_game:
    """
    Card game class that plays matches of the card game war and generates a
    statistical pandas dataFrame that can be utilized for analyzing the 
    results.
    """
    def __init__(self, cards, suits, unSuited=None, reshuffleWins=True, 
                 applyCheatsPreSplit=True):
        """
        Parms for when the class is first created
    
        cards --> Contains all the cards that are needed for the deck it is
          a dictionary broken down by card and then the value of the card
          {'A' : 13} (DICTIONARY)
      
        suits --> Contains all the suits for the deck hearts, spades, etc.
          There is no valule for this field since it will simply be used to 
          extend the number of cards in the deck.
          ('Heart','Spade') (LIST)
      
        unsuited --> Contains any additonal cards that are not associated to a 
          suit. This is defined in the same manner as cards name : value
          {'joker': 14} (DICTIONARY) Optional defaults to None
      
        reshuffleWins --> If set to false then the deck of discards is not
          reshuffled before being used again by the player. Although the rules
          are not clear here if cards are not reshuffled the games will take a 
          lot longer if they are not.
          (BOOL) Optional Defaults to True
          
       applyCheatsPreSplit --> perform cheats before or after splitting the
         deck. If performed after splitting the deck, there is the possibility
         of dropping a card before it can be assigned to a specific player.
         (BOOL) Optional Defaults to True
          
        """
        self.setDeck(cards, suits, unSuited)
        self._reshuffleWins = reshuffleWins
        self._applyCheatsPreSplit = applyCheatsPreSplit
        self._stat = None
        self._stats = pd.DataFrame()
        self._handSize = defaultdict(int)
        self._cheats = defaultdict(int)
        self._players = 0
        self._FINAL = 0
        self._rounds = 0
        
    def _shuffleDict(self, d):
        """
        Since there is no shuffle feature for dictionaries, that I know of,
        I had to convert the dictionary to a List.
          
        d --> the dictionary to shuffle
        
        RETURNS --> the shuffled dictionary
        """
        keys = list(d.items())  # Turn the dictionary into a list of items
        random.shuffle(keys)  # Shuffle them
        return dict(keys)  # Return the list as a a dictionary

    def _makeColors(self, vals, map="gist_rainbow_r", amt = 0.0):
        """
        This function returns a scaled map based on all the values passed
        to the function. It will assign one scaled color to each value passed

        vals --> values to be scaled 
    
        map --> optional string of the scale to use. defaults to gist_rainbow_r
          since this is the scale I chose to use for my graph.
        """
        
        colors = np.zeros((len(vals),3))
        norm = Normalize(vmin=vals.min(), vmax=vals.max())
        #Can put any colormap you like here.
        if amt == 0.0:
            # no point in calling darken if no change is requested
            colors = [cm.ScalarMappable(norm=norm, cmap=map).to_rgba(val)
                for val in vals]
        else:
            colors = [self._darken(cm.ScalarMappable(norm=norm,
                                                     cmap=map).to_rgba(val),
                                   amt)
                for val in vals]
            
        return colors

    def _darken(self, color, amt=.1):
        """
        This function will take a passed color value and darken it slighlty.
        
        color --> original color to be made darker
    
        amt --> amount to darken (or lighten if a value less than 1 is passed
        """
        c = list(color)
        c[0] = 1 - amt*(1-c[0])
        if c[0] < 0: c[0] = 0 # less than zero values generate an error
        c[1] = 1 - amt*(1-c[1])
        if c[1] < 0: c[1] = 0
        c[2] = 1 - amt*(1-c[2])
        if c[2] < 0: c[2] = 0    
        return c
        
    def _makeStats(self, *args):
        """
        Function called to create the statisical records. This function is 
        meant to be called from within the class.
        
        This function takes one parm. This parm works as follows:
          If one value is passed this will be flagged as the last round of
          play. It will also assign the winner to the second field by finding
          the first value that is not assigned.
          
         If two parms are specified then it is assumed that this player is 
           out of cards and 
        """
        curSize = self._players + 2

        if len(args) > 0:
            if len(self._stats.columns) != curSize:
                self._makeStats()

            if len(args) == 1:
                # If only one arg is sent, then this is the winner
                # we set the winner value by finding the first
                #   and hopefully only 0 number under a player
                self._stat[0] = args[0]
                self._stat[1] = next(
                    (cntr + 1 for cntr, x in enumerate(self._stat[2:])
                     if x == 0), self._stat[1:].index(max(self._stat)))
                    # Should never happen but if all the values are populated
                    #   then return the index of the highest value
                                
                self._stats.loc[len(self._stats.index) + 1] = self._stat
                self._stat = [0] * curSize
            else:
                # If 2 parms are sent then we are flagging a loser
                self._stat[args[1] + 1] = args[0]
        else:
            # create a datFrame with a key based on the match number
            # Number of Rounds
            # Winner flag, as well as values for each player
            # that are set to the number of Rounds that player lasted
            if len(self._stats.columns) != curSize:
                col = ["Match", "Rounds", "Winner"]
                plr = lambda x: "Player " + str(x + 1)
                colDyn = [plr(x) for x in range(self._players)]
                self._stats = pd.DataFrame(columns=col + colDyn,
                                           dtype=np.int64)
                self._stats.set_index("Match", inplace=True)
                self._stat = [0] * (curSize)    

    def setDeck(self, cards, suits, unSuited=None):
        """
        Sets the deck for this class
        Uses the first three parms for defining the class
        """
        deck = {}
        for suit in suits:
            for card, val in cards.items():
                deck["".join([card, "-", suit])] = val
        if isinstance(unSuited, dict):
            deck.update(unSuited)
        self._deck = deck
        
    def getDeck(self):
        """
        Returns the current deck
        """
        return self._deck
        

    def setReshuffleWins(self, reshuffleWins=True):
        """
        Allows for changing the reshuffle value. that will prevent shuffling
        of the already played cards for each player.
        """
        self._reshuffleWins = reshuffleWins

    def getReshuffleWins(self):
        """
        Returns the current reshiffleWins setting.
        """
        return self._reshuffleWins

    def setApplyCheatsPreSplit(self, applyCheatsPreSplit=True):
        """
        Allows for changing whether the cheats are applied before splitting the
        deck.
        """
        self._applyCheatsPreSplit = applyCheatsPreSplit

    def getApplyCheatsPreSplit(self):
        """
        Returns the current status of when the split is applied
        """
        return self._applyCheatsPreSplit

    def clearCheats(self):
        """
        Clears all cheats from the class
        """
        self._cheats.clear()

    def addCheats(self, cards, player=1):
        """
        Allows for adding additional cards that a specific player
        must always have. This will add additonal cards to the list.
        
        cards --> A list of cards that are to be always be given to a specific
          player.
          ["Heart-A", "Spade-A"] (LIST)
        
        player --> The specific player to assign these cards to. This card
          defaults to player 1. If a card is already assigned to a different
          player then it will be assigned a new player.
          2 (INT)
        """
        if isinstance(cards, list):
            for card in cards:
                if card in self._deck:
                    self._cheats[card] = player
                else:
                    raise ValueError(
                        "Specified card: " + card +
                        " not specified in the deck")
        elif cards in self._deck:
            self._cheats[cards] = player
        else:
            raise ValueError("Specified card:" + card +
                             " not specified in the deck")

    def setCheats(self, cards, player=1):
        """
        This function works exactly like addCheats, except that it clears the
        cheats first        
        """
        self.clearCheats()
        self.addCheats(cards, player)

    def getCheats(self):
        """
        Returns a dictionary of all the cards assigned to each player
        """
        return dict(self._cheats)

    def setPlayerHand(self, size, player=1):
        """
        Sets the size of a players hand to a specific size.
        
        size --> An integer that is specifies how big a speficic hand will be
        Note: if this value is less than the size specified in the addCheats
          then the deck will be larger than the one specified here.
        """
        self._handSize[player] = size

    def clearPlayerHand(self):
        """
        Clears the player hand size restrictions.
        """
        self._handSize.clear()

    def getPlayerHand(self):
        """
        Returns the current hand sizes that have been specified.
        """
        return dict(self._handSize)

    def clearStats(self):
        """
        Resets the statistical data.
        """
        self._stats = pd.DataFrame()

    def getStats(self):
        """
        Returns the DataFrame that contains all the statistics
        calculated by calling the battle or play functions
        """
        return self._stats

    def getPlayers(self):
        """
        Returns the number of players currently involved.
        Note: if the number of players changes, the statistics
          will be reset.   
        """
        return self._players
    
    def getRound(self):
        """
        Returns the current round.
        """
        return self._rounds
    
    def getWinner(self):
        """
        Returns the last winner. If we are in the middle of a game
        this will return a zero.
        """
        return self._FINAL
        
    def split_deck(self, num=2, message=False):
        """
        This function will split the deck for a specified number of player
        hands
        
        num --> Is the number of players that the deck will be split between.
          This parm is optional and defaults to 2 (INT)
        
        message --> if set to true will allow for verbose messages about
          what the program is currently doing. This parm is optional and
          defaults to False. (BOOL)
          
        Note: the program will always give players an even number of cards.
          For example if the are 52 cards and three players each player will
          be dealt 17 cards, one card at random will be dropped from the game.
          (Furthermore, if the card that is dropped was assigned to a player, 
           then that player will not get that card for that game, unless the 
           flag _applyCheatsPreSplit is set to true)
        """
        self._players = num
        self._CurrentPlayers = [1] * num
        self._rounds = 0
        self._FINAL = 0

        # Randomize the deck
        if message:
            print("Shuffling")
        tempDeck = self._shuffleDict(self._deck)

        # Clear out the current decks
        self._decks = [{} for _ in range(num)]
        self._won = [{} for _ in range(num)]

        # build forced deck cards first if the flag _applyCheatsPreSplit
        if self._applyCheatsPreSplit:
            if message:
                print("Applying Cheats")

            for card, player in self._cheats.items():
                self._decks[player-1][card] = tempDeck[card]
                del tempDeck[card]

        # Determine if any specific deck sizes are defined
        tmpLst = list(value for key, value in self._handSize.items()
                      if int(key) < num)
        v, c = sum(tmpLst), len(tmpLst)

        if num > c:
            n = (len(self._deck) - v) // (num - c)
        else:
            n = 0

        i = iter(tempDeck.items())

        # split the deck num ways. all decks are of equal size, extra cards are
        #   randomly eliminated
        # Note 1: the option setPlayerHand will override deck sizes for that
        #   specific player
        # Note 2: if setCheats mandates a bigger deck size than the split or
        #   setPlayerHand would specify then the number of cards specified
        #   in setCheats will take precedence
        for cntr in range(num):
            cur = len(self._decks[cntr])
            if cntr + 1 in self._handSize:
                splt = self._handSize[cntr + 1] - cur
            else:
                splt = n - cur
            if message:
                print("Dealing player:", str(cntr + 1), "Cards", str(splt+cur))
            if splt > 0:
                self._decks[cntr].update(dict(itertools.islice(i, splt)))
            self._decks[cntr] = self._shuffleDict(self._decks[cntr])
            #self._decks.append(dict(itertools.islice(i, splt)))

        # Apply any Cheats after the deck is split, if the flag is set
        if not self._applyCheatsPreSplit and len(self._cheats) > 0:
            if message:
                print("Applying Cheats")
                
            # Create a tempory list of num empty dictionaries
            w = [dict() for x in range(num + 1)]  

            # Get the lengths for each deck we ant to try to maintain
            x = []
            for i in range(num):
                x.append(len(self._decks[i]))

            # Find and remove cards that are to be assigned to specific player
            for card, pos in self._cheats.items():
                for d in self._decks:
                    if card in d:
                        w[pos - 1][card] = d[card]
                        del d[card]
                        exit

            # Build list of extra entries and remove them from the players hand
            for cntr, d in enumerate(self._decks):
                dev = x[cntr] - (len(d) + len(w[cntr]))

                for _ in range(dev, 0):
                    if len(d) > 0:
                        card = list(d.keys())[0]
                        w[num][card] = d[card]
                        del d[card]

            # Add the extra entries to the correct decks
            for cntr, d in enumerate(self._decks):
                dev = x[cntr] - (len(d) + len(w[cntr]))
                if len(w[cntr]) > 0:
                    d.update(w[cntr])
                for i in range(dev):
                    if len(w[num]) > 0:
                        card = list(w[num].keys())[0]
                        d[card] = w[num][card]
                        del w[num][card]
                    else:
                        exit
                # Needed to return to the self._decks[cntr] deck since the 
                # function shuffleDict returns a new dictionary which will 
                # break the link between d and self._decks[cntr]
                self._decks[cntr] = self._shuffleDict(d)

        if message:
            for cntr, d in enumerate(self._decks):
                print("Player:", cntr + 1,
                      "Number of Cards:", len(d)
                )
                # Only flush the print after all shuffling, cheats, etc. 
                print(d, flush = True)

    def battle(self, message=False, _ignore=False):
        """
        This function will play a single hand of war. This function is designed
        to be called recursively. This recursion call is only meant for when
        WAR is found, since there will be no evaluation of the cards when
        the function is called in this manner.
        
        message --> if set to true will allow for verbose messages about
          what the program is currently doing. This parm is optional and
          defaults to False. (BOOL)
          
        _ignore --> This is meant to be utilized by the function itself and not
          actually used for an external call. If set to true no card evaluation
          will take place for that round.
          defaults to False. (BOOL)
          
        RETURN --> Returns the winner of the round, 
         -1 if there are no more players and we know who won the game
         None if we are in the middle of a WAR round.
        """
        
        if sum(self._CurrentPlayers) > 1: 
            # If there are still any active players play the next hand
            self._rounds += 1
            entry = [None] * self._players
            for cntr, deck in enumerate(self._decks): # Loop thorugh players
                if len(deck) > 0:
                    # Player currently has cards in active deck
                    entry[cntr] = deck.popitem()
                elif len(self._won[cntr]) > 0:
                    # Player has no active deck but has a played deck
                    
                    # Use the option _reshuffleWins to determine if we will
                    #   shuffle the played deck. 
                    if self._reshuffleWins:
                        if message:
                            print("Player:", cntr + 1,
                                  "Shuffling", len(self._won[cntr]), "Cards",
                            flush = True)
                        self._decks[cntr] = self._shuffleDict(self._won[cntr])
                    else:
                        if message:
                            print("Player:", cntr + 1,
                                  "Using", len(self._won[cntr]), "Cards",
                            flush = True)
                        self._decks[cntr] = self._won[cntr]
                        
                    self._won[cntr] = {}
                    entry[cntr] = self._decks[cntr].popitem()
                else:
                    # Well it looks like this player is out of cards :(
                    if sum(self._CurrentPlayers) > 1:
                        if (self._CurrentPlayers[cntr]) == 1:
                            # must subtract one from rounds since we do not 
                            # identify winners/losers until the next rounb
                            rounds = self._rounds - 1
                            if message:
                                print("Round:", rounds,
                                      "Player", cntr + 1, "*LOSES*",
                                flush = True)
                            self._CurrentPlayers[cntr] = 0
                            # Update stats for loser
                            self._makeStats(rounds, cntr + 1) 
                    else:
                        self._FINAL = cntr + 1
                        self._rounds -= 1
                        if message:
                            print("Total Rounds:", str(self._rounds),
                                   "Player:", self._FINAL, "*WINS*",
                            flush = True)
                        # Update stats for a winner
                        self._makeStats(self._rounds) 
                        return -1
                    entry[cntr] = ("None", 0)

            # We now evaluate the players unless the call is dictating we do
            #   not because we are playing a WAR hand or if there is only 1 
            #   player we cannot have a WAR hand.
            # Note: a tie between any two players will constitute a WAR.
            if not _ignore or sum(self._CurrentPlayers) == 1:
                winner = entry.index(max(entry, key=lambda x: x[1]))
                if sum(self._CurrentPlayers) == 1:
                    self._FINAL = winner + 1
                    self._rounds -= 1
                    if message:
                        print("Total Rounds:", str(self._rounds),
                              "Player:", self._FINAL, "*WINS*",
                        flush = True)
                    self._makeStats(self._rounds) # Update stats for a winner
                    return -1
                score = entry[winner][1]
                winners = len(list(filter(lambda x: x[1] == score, entry)))
                if message:
                    print("Played:", entry, flush = True)
                if winners == 1:
                    if message:
                        print("Round:", str(self._rounds),
                              "Player: ", str(winner + 1), "wins",
                        flush = True)
                    self._won[winner] = dict(
                        list(self._won[winner].items())
                             + list(filter(lambda x: x[0] != "None", entry))
                    )
                    return winner
                else:
                    winner = self.battle(message, True)
                    if winner != None:
                        self._won[winner] = dict(
                            list(self._won[winner].items())
                            + list(filter(lambda x: x[0] != "None", entry))
                        )
                        return winner
            else:
                # we are in a round of WAR, we will call this function again
                if message:
                    print("WAR: ", entry, flush = True)
                winner = self.battle(message)
                if winner != None:
                    self._won[winner] = dict(
                        list(self._won[winner].items())
                        + list(filter(lambda x: x[0] != "None", entry))
                    )
                    return winner
        else:
            # If called again we will print out the final stats, even if the
            #   message parm is set to false.
            print("Total Rounds:", self._rounds,
                  "Player:", self._FINAL, "*WINS*", flush = True)
            return -1

    def play(self, players=2, message=False, maxRounds=0):
        """
        This function will play an entire game
        
        players --> sets the number of players for this game.
          default 2 (INT)
        Note: if this number changes between calls the stats will be reset
          because the size of the DataFrame will change
          
        message --> if set to true will allow for verbose messages about
          what the program is currently doing. This parm is optional and
          defaults to False. (BOOL)
          
        maxRounds --> The maximum number of rounds to have before ending the
          game.
          Defaults to zero (for infinite) (INT)
        """
        if players < 2:
            raise IndexError("Players must be more than 1")
        else:
            ds = len(self.getDeck())
            if players > ds:
                raise IndexError(f"Players must be less than deck size: {ds}")
        
        self.split_deck(players, message) # Split the deck
        
        # Since this loop is called iteratively, I chose to use a separate
        #   loop depending on how the function is called
        if maxRounds == 0:
            while 1 == 1:
                if self.battle(message) == -1:
                    break
        else:
            for _ in range(maxRounds):
                if self.battle(message) == -1:
                    break
        
    def playMatch(self, players = 2, matches = 10000, maxRounds=0):
        """
        Play a sepcified number of matches
        
        players --> Number of players for this match.
          Changing the number of players will automatically reset the stats.
          Defaults to 2 (INT)
          
        matches --> Number of matches to play.
         Defaults to 10000 (INT)
         
        maxRounds --> Maximum number of rounds to play before aborting a game
          a 0 means to play an infinite number of times.
          Defaults to 0 (INT)
        """
        for _ in range(matches):
            self.play(players, False, maxRounds)
            
    def valueCountGraph(self):
        """
        Display a value count graph for our statistic data
        """
        df = self.getStats()
        totalMatches = len(df)
        if (totalMatches == 0):
            print("Need to play at least one match for a graph", flush = True)
            return
        
        # generate a series for graphing later
        vc = df["Rounds"].value_counts()

        # Set the graph size using "plt.figure"  
        # This resolved an issue where setting from a previously displayed 
        # plot was overriding the current plot. 
        # This was on a problem in Anaconda where a previous graph might still
        # exist; otherwise, I could have just set the size by using:
        # x = vc.plot(legend=True, figsize=[10, 8])
        plt.figure(figsize=(10, 8))
        
        x = vc.plot(legend=True)
        x.set_xlabel("Number of Rounds to Win")
        x.set_ylabel("Occurrences")
        x.set_title("Value-Count Graph (Matches: " + str(totalMatches) + ")")
        
        Occurences = max(vc) + 1
        # for small values set y tic mark text
        if Occurences <= 20:
            yText = [] # Stores our Y-Axis text
            
            for i in range(1, Occurences):
                yText.append(f'${i}$')
            plt.yticks(np.arange(1, Occurences), yText, ha="center")
        
        plt.show()

    def valueCountScatter3D(self):
        """
        Display a value count Scatter plot for our statistical data
        """
        df = self.getStats()
        totalMatches = len(df)
        if (totalMatches == 0):
            print("Need to play at least one match for a graph", flush = True)
            return
                
        # generate a series for graphing later
        vc = df["Rounds"].value_counts()
        Occurences = max(vc) + 1 # get the maximum count of hands
        
        players = self.getPlayers() + 1 # We need this to build our player text
               
        yText = [] # Stores our Z-Axis text
        # Loop through each player to build them as a standard 
        # (Ignore the warning we need the percent symbol)
        # We will build a separate color scheme for each play to enable
        #   finding out where the average number rounds for that player
        #   resided
        for i in range(1, players):
            tmp = df[(df.Winner==i)]["Rounds"]
            winPercent = round(len(tmp)/totalMatches*100,1)
            yText.append(f'${i}$\n$({winPercent}\%)$')
         
        # We need to take the value counts and renmae the axis to "Rounds"
        #  an the index name to "Counts" so that we can use the columns
        #  in opur scatter plot.        
        vc = vc.rename_axis("Rounds").reset_index(name="Counts")
        
        # we need to "inner" join vc to df to extract the winners for the 
        #   points on our 3d graph
        # Note: If the columns names had been different, we could use
        #   left_on = ... and right_on = ...
        df = df.merge(vc, on="Rounds", how = "inner")

        # Pick a color from the rainbow for each player
        # Note: Since I wanted to use the samwe pallet as for the pie chart
        #   I had to darken the Paired mapping a little. This way the colors
        #   will essentially the same as in the pie chart (just a little darke)
        color = self._makeColors(np.asarray(df["Winner"], dtype=np.int64),
                                            "Paired", 1.5)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df["Counts"], df["Winner"], df["Rounds"],
                   c=color, marker='o')
        ax.set_xlabel("Occurrences", labelpad=10)
        ax.set_ylabel("Winner", labelpad=20)
        ax.set_zlabel("Number of Rounds to Win")
        ax.set_title("Value-Count Scatter 3D Plot (Matches: " +
                      str(totalMatches) + ")")
        
        # for small values set y tic mark text
        if Occurences <= 20:
            xText = [] # Stores our Y-Axis text
            for i in range(1, Occurences):
                xText.append(f'${i}$')
            plt.xticks(np.arange(1, Occurences), xText, ha="center")
        plt.yticks(np.arange(1, players),   yText, ha="center")
        plt.show()

    def kernelDensityScatter(self):
        """
        Display a value count Scatter plot using Kernel Density to change the
        colors of our plot based on how dense the resulting plot is. This will
        turn the color closer to red when the values become more densely
        packed.        
        """
        df = self.getStats()
        totalMatches = len(df)
        if (totalMatches == 0):
            print("Need to play at least one match for a graph", flush = True)
            return
        players = self.getPlayers() # We need this to build our player text

        c = [] # Stores our colors
        yText = [] # Stores our Y-Axis text
        
        # Loop through each player to build them as a standard 
        # (Ignore the warning we need the percent symbol)
        # We will build a separate color scheme for each play to enable
        #   finding out where the average number rounds for that player
        #   resided
        for i in range(1, players+1):
            tmp = df[(df.Winner==i)]["Rounds"]
            wins = len(tmp)
            winPercent = round(len(tmp)/totalMatches*100,1)
            yText.append(f'${i}$\n$({winPercent}\%)$')
            if wins > 1:
                try:
                    densObj = kde(tmp)
                    c += self._makeColors(densObj.evaluate(tmp))
                except:
                    # if we have problems just use black
                    c += [(0.0, 0.0, 0.0, 1.0)] *wins
            elif wins == 1:
                # When only 1 win, use the color black
                c += [(0.0, 0.0, 0.0, 1.0)] 
                
        st = df.sort_values(['Winner','Match'])
        
        fig, ax = plt.subplots(figsize=[10, 2 + players*.5]) # we need s subplot to offeset our player labels
        ax.scatter(st['Rounds'], st['Winner'], color=c, s=35)
        plt.yticks(np.arange(1, players+1),yText,ha="center")

        # We need to offset our y labels so they look nice :)
        dx, dy = -15/72, 0/72 # need to move 15 pixels to the left
        offset = matplotlib.transforms.ScaledTranslation(dx, dy, 
                                                         fig.dpi_scale_trans)
        for label in ax.yaxis.get_majorticklabels():
            label.set_transform(label.get_transform() + offset)

        plt.title(
"""Kernel Density Estimate (KDE) Scatter Plot 
Turns until winning (Matches: """ + str(totalMatches) + ")")
        plt.ylabel("Player (Percentage of Wins)")
        plt.xlabel("Number of Rounds to Win")
        plt.show()
        
    def pieGraph(self):
        df = self.getStats()
        totalMatches = len(df)
        if (totalMatches == 0):
            print("Need to play at least one match for a graph", flush = True)
            return
        
        vc = df["Winner"].value_counts() # get a list of the number of winners
        wins = vc.tolist() # get a list of winners
        labels = list(vc.index) # get a list of all the winners
        # Get the number of winners (may be les than the number of players
        winners = len(wins) 
        # Sort the winners sequentially based on the player #
        wins = [wins for _,wins in sorted(zip(labels,wins))]
        labels = sorted(labels) # Sort the winners
        explode = [0.1]*winners # All entries will be bumped out
        pos = wins.index(max(wins))
        explode[pos] = 0 # except for the highest number of wins
        
        # Get rainbow colors for players
        color = self._makeColors(np.asarray(range(1, winners+1),
                                             dtype=np.int64), "Paired")
        
        #If any percentage is less than 1 percent then use a legend       
        if (min(wins)/totalMatches*100) >= 1:
            _, ax = plt.subplots(figsize=[9, 9])

            ax.set_prop_cycle("color", color)
            
            ax.pie(x=wins, labels=labels, explode=explode, shadow=True,
                   autopct='%1.1f%%', labeldistance=1.05
            )
   
            plt.ylabel("Players")
        else:
            labels = ["".join(["Player: ",str(i),
                 "".join([" (",str(round(wins[pos]/totalMatches*100,1)),"%)"])])
                 for pos, i in enumerate(labels)]

            _, ax = plt.subplots(figsize=[9, 9])
            
            ax.set_prop_cycle("color", color)
            
            patches, _ = ax.pie(x=wins, explode=explode, shadow=True)
            ax.legend(patches, labels, loc="best")

        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal') 
        plt.title("Percentage of wins")
        plt.ylabel("Players")
        plt.show()
        
        # We want to return player 0, since this is the player that we
        #   are giving the aces to
        return totalMatches, round(wins[0]/totalMatches*100,2)

    def getTally(self):
        """
        returns the total matches played and the highest percentage
        """
        df = self.getStats()
        totalMatches = len(df)
        vc = df["Winner"].value_counts()
        wins = vc.tolist()
        return totalMatches, round(max(wins)/totalMatches*100,2)
        
    def allGraphs(self):
        self.valueCountGraph()
        self.valueCountScatter3D()
        self.kernelDensityScatter()
        return self.pieGraph()
        
    def export_Stats(self, file):
        self.getStats().to_csv(file, index=None, header=True)


class timeStamp:
    """
    Simple time stamp clas used for tracking how long the program takes
    to execute
    """

    def __init__(self):
        self._current = time.time()
        self._total_time = time.time()
        
    def Start(self):
        self._current = time.time()
        
    def elapsed(self, _total = False):
        """
        Elapsed time from when Start was last called
        """
        TOTAL = "E"
        if _total:
            tm = time.time() - self._total_time
            TOTAL = "Total e"
        else:
            tm = time.time() - self._current
        
        offset = timedelta(seconds=tm)

        return f"{TOTAL}lapsed time {offset}"
        
    def totalElapsed(self):
        """
        Total elapsed time from when object was first instantiated
        """
        return self.elapsed(True)


#Non class functions
def getPerformance(gm, maxPlayers, matches, maxRounds):
    """
    Process get a total time stamp for execution time and generate
    a log file
    """
    tm = timeStamp()
    deckSize = len(gm.getDeck())
    col = ["Hands", f"Cards ({deckSize})", "Players", 
           "Shuffling/cheats before",    "Matches (SCB)", "Time (SCB)",
           "Shuffling/cheats after",     "Matches (SCA)", "Time (SCA)",
           "No shuffling/cheats before", "Matches (NCB)", "Time (NCB)", 
           "No shuffling/cheats after",  "Matches (NCA)", "Time (NCA)",
           "Total Elapsed Time"]
    # Do not hard code the number of columns to simplify future changes
    numCols = len(col) 
    pStats = pd.DataFrame(columns=col, dtype=np.int64)
    
    # Need to use a for loop to assign str type to multipel dataFrame columns
    for col in ["Hands","Time (SCB)","Time (SCB)", "Time (NCB)",
                "Time (NCA)", "Total Elapsed Time"]:
        pStats[col].astype(str)
    
    for i in range(2, maxPlayers + 1):
        # Chose to use a separate timer to stamp the total time for each set
        tmRound = timeStamp()
        pStat = [0]*numCols
        p = i - 1
        playerDeck = (deckSize - 4) // p
        allDecks = playerDeck * p + 4
        pStat[0], pStat[1], pStat[2] = f"4, {playerDeck} x {p}", allDecks, i
        tm.Start()
        print("***", i, "Players ***")
        print("** Shuffle player discards **")
        gm.setReshuffleWins(True)
        print("* Assign cheats before splitting *")
        gm.setApplyCheatsPreSplit(True)
        gm.playMatch(i, matches, maxRounds)
        pStat[3], pStat[4]  = gm.getTally()
        pStat[5] = tm.elapsed()
        print(pStat[5], flush = True)
        
        tm.Start()
        gm.clearStats() 
        print("* Assign cheats after splitting *")
        gm.setApplyCheatsPreSplit(False)
        gm.playMatch(i, matches, maxRounds)
        pStat[6], pStat[7] = gm.getTally()
        pStat[8] = tm.elapsed()
        print(pStat[8], flush = True)
        
        tm.Start()
        gm.clearStats()
        print("** Do not shuffle player discards **")
        gm.setReshuffleWins(False)
        print("* Assign cheats before splitting *")
        gm.setApplyCheatsPreSplit(True)
        gm.playMatch(i, matches, maxRounds)
        pStat[9], pStat[10] = gm.getTally()
        pStat[11] = tm.elapsed()
        print(pStat[11], flush = True)
        
        tm.Start()
        gm.clearStats() 
        print("* Assign cheats after splitting *")
        gm.setApplyCheatsPreSplit(False)
        gm.playMatch(i, matches, maxRounds)
        pStat[12], pStat[13] = gm.getTally()
        pStat[14] = tm.elapsed()
        print(pStat[14], flush = True)
        # get the total elapsed time for the round
        pStat[15] = "".join(["Round ",tmRound.totalElapsed()])
        pStats.loc[len(pStats.index) + 1] = pStat
        print(pStat[15], flush = True)
    
    pStat = [0]*numCols
    pStat[15] = tm.totalElapsed()
    pStats.loc[len(pStats.index) + 1] = pStat
    print(pStat[15], flush=True)
    pStats.to_csv("War_Stats_Total.csv", index=None, header=True)
    
    # For testing purposes this code will print the entire pStats dataFrame 
    #   to the screen
    # pd.options.display.max_columns = 30
    # print(pStats)

if __name__ == "__main__":
    # The primary deck of cards to be used have defined Ace - 2
    cards = {"A": 14,  "K": 13, "Q": 12, "J": 11,
             "10": 10, "9": 9,  "8": 8,  "7": 7,
             "6": 6,   "5": 5,  "4": 4,  "3": 3,
             "2": 2}

    # All the suits in our deck
    suits = ["Heart", "Club", "Diamond", "Spade"]

    # all the unsuited cards for example the two jokers and the rules card
    # We never played with them so I just set this to none
    unsuited = None # {'Joker_1': 15, 'Joker_2': 15, 'Super': 16}

    # build our deck and assign the special rules for analysis
    games = War_card_game(cards, suits, unsuited)
    # Always assign the aces to placyer 1
    games.addCheats(["A-Heart", "A-Spade", "A-Diamond", "A-Club"]) 
    games.setPlayerHand(4, 1) # Limit player one to four cards

    maxPlayers = 11 # maximum players to run analysis for
    matches = 1000 # total number of matches to play per test
    # had to set maxRounds for games where we are not shuffling player hands
    #   from win pile
    maxRounds = 5000
    getPerformance(games, maxPlayers, matches, maxRounds)

    # Play for range of playes 2 and 3 and only change when applying the 
    #   specific cards to a player since these we the most interesting results

    games.setApplyCheatsPreSplit(True)
    games.setReshuffleWins(True)
    print("*** 3 Players Generating Graphs ***", flush = True)
    games.playMatch(4, 10000)
    games.export_Stats(fr"War_3_player_dataframe.csv")
    games.allGraphs() 
        
    # For testing purposes this code will print the entire dataFrame 
    #   to the screen
    # pd.options.display.max_rows = 10000
    # print(games.getStats())   
