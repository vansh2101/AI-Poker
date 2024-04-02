from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import estimate_hole_card_win_rate, gen_cards
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CustomPokerPlayer(BasePokerPlayer):
    def __init__(self):
        self.data = {
            "playing": 0,
            "seat_position": 2,
            'stack': 0,
            "round": "",
            "pot": 0,
            "p1_action": 'x',
            "p1_stack": 0,
            "p1_bet": 0,
            "p2_action": 'x',
            "p2_stack": 0,
            "p2_bet": 0,
            "community_card": [],
            "hole_card": [],
            "combination": "High Card"
        }
        self.hole_card_separated = []
        self.com_card_separated = []

        self.players = {}

        self.SUIT_MAP = {
            'C': 0,
            'D': 1,
            'H': 2,
            'S': 3,
            'U': 4
        }

        self.RANK_MAP = {
            '2': 0,
            '3': 1,
            '4': 2,
            '5': 3,
            '6': 4,
            '7': 5,
            '8': 6,
            '9': 7,
            'T': 8,
            'J': 9,
            'Q': 10,
            'K': 11,
            'A': 12,
            'U': 13
        }

        self.ROUND_MAP = {'preflop': 0, 'flop': 1, 'turn': 1, 'river': 2}

        self.ACTION_MAP = {'fold':0, 'raise': 2, 'call': 1, 'show': 3, 'x': 4}

        self.COMBINATION_MAP = {
            'Straight Flush': 9,
            'Four of a Kind': 8,
            'Full House': 7,
            'Flush': 6,
            'Straight': 5,
            'Three of a Kind': 4,
            'Two Pair': 3,
            'One Pair': 2,
            'High Card': 1,
            'x': 0
        }


    def receive_game_start_message(self, game_info):
        input_size = 26
        hidden_size_cls = 64
        output_size_cls = 4
        regression_output_size = 1

        # Initialize the combined model
        self.model = CombinedModel(input_size, hidden_size_cls, output_size_cls, regression_output_size)
        self.model.load_state_dict(torch.load('model.pth'))
        self.model.eval()
        
        self.data["playing"] = game_info['player_num']

        ind = 0
        for i in range(self.data["playing"]):
            if game_info['seats'][i]['uuid'] == self.uuid:
                ind = i
                break
        self.players[game_info['seats'][(ind+1)%3]['uuid']] = 'p1'
        self.players[game_info['seats'][(ind+2)%3]['uuid']] = 'p2'


    def receive_round_start_message(self, round_count, hole_card, seats):
        self.data['hole_card'] = hole_card
        self.data['community_card'] = []

        self.hole_card_separated = self.separateCards(hole_card)
        self.data['combination'] = self.getCombination(self.hole_card_separated)

        for i in range(self.data["playing"]):
            if seats[i]['uuid'] != self.uuid:
                try:
                    self.data[self.players[seats[i]['uuid']] + '_stack'] = seats[i]['stack']
                except:
                    continue
            else:
                self.data['stack'] = seats[i]['stack']


    def receive_street_start_message(self, street, round_state):
        self.data["round"] = street
        self.data['p1_action'] = 'x'
        self.data['p2_action'] = 'x'
        self.data['p1_bet'] = 0
        self.data['p2_bet'] = 0


        if street == "preflop":
            for i in range(2):
                if round_state['action_histories']['preflop'][i]['uuid'] == self.uuid:
                    self.data['seat_position'] = i
                    break
                else:
                    self.data['seat_position'] = 2

        if round_state['community_card']:
            self.data['community_card'] = round_state['community_card']
            self.com_card_separated = self.separateCards(round_state['community_card'])

            self.data['combination'] = self.getCombination(self.hole_card_separated + self.com_card_separated)

    
    def declare_action(self, valid_actions, hole_card, round_state):
        self.data['pot'] = round_state['pot']['main']['amount']

        win_rate = estimate_hole_card_win_rate(
                    nb_simulation=1000,
                    nb_player=self.data['playing'],
                    hole_card=gen_cards(self.data['hole_card']),
                    community_card=gen_cards(self.data['community_card'])
                )
        
        if win_rate >= 0.5:
            rand = np.random.randint(10)

            if rand < 2:
                return valid_actions[2]['action'], valid_actions[2]['amount']['max']
        

        if win_rate < (1/self.data['playing'] + 0.05) and win_rate > 0.1:
            return valid_actions[1]['action'], valid_actions[1]['amount']
        elif win_rate <= 0.1:
            return valid_actions[0]['action'], valid_actions[0]['amount']

        input = [
            self.data['playing'],
            self.data['seat_position'],
            self.data['stack'],
            self.ROUND_MAP[self.data['round']],
            self.data['pot'],
            self.ACTION_MAP[self.data['p1_action']],
            self.data['p1_stack'],
            self.data['p1_bet'],
            self.ACTION_MAP[self.data['p2_action']],
            self.data['p2_stack'],
            self.data['p2_bet'],
            self.COMBINATION_MAP[self.data['combination']],
            self.RANK_MAP[self.hole_card_separated[1]],
            self.SUIT_MAP[self.hole_card_separated[0]],
            self.RANK_MAP[self.hole_card_separated[3]],
            self.SUIT_MAP[self.hole_card_separated[2]],
        ]

        com_cards = len(self.com_card_separated)
        for i in range(1,10,2):
            if i < com_cards:
                input.append(self.RANK_MAP[self.com_card_separated[i]])
                input.append(self.SUIT_MAP[self.com_card_separated[i-1]])
            else:
                input.append(self.RANK_MAP['U'])
                input.append(self.SUIT_MAP['U'])

        input = torch.tensor(input, dtype=torch.float32).unsqueeze(0)
        predictions_cls, predictions_reg = self.model(input)
        softmax_predictions_cls = F.softmax(predictions_cls, dim=1)

        action = np.argmax(softmax_predictions_cls.detach().numpy())

        if action == 2:
            amount = predictions_reg.detach().numpy()[0][0]

            maximum = valid_actions[2]['amount']['max']
            minimum = valid_actions[2]['amount']['min']

            if maximum < 0:
                return valid_actions[1]['action'], valid_actions[1]['amount']

            
            if int(amount/100) == 0:
                factor = 100
            else:
                factor = 1000

            amount = int((amount/factor)*(maximum-minimum)) + minimum
        else:
            amount = valid_actions[action]['amount']
            
        action = valid_actions[action]['action']

        return action, amount


    def receive_game_update_message(self, action, round_state):
        try:
            if action['player_uuid'] != self.uuid:
                self.data[self.players[action['player_uuid']] + '_action'] = action['action']
                self.data[self.players[action['player_uuid']] + '_bet'] = action['amount']
                self.data[self.players[action['player_uuid']] + '_stack'] -= action['amount']
            else:
                self.data['stack'] -= action['amount']
        except:
            pass

        self.data['pot'] += action['amount']

        
    def receive_round_result_message(self, winners, hand_info, round_state):
        # print()
        pass


    def separateCards(self, cards):
        arr = []
        for i in cards:
            arr += i.replace('', ' ').split()

        return arr
    

    def getCombination(self, cards):

        def is_straight_flush(suits, ranks):
            return is_straight(ranks) and is_flush(suits)
        
        def is_four_of_a_kind(ranks):
            for rank in ranks:
                if ranks.count(rank) == 4:
                    return True
            return False

        def is_full_house(ranks):
            return len(set(ranks)) == 2 and (ranks.count(ranks[0]) == 2 or ranks.count(ranks[0]) == 3)

        def is_flush(suits):
            return len(set(suits)) == 1 and len(suits) == 5

        def is_straight(ranks):
            return max(ranks) - min(ranks) == 4 and len(set(ranks)) == 5

        def is_three_of_a_kind(ranks):
            for rank in ranks:
                if ranks.count(rank) == 3:
                    return True
            return False

        def is_two_pair(ranks):
            return len(ranks) - len(set(ranks)) == 2

        def is_one_pair(ranks):
            return len(ranks) - len(set(ranks)) == 1
    
        suits = cards[::2]
        nums = cards[1::2]

        ranks = sorted([self.RANK_MAP[num] for num in nums], reverse=True)

        if is_straight_flush(suits, ranks):
            return "Straight Flush"
        
        elif is_four_of_a_kind(ranks):
            return "Four of a Kind"
        
        elif is_full_house(ranks):
            return "Full House"
        
        elif is_flush(suits):
            return "Flush"
        
        elif is_straight(ranks):
            return "Straight"
        
        elif is_three_of_a_kind(ranks):
            return "Three of a Kind"
        
        elif is_two_pair(ranks):
            return "Two Pair"
        
        elif is_one_pair(ranks):
            return "One Pair"
        
        else:
            return "High Card"

    
class CombinedModel(nn.Module):
    def __init__(self, input_size, hidden_size_cls, output_size_cls, regression_output_size):
        super(CombinedModel, self).__init__()
        # Classification branch
        self.fc1_cls = nn.Linear(input_size, hidden_size_cls)
        self.bn1_cls = nn.BatchNorm1d(hidden_size_cls)
        self.dropout1_cls = nn.Dropout(0.3)
        self.fc2_cls = nn.Linear(hidden_size_cls, hidden_size_cls)
        self.bn2_cls = nn.BatchNorm1d(hidden_size_cls)
        self.dropout2_cls = nn.Dropout(0.3)
        self.fc3_cls = nn.Linear(hidden_size_cls, hidden_size_cls)
        self.bn3_cls = nn.BatchNorm1d(hidden_size_cls)
        self.dropout3_cls = nn.Dropout(0.3)
        self.fc4_cls = nn.Linear(hidden_size_cls, output_size_cls)
        
        # Regression branch
        self.fc1_reg = nn.Linear(input_size, 32)
        self.fc2_reg = nn.Linear(32, 64)
        self.fc3_reg = nn.Linear(64, 128)
        self.fc35_reg = nn.Linear(128, 32)
        self.fc375_reg = nn.Linear(32, 16)
        self.fc4_reg = nn.Linear(16, regression_output_size)

    def forward(self, x):
        # Classification branch
        x_cls = torch.relu(self.bn1_cls(self.fc1_cls(x)))
        x_cls = self.dropout1_cls(x_cls)
        x_cls = torch.relu(self.bn2_cls(self.fc2_cls(x_cls)))
        x_cls = self.dropout2_cls(x_cls)
        x_cls = torch.relu(self.bn3_cls(self.fc3_cls(x_cls)))
        x_cls = self.dropout3_cls(x_cls)
        x_cls = self.fc4_cls(x_cls)
        
        # Regression branch
        x_reg = torch.relu(self.fc1_reg(x))
        x_reg = torch.relu(self.fc2_reg(x_reg))
        x_reg = torch.relu(self.fc3_reg(x_reg))
        x_reg = torch.relu(self.fc35_reg(x_reg))
        x_reg = torch.relu(self.fc375_reg(x_reg))
        x_reg = self.fc4_reg(x_reg)
        
        return x_cls, x_reg