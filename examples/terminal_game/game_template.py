#!/usr/bin/env python3
"""
Terminal Game Template

This is a skeleton for a simple terminal-based adventure game.
The agents (planner and executor) will work together to implement this game.

The game should have:
1. A story/narrative
2. Player movement between locations
3. Items that can be picked up and used
4. Simple puzzles or challenges
5. Win/lose conditions

The agents should fill in the TODOs and expand the game as they see fit.
"""

import time
import random
import sys
import os

# TODO: Define game items
items = {
    # Format: "item_name": {"description": "...", "use_message": "...", "can_pickup": True/False}
}

# TODO: Define game locations
locations = {
    # Format: "location_name": {
    #     "description": "...",
    #     "connections": {"north": "location_name", "south": "location_name", ...},
    #     "items": ["item_name1", "item_name2", ...],
    #     "requires_item": None or "item_name"
    # }
}

# Game state
player = {
    "location": None,  # TODO: Set starting location
    "inventory": [],
    "game_over": False,
    "win": False
}

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_slow(text):
    """Print text with a typewriter effect."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(0.02)
    print()

def display_location():
    """Display the current location and available items/exits."""
    location = locations[player["location"]]
    
    print_slow(f"\n=== {player['location']} ===")
    print_slow(location["description"])
    
    # Display items in the location
    if location["items"]:
        print_slow("\nYou see:")
        for item in location["items"]:
            print_slow(f"  - {item}: {items[item]['description']}")
    
    # Display available exits
    print_slow("\nExits:")
    for direction, connected_location in location["connections"].items():
        print_slow(f"  - {direction}: {connected_location}")

def display_inventory():
    """Display the player's inventory."""
    if not player["inventory"]:
        print_slow("\nYour inventory is empty.")
    else:
        print_slow("\nInventory:")
        for item in player["inventory"]:
            print_slow(f"  - {item}: {items[item]['description']}")

def move(direction):
    """Move the player in the specified direction."""
    current_location = locations[player["location"]]
    
    if direction in current_location["connections"]:
        next_location_name = current_location["connections"][direction]
        next_location = locations[next_location_name]
        
        # Check if the location requires an item
        if next_location.get("requires_item") and next_location["requires_item"] not in player["inventory"]:
            required_item = next_location["requires_item"]
            print_slow(f"You need {required_item} to go there.")
            return
        
        player["location"] = next_location_name
        clear_screen()
        display_location()
    else:
        print_slow(f"You can't go {direction} from here.")

def take(item_name):
    """Attempt to pick up an item."""
    current_location = locations[player["location"]]
    
    if item_name in current_location["items"]:
        if items[item_name]["can_pickup"]:
            player["inventory"].append(item_name)
            current_location["items"].remove(item_name)
            print_slow(f"You picked up the {item_name}.")
        else:
            print_slow(f"You can't pick up the {item_name}.")
    else:
        print_slow(f"There is no {item_name} here.")

def use(item_name):
    """Use an item from the inventory."""
    if item_name in player["inventory"]:
        print_slow(items[item_name]["use_message"])
        
        # TODO: Implement item use effects
        # This could change game state, unlock locations, solve puzzles, etc.
        
    else:
        print_slow(f"You don't have {item_name} in your inventory.")

def intro():
    """Display the game introduction."""
    clear_screen()
    print_slow("=" * 60)
    # TODO: Replace with actual game title
    print_slow("                ADVENTURE GAME TITLE")
    print_slow("=" * 60)
    
    # TODO: Replace with actual game introduction
    print_slow("\nWelcome to the adventure game!")
    print_slow("Your mission is to...")
    
    print_slow("\nCommands: go [direction], take [item], use [item], inventory, look, quit")
    input("\nPress Enter to start...")
    clear_screen()

def game_loop():
    """Main game loop."""
    while not player["game_over"]:
        command = input("\n> ").lower().strip()
        
        if command == "quit":
            print_slow("Thanks for playing!")
            player["game_over"] = True
        
        elif command == "look":
            display_location()
        
        elif command == "inventory":
            display_inventory()
        
        elif command.startswith("go "):
            direction = command[3:]
            move(direction)
        
        elif command.startswith("take "):
            item_name = command[5:]
            take(item_name)
        
        elif command.startswith("use "):
            item_name = command[4:]
            use(item_name)
        
        else:
            print_slow("I don't understand that command.")
        
        # Check win/lose conditions
        # TODO: Implement win/lose conditions

def main():
    """Start the game."""
    intro()
    display_location()
    game_loop()
    
    if player["win"]:
        print_slow("\nCongratulations! You won the game!")
    
if __name__ == "__main__":
    main() 