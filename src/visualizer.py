import numpy as np
import pygame

# Initialize pygame
pygame.init()

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (50, 50, 50)
GREEN = (50, 200, 50)
DARK_GREEN = (25, 125, 25)
RED = (255, 50, 50)
BLUE = (100, 150, 255)

# Global screen variable
screen = None
clock = None
CELL_SIZE = 20
PADDING = 10
HEADER_HEIGHT = 40


def init_display(num_envs=4, grid_size=10, cell_size=20):
    """Initialize pygame display
    
    Args:
        num_envs: Number of environments to display (max 4)
        grid_size: Size of the game grid (e.g., 10 for 10x10)
        cell_size: Size of each cell in pixels
    """
    global screen, clock, CELL_SIZE
    CELL_SIZE = cell_size
    
    num_envs = min(num_envs, 4)
    
    # Calculate window size for 2x2 grid
    single_game_width = grid_size * CELL_SIZE
    single_game_height = grid_size * CELL_SIZE + HEADER_HEIGHT
    
    # 2x2 grid with padding
    window_width = single_game_width * 2 + PADDING * 3
    window_height = single_game_height * 2 + PADDING * 3
    
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Snake RL Agent")
    clock = pygame.time.Clock()


def display_game_state(obs, clear_screen=True, scores=None, dones=None, body_positions=None, snake_lengths=None):
    """Display game states for multiple environments using pygame
    
    Args:
        obs: Observations tensor (num_envs, channels, height, width)
        clear_screen: Whether to clear the screen
        scores: Optional array of scores for each environment
        dones: Optional array of done flags for each environment
        body_positions: Optional tensor of body segment positions (num_envs, max_len, 2)
        snake_lengths: Optional tensor of snake lengths (num_envs,)
    """
    global screen, clock, CELL_SIZE
    
    # Handle pygame events to prevent freezing
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            return
    
    num_envs = min(obs.shape[0], 4)
    grid_size = obs.shape[2]  # Height/width of game grid
    
    # Initialize screen if not done yet (with correct grid size)
    if screen is None:
        init_display(num_envs, grid_size, CELL_SIZE)
    
    if clear_screen:
        screen.fill(BLACK)
    
    # Calculate dimensions for each game
    single_game_width = grid_size * CELL_SIZE
    single_game_height = grid_size * CELL_SIZE
    
    # Draw each environment in 2x2 grid
    for i in range(num_envs):
        row = i // 2
        col = i % 2
        
        # Calculate position
        x_offset = PADDING + col * (single_game_width + PADDING)
        y_offset = PADDING + HEADER_HEIGHT + row * (single_game_height + HEADER_HEIGHT + PADDING)
        
        # Extract channels
        body = obs[i, 0].cpu().numpy()
        head = obs[i, 1].cpu().numpy()
        food = obs[i, 2].cpu().numpy()
        walls = obs[i, 3].cpu().numpy()
        
        # Create a gradient map if we have body positions
        gradient_map = {}
        if body_positions is not None and snake_lengths is not None:
            length = int(snake_lengths[i].item())
            positions = body_positions[i, :length].cpu().numpy()
            
            # Assign gradient values: 0 (head) to length-1 (tail)
            for segment_idx, pos in enumerate(positions):
                if pos[0] >= 0 and pos[1] >= 0:  # Valid position
                    gradient_map[(int(pos[0]), int(pos[1]))] = segment_idx / max(length - 1, 1)
        
        # Draw header with score and status
        header_y = y_offset - HEADER_HEIGHT
        font = pygame.font.Font(None, 28)
        
        # Build header text
        header_text = f"Game {i}"
        if scores is not None:
            header_text += f" | Score: {int(scores[i])}"
        
        text_color = WHITE
        if dones is not None and dones[i]:
            header_text += " | DONE"
            text_color = RED
        
        text_surface = font.render(header_text, True, text_color)
        screen.blit(text_surface, (x_offset, header_y))
        
        # Draw game grid
        for y in range(grid_size):
            for x in range(grid_size):
                rect = pygame.Rect(
                    x_offset + x * CELL_SIZE,
                    y_offset + y * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE
                )
                
                # Determine color based on cell content
                if walls[y, x] > 0:
                    color = GRAY
                elif food[y, x] > 0:
                    color = RED
                elif head[y, x] > 0:
                    # Brightest green for head
                    color = (120, 255, 120)
                elif body[y, x] > 0:
                    # Use gradient based on position in body
                    if (y, x) in gradient_map:
                        t = gradient_map[(y, x)]  # 0.0 at head, 1.0 at tail
                        # Gradient from bright green to dark green
                        r = int(120 - 95 * t)  # 120 -> 25
                        g = int(255 - 155 * t)  # 255 -> 100
                        b = int(120 - 95 * t)  # 120 -> 25
                        color = (r, g, b)
                    else:
                        # Fallback if no gradient info
                        color = (50, 200, 50)
                else:
                    color = BLACK
                
                pygame.draw.rect(screen, color, rect)
                
                # Draw grid lines
                pygame.draw.rect(screen, (30, 30, 30), rect, 1)
        
        # Draw red border if game is done
        if dones is not None and dones[i]:
            border_rect = pygame.Rect(
                x_offset - 2,
                y_offset - 2,
                single_game_width + 4,
                single_game_height + 4
            )
            pygame.draw.rect(screen, RED, border_rect, 4)
    
    pygame.display.flip()
    if clock:
        clock.tick(100)  # Cap at 60 FPS


def close_display():
    """Close the pygame display"""
    pygame.quit()