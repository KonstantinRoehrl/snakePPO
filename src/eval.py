import os
import time

import torch

import config as cfg
from agent import PPOAgent, SpatialAwarePPOAgent
from environment import TorchSnakeEnv
from visualizer import display_game_state

# Evaluation settings
NUM_EVAL_ENVS = 4
SLEEP_TIME = 0.1  # Seconds to sleep between steps for visualization

def evaluate_agent(model_path, num_envs=NUM_EVAL_ENVS, num_runs=1, sleep_time=SLEEP_TIME, loop_forever=False):
    """Evaluate a trained agent
    
    Args:
        model_path: Path to saved model checkpoint
        num_envs: Number of parallel environments
        num_runs: Number of consecutive game runs (each run = all envs complete)
        sleep_time: Sleep time between steps for visualization
        loop_forever: If True, keep playing indefinitely
    """
    # Load environment and agent
    env = TorchSnakeEnv(size=cfg.GAME_SIZE, num_envs=num_envs, device=cfg.DEVICE)
    agent = SpatialAwarePPOAgent().to(cfg.DEVICE)
    
    # Load trained weights
    if os.path.exists(model_path ):
        agent.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
        print(f"✓ Loaded model from {model_path}")
    else:
        print(f"✗ Model not found at {model_path}")
        return
    
    agent.eval()  # Set to evaluation mode
    
    # Statistics tracking
    all_scores = []
    all_lengths = []
    run_count = 0
    
    print(f"\n{'='*60}")
    if loop_forever:
        print(f"Playing Snake indefinitely (Ctrl+C to stop)")
    else:
        print(f"Playing {num_runs} run(s) with {num_envs} concurrent games")
    print(f"{'='*60}\n")
    
    try:
        while loop_forever or run_count < num_runs:
            run_count += 1
            print(f"\n{'='*60}")
            print(f"Starting Run {run_count}" + ("" if not loop_forever else " (Press Ctrl+C to stop)"))
            print(f"{'='*60}\n")
            
            # Reset environment
            obs, rotated_obs = env.reset()
            
            # Track which environments are still playing
            playing = torch.ones(num_envs, dtype=torch.bool, device=cfg.DEVICE)
            step_count = 0
            
            with torch.no_grad():
                while playing.any():
                    # Display game state with scores and done status
                    display_game_state(
                        obs, 
                        clear_screen=True,
                        scores=env.score.cpu().numpy(),
                        dones=env.done.cpu().numpy()
                    )
                    
                    # Get greedy actions (no exploration during evaluation)
                    actions, _, _, _ = agent.sample_action(rotated_obs)
                    
                    # Step environment
                    next_obs, next_rotated_obs, rewards, dones = env.step(actions)
                    
                    # Update playing status
                    playing = ~env.done
                    
                    # Print info when games end
                    if dones.any():
                        for env_idx in torch.where(dones)[0]:
                            score = env.score[env_idx].item()
                            print(f"  Game {env_idx.item()} finished - Score: {score}, Steps: {step_count}")
                            all_scores.append(score)
                            all_lengths.append(step_count)
                    
                    obs = next_obs
                    rotated_obs = next_rotated_obs
                    step_count += 1
                    
                    # Sleep for visualization
                    time.sleep(sleep_time)
            
            # Run summary
            print(f"\n{'='*60}")
            print(f"Run {run_count} Complete")
            print(f"{'='*60}")
            run_scores = env.score.cpu().numpy()
            for i, score in enumerate(run_scores):
                print(f"  Game {i}: Score = {score:.0f}")
            print(f"  Average Score: {run_scores.mean():.2f}")
            print(f"  Steps Taken: {step_count}")
            print(f"{'='*60}\n")
            
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    # Print overall summary if we recorded any scores
    if all_scores:
        print(f"\n{'='*60}")
        print(f"OVERALL SUMMARY ({len(all_scores)} games)")
        print(f"{'='*60}")
        print(f"Average Score: {sum(all_scores) / len(all_scores):.2f}")
        print(f"Max Score: {max(all_scores)}")
        print(f"Min Score: {min(all_scores)}")
        print(f"Average Length: {sum(all_lengths) / len(all_lengths):.2f}")
        print(f"{'='*60}\n")
    
    return {
        'all_scores': all_scores,
        'all_lengths': all_lengths,
        'num_runs': run_count
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained Snake AI agent")
    parser.add_argument(
        '--model', 
        type=str, 
        default=os.path.join(cfg.MODEL_DIR, "SpatialAwarePPOAgent/best_model.pt"),
        help='Path to model checkpoint (default: models/best_model.pt)'
    )
    parser.add_argument(
        '--runs', 
        type=int, 
        default=1,
        help='Number of consecutive game runs (default: 1)'
    )
    parser.add_argument(
        '--envs', 
        type=int, 
        default=4,
        help='Number of parallel environments (default: 4)'
    )
    parser.add_argument(
        '--sleep', 
        type=float, 
        default=0.1,
        help='Sleep time between steps in seconds (default: 0.1)'
    )
    parser.add_argument(
        '--loop',
        action='store_true',
        help='Play games indefinitely until interrupted'
    )
    
    args = parser.parse_args()
    
    evaluate_agent(
        args.model, 
        num_envs=args.envs, 
        num_runs=args.runs,
        sleep_time=args.sleep,
        loop_forever=args.loop
    )