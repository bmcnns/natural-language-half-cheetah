import random
import numpy as np
import pandas as pd
from openai import OpenAI

api_key = "<YOUR API KEY HERE>"

client = OpenAI(api_key=api_key)

# Define the ranges for x0, x1, and fitness
x0_values = np.arange(0, 1.05, 0.05)  # Steps from 0 to 1 with step size 0.05
x1_values = np.arange(0, 1.05, 0.05)
fitness_values = np.arange(0, 10001, 1000)

# Flatten the grid into a list of combinations
all_combinations = [(x0, x1, fitness) for x0 in x0_values for x1 in x1_values for fitness in fitness_values]

# Randomly sample 10,000 combinations uniformly
sampled_combinations = random.choices(all_combinations, k=10000)

# Helper function to generate prompts
def generate_prompt(x0, x1, fitness):
    # Define imaginative templates
    templates = [
        (
            f"In the HalfCheetah-v5 environment:\n"
            f"x0 = {x0:.2f} (back leg ground contact probability)\n"
            f"x1 = {x1:.2f} (front leg ground contact probability)\n"
            f"Fitness score: {fitness}\n"
            f"Describe the cheetah's movement. Is it hopping, smoothly running, or struggling to maintain balance? "
            f"Explain how x0 and x1 reflect the coordination of its legs and relate to the fitness score."
        ),
        (
            f"In the Mujoco HalfCheetah-v5 environment:\n"
            f"x0 = {x0:.2f}, indicating back leg ground contact probability.\n"
            f"x1 = {x1:.2f}, indicating front leg ground contact probability.\n"
            f"The fitness score is {fitness}. Based on these values, describe the cheetah's gait: "
            f"is it bounding forward, dragging its legs, or moving in a coordinated way? Relate this motion to its fitness score."
        ),
        (
            f"The HalfCheetah-v5 environment provides the following metrics:\n"
            f"x0 (back leg contact): {x0:.2f}\n"
            f"x1 (front leg contact): {x1:.2f}\n"
            f"Fitness score: {fitness}\n"
            f"Based on the probabilities, describe the cheetah's gait. Does it seem stable, alternating its legs efficiently, "
            f"or does it appear to favor one leg over the other? How do these patterns impact the fitness score?"
        ),
        (
            f"In the HalfCheetah-v5 environment:\n"
            f"x0 = {x0:.2f} (back leg contact probability)\n"
            f"x1 = {x1:.2f} (front leg contact probability)\n"
            f"Fitness score: {fitness}\n"
            f"Imagine the cheetah’s movement based on these values. Is it hopping awkwardly, smoothly bounding forward, "
            f"or stuck in a jerky, inefficient motion? How does this align with the reported fitness score?"
        ),
        (
            f"In the Mujoco HalfCheetah-v5 environment:\n"
            f"x0 = {x0:.2f} (back leg contact probability)\n"
            f"x1 = {x1:.2f} (front leg contact probability)\n"
            f"Fitness score: {fitness}\n"
            f"Given these probabilities, speculate on the cheetah’s gait. Is it achieving balanced motion, "
            f"or does it struggle to maintain coordination between its legs? Relate this motion to its fitness score."
        ),
    ]

    return random.choice(templates)


# Initialize the CSV file
output_file = "half_cheetah_descriptions.csv"
columns = ["x0", "x1", "fitness", "description"]
df = pd.DataFrame(columns=columns)
df.to_csv(output_file, index=False)  # Save the header initially

# Process each combination iteratively and save to CSV
for i, (x0, x1, fitness) in enumerate(sampled_combinations):
    # Generate the prompt
    prompt = generate_prompt(x0, x1, fitness)

    try:
        # Query GPT
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9
        )
        description = completion.choices[0].message.content

        # Append the result to the DataFrame
        new_row = {"x0": round(x0, 2), "x1": round(x1, 2), "fitness": fitness, "description": description}
        df = pd.DataFrame([new_row])
        df.to_csv(output_file, mode='a', index=False, header=False)  # Append to CSV
    except Exception as e:
        print(f"Error processing combination {x0}, {x1}, {fitness}: {e}")

    # Log progress
    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(sampled_combinations)} combinations...")

print("Finished processing and saving all prompts!")
