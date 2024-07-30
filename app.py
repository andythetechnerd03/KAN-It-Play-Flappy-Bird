"""
This module contains the Streamlit app for the project.
"""
import streamlit as st
from typing import List
import os

from src.agent import Agent

def get_models(path: str="runs") -> List[str]:
    """ Get the models from the specified path.

    Args:
        path (str, optional): Path to the folders that contain models. Defaults to "runs".
    
    Returns:
        List[str]: List of models by folder name.
    """
    models = []
    for folder in os.listdir(path):
        if os.path.isdir(os.path.join(path, folder)):
            models.append(folder)
    return models

def app():
    """ Streamlit app for the project.
    """
    st.title("How High KAN You Fly?")
    st.write("This is a fun experiment where a new KAN model will be used to play the classic\
             Flappy Bird game using Deep Q-Learning.")
    models = get_models()
    model = st.selectbox("Select a model", list(models.keys()))
    st.write(f"Selected model: {model}")
    
    # Run the model
    if st.button("Run model"):
        agent = Agent("flappybird", device="cpu")
        agent.load_model(model)
        agent.test(num_episodes=1, render=True, print_score=True)
        load_video(model)

def load_video(model_name: str):
    """ Load the video of the model playing the game.

    Args:
        model_name (str): Name of the model.
    """
    video_url = f"videos/{model_name}.mp4"
    st.video(video_url)

if __name__ == "__main__":

    models = get_models()
    print(models)
    