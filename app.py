"""
This module contains the Streamlit app for the project.
"""
import streamlit as st
from typing import List
import os

from src.agent import Agent

def get_models(path: str="videos") -> List[str]:
    """ Get the models from the specified path.

    Args:
        path (str, optional): Path to the folders that contain models. Defaults to "runs".
    
    Returns:
        List[str]: List of models by folder name.
    """
    models = []
    for file in os.listdir(path):
        if file.endswith(".mp4"):
            models.append(file.split(".")[0])
    return models

def app():
    """ Streamlit app for the project.
    """
    # Make 2 columns
    st.set_page_config(layout="wide")
    col1, col2 = st.columns([0.75, 0.25])
    col1.title("How High KAN You Fly?")
    col1.write("This is a fun experiment where a new Kolmogorov-Arnold Network (KAN) model will be used to play the classic\
             Flappy Bird game using Deep Q-Learning.")
    models = get_models()
    model = col1.selectbox("Select a model", list(models))
    col1.write(f"Selected model: {model}")
    
    # Run the model
    if col1.button("Run model"):
        load_video(model, col2)

def load_video(model_name: str, col: st.columns):
    """ Load the video of the model playing the game.

    Args:
        model_name (str): Name of the model.
    """
    video_url = f"videos/{model_name}.mp4"
    col.video(video_url)


if __name__ == "__main__":
    app()
    