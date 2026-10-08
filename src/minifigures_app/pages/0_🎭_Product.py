"""Streamlit image page."""

from __future__ import annotations

from random import choice

import streamlit as st
from PIL import Image

from minifigures_app.utils import get_image, get_similar, list_im_tags, predict_image


def main():
    """Main function."""
    st.set_page_config(page_title="Product", page_icon="🎭")

    # page header
    st.title("🎭 Product")
    st.markdown("---")

    # Check if the user navigated from the market page.
    if "selected_tag" in st.session_state:
        tag = st.session_state.pop("selected_tag")
        st.session_state.viewing_tag = tag
        im = get_image(tag)
        st.info(f"Viewing: {tag}")
    else:
        # Toggle what to show
        show = st.radio("What do you want to do?", ["Upload image", "Random example"])
        if show == "Upload image":
            im = get_upload()
        elif show == "Random example":
            im = get_random()
        else:
            st.error("No option selected.")
            return

    # Show error if no image
    if im is None:
        st.error("No image selected.")
        return

    # Show the image
    st.image(im)

    # Make prediction
    pred = predict_image(image=im)

    # Show result
    st.write("Predictions:")
    st.write(pred)

    # Show similar products
    if "viewing_tag" in st.session_state:
        st.markdown("---")
        st.subheader("Similar Products")
        similar = get_similar(st.session_state.viewing_tag, k=5)
        if similar:
            cols = st.columns(len(similar))
            for col, item in zip(cols, similar):
                sim_img = get_image(item["tag"])
                col.image(sim_img, use_container_width=True)
                col.caption(f"{item['tag']}\nSimilarity: {item['score']:.2f}")


def get_upload() -> Image.Image | None:
    """Get an image from the user."""
    # Upload a file
    uploaded_file = st.file_uploader("Upload image", ["png", "jpg"], accept_multiple_files=False)

    # Create prediction for the file
    if uploaded_file:
        # Convert to PIL
        return Image.open(uploaded_file).convert("RGB")
    return None


def get_random() -> Image.Image | None:
    """Get a random image."""
    # Get all possible image tags
    im_tags = list_im_tags()

    # Randomly select one and return it
    tag = choice(im_tags)
    st.session_state.viewing_tag = tag
    return get_image(tag)


if __name__ == "__main__":
    main()
