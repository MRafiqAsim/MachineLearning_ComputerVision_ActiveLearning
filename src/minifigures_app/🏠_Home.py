"""Streamlit app."""

import streamlit as st

from minifigures_model.constants import DEFAULT_MODEL_TAG


def main():
    """Main function."""
    # main page config
    st.set_page_config(page_title="Minifigure Vision", page_icon="🏠")

    st.title("🏠 Minifigure Vision")  # type: ignore[no-untyped-call]
    st.write(
        "A catalogue that tags minifigure images with attributes and finds visually similar items."
    )
    st.caption(f"Model: {DEFAULT_MODEL_TAG}")
    st.markdown("---")
    st.markdown("### 👈 Pages")
    st.markdown(" - **🎭 Product**: Upload and view your product")
    st.markdown(" - **💰 Market**: Go to our marketplace")


if __name__ == "__main__":
    main()
