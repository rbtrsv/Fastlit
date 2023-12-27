import streamlit as st

st.set_page_config(
    page_title="Welcome to V7",
    page_icon="👋",
    menu_items={'Get Help': 'https://www.finpy.tech'}
)

st.write("# Welcome to V7! 👋")

# st.sidebar.success("Select a demo above.")
st.sidebar.button('Login')

st.markdown(
    """
    **👈 Select something from the sidebar** to see what I can do!
    """
)
