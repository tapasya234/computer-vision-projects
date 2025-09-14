import streamlit

streamlit.title("Hey hey hey!")
streamlit.header("Hello World")

img = streamlit.file_uploader("Upload a file")
if img is not None:
    streamlit.image(img)

streamlit.text("Lord of the Rings!")


selectionOption = streamlit.selectbox(
    "Select your favourite book",
    ["Fellowship of the Ring", "Two Towers", "Return of the King"],
)
streamlit.write("I like ", selectionOption, " as well!")
streamlit.balloons()
