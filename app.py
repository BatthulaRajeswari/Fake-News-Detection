import streamlit as st

st.title("Fake News Detection Model")

news = st.text_area("Enter News Article")

if st.button("Predict"):

```
fake_keywords = [
    "shocking", "secret", "viral", "click here",
    "conspiracy", "breaking", "unbelievable"
]

if any(word in news.lower() for word in fake_keywords):
    st.error("Prediction: FAKE NEWS")
else:
    st.success("Prediction: REAL NEWS")
```
