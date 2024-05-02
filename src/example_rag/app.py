import streamlit as st

RETRIEVER_ROLE = '📔'


@st.cache_resource
def load_messenger():
    from src.example_rag.chatbot.messenger import Messenger
    from src.example_rag.chatbot.model import MiniLlama
    from src.example_rag.chatbot.retriever import get_retriever
    from src.example_rag.settings import COLLECTION_NAME, GenerationSettings

    gen_settings = GenerationSettings()

    chat_model = MiniLlama(
        max_new_tokens=gen_settings.max_new_tokens,
        temperature=gen_settings.temperature,
        top_p=gen_settings.top_p,
    )
    retriever = get_retriever(COLLECTION_NAME, gen_settings.similiarity_top_k)
    return Messenger(chat_model, retriever)


messenger = load_messenger()

st.title('Shadow of Altars Chatbot 📚')

if 'messages' not in st.session_state:
    st.session_state.messages = [
        {
            'role': 'assistant',
            'content': 'Hello! I am a chatbot that can answer questions about the book "Shadow of Altars" by Vincas Mykolaitis-Putinas. Ask me anything! 📚',
        }
    ]
    messenger.reset_context()

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        if message['role'] == RETRIEVER_ROLE:
            with st.expander('Chatbot retrieved this context:'):
                st.markdown(message['content'])
        else:
            st.markdown(message['content'])

if prompt := st.chat_input('Ask me anything!'):
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    response, display_context = messenger.chat(prompt)
    with st.chat_message(RETRIEVER_ROLE), st.expander('Chatbot retrieved context:'):
        st.markdown(display_context)
    with st.chat_message('assistant'):
        st.markdown(response)

    st.session_state.messages.append(
        {'role': RETRIEVER_ROLE, 'content': display_context}
    )
    st.session_state.messages.append({'role': 'assistant', 'content': response})
