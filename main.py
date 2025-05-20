import streamlit as st
from keyword_search import keyword_search
from semantic_search import semantic_search
from hybrid_search import hybrid_search

st.set_page_config(page_title="Hotel Image Search", layout="wide")
st.title("🏨 Hotel Görseli Arama Uygulaması")


if "search_history" not in st.session_state:
    st.session_state.search_history = []


query = st.text_input("🔍 Arama yapmak istediğiniz sorguyu girin:", "")
method = st.selectbox("🔧 Kullanmak istediğiniz arama yöntemi:", ["Keyword Search", "Semantic Search", "Hybrid Search"])
top_k = st.slider("📊 Kaç sonuç gösterilsin?", min_value=1, max_value=20, value=5)


if method == "Hybrid Search":
    st.markdown("### ⚖️ Ağırlıkları Ayarlayın")
    weight_keyword = st.slider("Keyword Search Ağırlığı", 0.0, 1.0, 0.5, 0.05)
    weight_semantic = 1.0 - weight_keyword
    st.markdown(f"Keyword: `{weight_keyword}`, Semantic: `{weight_semantic}`")
else:
    weight_keyword = 0.5
    weight_semantic = 0.5

if st.button("🔎 Ara") and query:
    st.info(f"Seçilen yöntem: {method}")

    if method == "Keyword Search":
        results = keyword_search(query, top_k=top_k)
    elif method == "Semantic Search":
        results = semantic_search(query, top_k=top_k)
    else:
        results = hybrid_search(query, top_k=top_k, weight_keyword=weight_keyword, weight_semantic=weight_semantic)

 
    st.session_state.search_history.append({
        "query": query,
        "method": method,
        "top_k": top_k
    })

    st.markdown("---")
    st.markdown("### 🔗 Sonuçlar")
    cols = st.columns(min(5, top_k))
    for i, (url, _) in enumerate(results):
        with cols[i % len(cols)]:
            st.markdown(f"[📷 Resme Git]({url})")
            st.image(url, use_container_width=True)


if st.session_state.search_history:
    st.markdown("---")
    st.markdown("### 🕘 Arama Geçmişi")
    for i, item in enumerate(reversed(st.session_state.search_history), 1):
        st.markdown(f"{i}. **Sorgu:** `{item['query']}` — **Yöntem:** `{item['method']}` — **Sonuç Sayısı:** `{item['top_k']}`")
