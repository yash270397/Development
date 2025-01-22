# PART-1 :#
# Import the libraries
import PyPDF2
import fitz  # PyMuPDF
import streamlit as st
import ollama
import pandas as pd
import re
import time  # Added for response timing
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
from concurrent.futures import ThreadPoolExecutor
#-------------------------------------------
# Function to extract text from a PDF file
# Parallel PDF text extraction
def extract_text_from_pdf_parallel(pdf_files):
    def extract_single_pdf(pdf_file):
        try:
            pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
            text = ""
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                text += page.get_text("text")
            return pdf_file.name, text
        except Exception as e:
            st.error(f"Error extracting text from {pdf_file.name}: {e}")
            return pdf_file.name, None
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(extract_single_pdf, pdf_files))
    # Convert results into a dictionary for session state
    pdf_texts = {name: text for name, text in results if text is not None}
    return pdf_texts

# PART-2 :#
#-------------------------------------------
# Function to summarize document text    
def summarize_text(doc_text, summary_type, bullet_points):
    try:
        # Start timing
        start_time = time.time()
        # Determine the prompt based on summary type and bullet points
        if summary_type == "Short":
            prompt = "Provide a concise summary of the text in no more than 2 sentences."
        elif summary_type == "Detailed":  # Detailed summary
            if bullet_points:
                prompt = (
                    "Provide a detailed summary of the text in 6-8 lines. "
                    "Use bullet points for each key point."
                )
            else:
                prompt = "Provide a detailed summary of the text in 6-8 lines."
        elif summary_type == "Tabular":
            prompt = (
                "Summarize the full pdf data in a table format with max 6-7 points."
            )
        # Using Ollama API to call the LLaMA model with streaming response
        response = ollama.chat(
            model="llama3.2:1b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a summarization expert. " + prompt
                    ),
                },
                {
                    "role": "user",
                    "content": doc_text # Truncate to avoid exceeding token limits
                },
            ],
            stream=True,  # Enable streaming
        )
        # Caching repeated tasks to avoid unnecessary recomputation
        @st.cache_data
        def get_cached_answer(question, text):
            return ask_llama_question(text, question)
        # Create placeholders for response and timer
        response_placeholder = st.empty()
        timer_placeholder = st.empty()
        response_text = ""
        # Bubble template with dynamic theming
        bubble_template = """
        <div style="
            background-color: {bg_color}; 
            color: {text_color}; 
            padding: 10px 15px; 
            border-radius: 10px; 
            box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.1); 
            margin: 10px 0;
            font-size: 16px;
            line-height: 1.6;
            font-family: Arial, sans-serif;
            overflow-wrap: break-word;  /* Allow long text to wrap */
            word-break: break-word;    /* Break long words */
            white-space: pre-wrap;     /* Preserve whitespace and enable wrapping */
            overflow: auto;            /* Add scrollbars if content overflows */
            max-width: 100%;           /* Ensure the bubble doesn’t exceed the screen width */            
            transition: background-color 0.3s ease, color 0.3s ease;">
        {content}
        </div>
        """
        # Detect user's theme preference for adaptive styling
        theme_mode = st.get_option("theme.base")  # Fetch current theme dynamically
        if theme_mode == "dark":
            bg_color = "#28c9b7"  # Dark blue for dark mode
            text_color = "#ffffff"  # White text for dark mode
        else:
            bg_color = "#95f5ea"  # Pastel blue for light mode
            text_color = "#000000"  # Black text for light mode
        # Stream the response and update dynamically
        for chunk in response:
            if "message" in chunk:
                content = chunk["message"].get("content", "")
                response_text += content             
                # Update timer dynamically
                elapsed_time = time.time() - start_time
                timer_placeholder.markdown(f"**Response Time:** {elapsed_time:.2f} seconds", unsafe_allow_html=True)
                # Format the response with adaptive colors
                styled_response = bubble_template.format(
                    bg_color=bg_color, text_color=text_color, content=response_text
                )
                response_placeholder.markdown(styled_response, unsafe_allow_html=True)
        # Final response time update
        elapsed_time = time.time() - start_time
        timer_placeholder.markdown(f"**Response Time:** {elapsed_time:.2f} seconds", unsafe_allow_html=True)
        # Return the response text along with elapsed time
        return response_text, elapsed_time
    except Exception as e:
        st.error(f"Error querying LLaMA: {e}")
        return None, None

# PART-3:#
#-------------------------------------------
# Function to interact with LLaMA 3.2 model via Ollama
def ask_llama_question(text, question):
    try:
        # Determine bot personality
        personality = st.session_state.get('selected_personality', 'Neutral')
        personality_tone = {
            "Neutral": "You are an excellent pdf-document chatbot.",
            "Formal": "You are a highly professional and formal pdf-document chatbot.",
            "Casual": "You are a friendly and casual pdf-document chatbot.",
            "Technical": "You are a highly technical and detail-oriented pdf-document chatbot."
        }.get(personality, "You are an excellent pdf-document chatbot.")
        # Start timing
        start_time = time.time()
        # Using Ollama API to call the LLaMA model with streaming response
        response = ollama.chat(
            model="llama3.2:1b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"{personality_tone} "
                        "Give accurate and concise answers only to the question that is asked. "
                        "You are able to handle data from PDFs such as key-value pairs, tabular data, graphs, numbers, calculations, etc."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Here is the combined text from all uploaded documents: {text}... "
                        f"Please answer the question: {question}"
                    ),
                },
            ],
            stream=True,  # Enable streaming
        )
        # Create placeholders for response and timer
        response_placeholder = st.empty()
        timer_placeholder = st.empty()
        response_text = ""
        # Bubble template with dynamic theming
        bubble_template = """
        <div style="
            background-color: {bg_color}; 
            color: {text_color}; 
            padding: 10px 15px; 
            border-radius: 10px; 
            box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.1); 
            margin: 10px 0;
            font-size: 16px;
            line-height: 1.6;
            font-family: Arial, sans-serif;
            overflow-wrap: break-word;  /* Allow long text to wrap */
            word-break: break-word;    /* Break long words */
            white-space: pre-wrap;     /* Preserve whitespace and enable wrapping */
            overflow: auto;            /* Add scrollbars if content overflows */
            max-width: 100%;           /* Ensure the bubble doesn’t exceed the screen width */            
            transition: background-color 0.3s ease, color 0.3s ease;">
        {content}
        </div>
        """
        # Detect user's theme preference for adaptive styling
        theme_mode = st.get_option("theme.base")  # Fetch current theme dynamically
        if theme_mode == "dark":
            bg_color = "#2454a6"  # Dark blue for dark mode
            text_color = "#ffffff"  # White text for dark mode
        else:
            bg_color = "#b0d9f5"  # Pastel blue for light mode
            text_color = "#000000"  # Black text for light mode
        # Stream the response and update dynamically
        for chunk in response:
            if "message" in chunk:
                content = chunk["message"].get("content", "")
                response_text += content             
                # Update timer dynamically
                elapsed_time = time.time() - start_time
                timer_placeholder.markdown(f"**Response Time:** {elapsed_time:.2f} seconds", unsafe_allow_html=True)
                # Format the response with adaptive colors
                styled_response = bubble_template.format(
                    bg_color=bg_color, text_color=text_color, content=response_text
                )
                response_placeholder.markdown(styled_response, unsafe_allow_html=True)
        # Final response time update
        elapsed_time = time.time() - start_time
        timer_placeholder.markdown(f"**Response Time:** {elapsed_time:.2f} seconds", unsafe_allow_html=True)
        # Return the response text along with elapsed time
        return response_text, elapsed_time
    except Exception as e:
        st.error(f"Error querying LLaMA: {e}")
        return None, None

# PART-4:#
#-------------------------------------------
# Helper function to detect tabular data and create CSV
def detect_and_save_csv(answer):
    if re.search(r'\b(comparison|comparing|compare|tabular|table)\b', answer, re.IGNORECASE):
        try:
            # Extract the table-like data from the answer
            table_pattern = re.compile(r'(\|.*\|(?:\n\|.*\|)*)')
            table_match = table_pattern.search(answer)
            if table_match:
                table_text = table_match.group(0)
                rows = []
                for line in table_text.split("\n"):
                    line = line.strip()
                    if line:
                        columns = [col.strip() for col in line.split("|")[1:-1]]
                        rows.append(columns)
                # Save CSV as a binary stream for download
                csv_data = pd.DataFrame(rows[1:], columns=rows[0]).to_csv(index=False)
                return csv_data
            else:
                return None
        except Exception as e:
            st.error(f"Error creating CSV: {e}")
    return None

# PART-5:#
#-------------------------------------------
# Streamlit interface
def main():
    st.title("Chat with PDF Documents 🗂️")
    # Clear button moved to the top
    def clear_all():
        st.session_state.uploaded_files = []
        st.session_state.chat_history = []
        st.session_state.pdf_texts = {}
        st.session_state.clear_flag = True  # Mark that the chat was cleared
    st.button("Clear All", on_click=clear_all)
    # Initialize session state for uploaded files, chat history, and PDF texts if not present
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'pdf_texts' not in st.session_state:
        st.session_state.pdf_texts = {}
    if 'clear_flag' not in st.session_state:
        st.session_state.clear_flag = False  # Flag to track if chat was cleared
    if 'selected_personality' not in st.session_state:
        st.session_state.selected_personality = "Neutral"  # Default personality
    # Allow multiple PDF uploads
    st.subheader("Upload PDFs 🖨️")
    uploaded_files = st.file_uploader("Click on Browse Files to choose the PDFs to be uploaded :",type="pdf", accept_multiple_files=True)
    if uploaded_files:
        total_files = len(uploaded_files)  # Total number of files
        progress_bar = st.progress(0)  # Initialize the progress bar
        # Process PDFs in parallel
        uploaded_file_names = [file.name for file in uploaded_files]
        new_files = [file for file in uploaded_files if file.name not in st.session_state.pdf_texts]
        if new_files:
            pdf_texts = extract_text_from_pdf_parallel(new_files) 
            # Update session state
            st.session_state.pdf_texts.update(pdf_texts)
            st.session_state.uploaded_files.extend(uploaded_file_names)   
            progress_bar.progress(1.0)  # Set progress to 100%
            st.success("All PDFs have been processed. ✅")
    # Combine all extracted text from PDFs
    combined_text = " ".join(st.session_state.pdf_texts.values())
    # Function to get bot personality icon
    def get_personality_icon(personality):
        icons = {
            "Neutral": "💬",  # Default robot icon
            "Formal": "📝",   # Scroll icon for formal tone
            "Casual": "🧑‍💼",   # Smile face for casual tone
            "Technical": "⚙️",  # Gear icon for technical tone
        }
        return icons.get(personality, "💬")  # Default to "Neutral" if undefined
    # Only show the below sections if documents are uploaded
    if st.session_state.pdf_texts:
        # Chat History
        st.subheader("Conversation History ⏳")
        with st.expander("Traceback", expanded=True):
            # Export conversation as PDF
            # Export conversation as PDF with word wrapping
            def export_chat_history_as_pdf(chat_history):
                buffer = BytesIO()
                c = canvas.Canvas(buffer, pagesize=letter)
                width, height = letter
                y = height - 40  # Start at the top of the page, leaving some margin
                margin = 40
                line_width = width - 2 * margin  # Adjust line width for margins
                c.setFont("Helvetica", 12)
                c.drawString(margin, y, "PDF Chatbot - Conversation History")
                y -= 30
                for chat in chat_history:
                    role = "User" if chat["role"] == "user" else "Bot"
                    content = chat["content"]
                    response_time = chat.get("response_time", None)  
                    if response_time is not None:
                        content += f" (Response Time: {response_time:.2f} seconds)"     
                    # Split content into multiple lines based on page width
                    lines = simpleSplit(content, "Helvetica", 12, line_width)       
                    c.drawString(margin, y, f"{role}:")
                    y -= 20
                    for line in lines:
                        c.drawString(margin + 20, y, line)  # Indent for content lines
                        y -= 20
                        if y < 40:  # Check if we need a new page
                            c.showPage()
                            y = height - 40
                            c.setFont("Helvetica", 12)
                c.save()
                buffer.seek(0)
                return buffer
            if st.button("Export Chat"):
                if st.session_state.chat_history:
                    pdf_buffer = export_chat_history_as_pdf(st.session_state.chat_history)
                    st.download_button(
                        label="Download Chat History as PDF",
                        data=pdf_buffer,
                        file_name="chat_history.pdf",
                        mime="application/pdf"
                    )
                else:
                    st.info("No chat history to export.")
            if st.session_state.chat_history:
                for chat in st.session_state.chat_history:
                    if chat["role"] == "user":
                        # User bubble with icon 
                        st.markdown(
                            f"""
                            <div style="
                                display: flex; 
                                align-items: flex-start; 
                                margin: 10px 0; 
                                justify-content: flex-start;">
                                <div style="
                                    margin-right: 10px; 
                                    width: 35px; 
                                    height: 35px; 
                                    background-color: #8383eb; 
                                    display: flex; 
                                    align-items: center; 
                                    justify-content: center; 
                                    border-radius: 50%;">
                                    <div class="icon">👤</div>
                                </div>
                                <div style="
                                    background-color: #f7f7d2; 
                                    color: #444; 
                                    padding: 10px 15px; 
                                    border-radius: 10px; 
                                    max-width: 80%;
                                    font-size: 14px;
                                    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);">
                                    {chat['content']}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    elif chat["role"] == "bot":
                        # Get the personality icon based on the saved personality
                        personality_icon = get_personality_icon(chat.get("personality", "Neutral"))  # Use saved personality here
                        response_time = chat.get("response_time", "")
                        # Bot bubble with icon
                        st.markdown(
                            f"""
                            <div style="
                                display: flex; 
                                align-items: flex-start; 
                                margin: 10px 0; 
                                justify-content: flex-start;">
                                <div style="
                                    margin-right: 10px; 
                                    width: 35px; 
                                    height: 35px; 
                                    background-color: #20c997; 
                                    display: flex; 
                                    align-items: center; 
                                    justify-content: center; 
                                    border-radius: 50%;">
                                    <div class="icon">🤖</div>
                                </div>
                                <div style="
                                    background-color: #f7f9fc; 
                                    color: #444; 
                                    padding: 10px 15px; 
                                    border-radius: 10px; 
                                    max-width: 80%;
                                    font-size: 14px;
                                    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);">
                                    {chat['content']}<br><span style="font-size: 12px; color: #888;">(Response Time: {response_time:.2f} seconds)</span>
                                </div>
                                <div class="icon">{personality_icon}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    elif chat["role"] == "summary":
                        st.markdown(
                            f"""
                            <div style="
                                display: flex; 
                                align-items: flex-start; 
                                margin: 10px 0; 
                                justify-content: flex-start;">
                                <div style="
                                    margin-right: 10px; 
                                    width: 35px; 
                                    height: 35px; 
                                    background-color: #f4b400; 
                                    display: flex; 
                                    align-items: center; 
                                    justify-content: center; 
                                    border-radius: 50%;">
                                    <div class="icon">📄</div>
                                </div>
                                <div style="
                                    background-color: #f7f9fc; 
                                    color: #444; 
                                    padding: 10px 15px; 
                                    border-radius: 10px; 
                                    max-width: 80%;
                                    font-size: 14px;
                                    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);">
                                    {chat['content']}<br><span style="font-size: 12px; color: #888;">(Response Time: {chat['response_time']:.2f} seconds)</span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
            else:
                st.markdown(
                    """
                    <div style="text-align: center; font-style: italic; color: #888;">
                        No chat history yet.
                    </div>
                    """, unsafe_allow_html=True
                )
        # --- Inserted Search Section ---
        st.subheader("Search in Documents 🔍")
        def search_text_in_pdf(keyword):
            results = {}
            for doc_name, text in st.session_state.pdf_texts.items():
                matches = [line for line in text.splitlines() if keyword.lower() in line.lower()]
                if matches:
                    results[doc_name] = matches
            return results
        search_query = st.text_input("Enter keyword or phrase to search :")
        bubble_template = """
        <div style="
            background-color: {bg_color}; 
            color: {text_color}; 
            padding: 10px 15px; 
            border-radius: 10px; 
            box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.1); 
            margin: 10px 0;
            font-size: 16px;
            line-height: 1.6;
            font-family: Arial, sans-serif;
            overflow-wrap: break-word;
            word-break: break-word;
            white-space: pre-wrap;
            overflow: auto;
            max-width: 100%;
            transition: background-color 0.3s ease, color 0.3s ease;">
        {content}
        </div>
        """
        # Detect user's theme preference for adaptive styling
        theme_mode = st.get_option("theme.base")
        if theme_mode == "dark":
            bg_color = "#9874f2"  # Dark blue for dark mode
            text_color = "#ffffff"  # White text for dark mode
        else:
            bg_color = "#d3c2fc"  # Pastel blue for light mode
            text_color = "#000000"  # Black text for light mode
        if search_query:
            search_results = search_text_in_pdf(search_query)
            if search_results:
                formatted_results = ", ".join(search_results)  # Combine PDF file names into a single string
                content = f"**Search Query:** {search_query}<br>**Results in:** {formatted_results}"
                styled_response = bubble_template.format(
                    bg_color=bg_color, text_color=text_color, content=content
                )
                st.markdown(styled_response, unsafe_allow_html=True)
            else:
                st.info("No matching results found.")
        # Summarize dropdown functionality
        if st.session_state.pdf_texts:
            st.subheader("Summarize Documents 📜")
            selected_doc = st.selectbox("Select a document to summarize :", options=list(st.session_state.pdf_texts.keys()))
            summary_type = st.radio("Select Summary Type :", ["Short", "Detailed","Tabular"])
            # Conditional display of "Enable Bullet Points" checkbox
            bullet_points = False  # Default value
            if summary_type == "Detailed":
                bullet_points = st.checkbox("Enable Bullet Points")
            if st.button("Generate Summary"):
                doc_text = st.session_state.pdf_texts[selected_doc]
                summary, elapsed_time = summarize_text(doc_text, summary_type, bullet_points)
                if summary:
                    st.session_state.chat_history.append({
                        "role": "summary",
                        "content": f"Summary for {selected_doc}: {summary}",
                        "response_time": elapsed_time
                    })
        # Dropdown for bot personality
        st.subheader("Choose Bot Personality 🎭")
        st.session_state.selected_personality = st.selectbox(
            "Select the behavioural mode of the Bot for chatting with documents :",
            options=["Neutral", "Formal", "Casual", "Technical"],
            index=0  # Default to "Neutral"
        )
        # Chat History section: Includes User and Bot chat bubbles with icons
        st.subheader("Chat with Documents 💭")
        # Ask question section
        if st.session_state.pdf_texts:
            # Use text_area for multiline input to handle Shift+Enter
            question = st.text_area("Ask a question about the documents (Shift+Enter for new line) :", height=150)
            # When the user submits a question, save the selected personality with the response
            if st.button("Submit Question"):
                st.write("Getting answer from LLaMA...")
                answer, response_time = ask_llama_question(combined_text, question)
                if answer:
                    st.session_state.chat_history.append({
                        "role": "user", 
                        "content": question
                    })
                    st.session_state.chat_history.append({
                        "role": "bot", 
                        "content": answer,
                        "response_time": response_time,
                        "personality": st.session_state.selected_personality  # Save the personality here
                    })
                    # Check for tabular/comparison keywords and offer CSV download
                    csv_data = detect_and_save_csv(answer)
                    if csv_data:
                        # Provide CSV download button
                        st.download_button(
                            label="Download CSV",
                            data=csv_data,
                            file_name="comparison_data.csv",
                            mime="text/csv"
                        )
#-------------------------------------------
# Run the main app
if __name__ == "__main__":
    main()