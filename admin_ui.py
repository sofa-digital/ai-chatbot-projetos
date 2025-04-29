import streamlit as st
import traceback
import time
import sofia_logic

# FAISS index diagnostic function
def diagnose_faiss_index(repository):
    """
    Diagnostic function to check FAISS index details
    """
    st.sidebar.subheader("Index Diagnostics")
    
    with st.sidebar.expander("View Diagnostics"):
        if repository._vectorstore is None:
            st.warning("FAISS index is not loaded in memory.")
            return
        
        # Check how many documents are indexed
        try:
            # Direct method to see number of vectors
            if hasattr(repository._vectorstore, 'index'):
                num_vectors = repository._vectorstore.index.ntotal
                st.info(f"Number of vectors in index: {num_vectors}")
            elif hasattr(repository._vectorstore, 'docstore'):
                num_docs = len(repository._vectorstore.docstore._dict)
                st.info(f"Number of documents in index: {num_docs}")
            else:
                st.warning("Could not determine the size of the index.")
            
            # List document metadata
            if hasattr(repository._vectorstore, 'docstore') and hasattr(repository._vectorstore.docstore, '_dict'):
                st.subheader("Indexed Documents:")
                docs_list = list(repository._vectorstore.docstore._dict.values())
                
                # Group by source
                source_counts = {}
                for doc in docs_list:
                    source = doc.metadata.get('source', 'Unknown')
                    if source in source_counts:
                        source_counts[source] += 1
                    else:
                        source_counts[source] = 1
                
                # Show count by source
                for source, count in source_counts.items():
                    st.write(f"- {source}: {count} chunks")
                
                # Show document samples
                st.subheader("Document Samples:")
                for i, doc in enumerate(docs_list[:3]):  # Just the first 3
                    st.markdown(f"**Document {i+1}:**")
                    st.markdown(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                    st.markdown(f"**Content:** {doc.page_content[:200]}...")
        except Exception as e:
            st.error(f"Error analyzing index: {str(e)}")

# Document upload section function
def upload_document_section(bucket_name):
    """
    Creates a section for uploading documents to the S3 bucket
    """
    st.sidebar.subheader("Upload Documents")
    with st.sidebar.expander("Upload New Document"):
        uploaded_file = st.file_uploader("Choose a file", type=['txt', 'md', 'csv', 'json', 'pdf'])
        
        custom_filename = st.text_input("Custom filename (optional)", 
                                        help="If not specified, the original filename will be used")
        
        if st.button("Upload Document"):
            if uploaded_file is not None:
                with st.spinner("Uploading document to S3..."):
                    # Define S3 object name
                    object_name = custom_filename.strip() if custom_filename.strip() else uploaded_file.name
                    
                    # Upload to S3
                    success = sofia_logic.upload_file_to_s3(bucket_name, uploaded_file, object_name)
                    
                    if success:
                        st.success(f"Document '{object_name}' uploaded successfully!")
                        st.info("Please reindex the documents to include the new file.")
                    else:
                        st.error("Error uploading the document. Check the logs for details.")
            else:
                st.warning("Please select a file to upload.")

    # Explore and manage S3 documents function
def explore_s3_documents(bucket_name):
    """
    Explore, view and delete content of documents stored in the S3 bucket
    """
    st.sidebar.subheader("Explore & Manage Documents")
    
    with st.sidebar.expander("View & Manage S3 Documents"):
        try:
            # Get documents from S3
            documents = sofia_logic.list_s3_documents(bucket_name)
            
            if documents:
                # Create document selector
                doc_options = [item['Key'] for item in documents]
                selected_doc = st.selectbox("Select a document:", doc_options)
                
                if selected_doc:
                    st.write(f"**Document:** {selected_doc}")
                    
                    # Button to view content
                    if st.button("View Content"):
                        with st.spinner("Loading content..."):
                            content = sofia_logic.view_s3_file_content(bucket_name, selected_doc)
                            st.text_area("Document Content:", value=content, height=300)
                    
                    # Separate section for deletion with direct implementation
                    st.write("---")
                    st.write("**Delete Document**")
                    with st.form(key='delete_form'):
                        st.warning("This action cannot be undone!")
                        confirm_text = st.text_input("Type 'DELETE' to confirm:", key="delete_confirm")
                        delete_submit = st.form_submit_button("Delete File")

                    if delete_submit and confirm_text == "DELETE":
                        try:
                            result = sofia_logic.delete_file_direct(bucket_name, selected_doc)
                            if result:
                                st.success(f"File '{selected_doc}' has been deleted!")
                                st.info("Please reindex the documents to update the search index.")
                                time.sleep(2)  # Give a moment to read the success message
                                st.experimental_rerun()  # Force a complete rerun to refresh the list
                            else:
                                st.error("Failed to delete the file. See logs for details.")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                            st.code(traceback.format_exc())
            else:
                st.warning("No documents found in the bucket.")
        except Exception as e:
            st.error(f"Error listing documents: {str(e)}")

# Add memory cleanup button to sidebar
def add_memory_cleanup_button():
    """Add a memory cleanup button to the admin sidebar"""
    st.sidebar.subheader("Memory Maintenance")
    
    if st.sidebar.button("Clear Memory"):
        with st.sidebar:
            success = sofia_logic.cleanup_memory()
            if success:
                st.success("Memory cleared successfully. Reloading application...")
                time.sleep(1)
                from app import restart_application
                restart_application()
            else:
                st.error("Failed to clear memory. Try restarting the application manually.")

# Add reindexing section
def add_reindexing_section(bucket_name):
    """
    Add section for forced reindexing
    """
    st.sidebar.subheader("Force Reindexing")
    if st.sidebar.button("Force Complete Reindexing"):
        with st.sidebar:
            with st.spinner("Reindexing documents..."):
                try:
                    st.info("1. Clearing memory and existing indices...")
                    
                    # Force reindexing
                    new_repository, vectorstore, elapsed_time, doc_count = sofia_logic.force_reindex(bucket_name)
                    
                    # Update session state with new repository
                    if new_repository:
                        st.session_state.repository = new_repository
                        
                        # Show success message
                        st.success(f"Found {doc_count} documents in S3 bucket.")
                        st.success(f"FAISS index created successfully in {elapsed_time:.2f} seconds.")
                        
                        # Button to restart
                        if st.button("Restart Application (Recommended)"):
                            from app import restart_application
                            restart_application()
                    else:
                        st.error("Failed to create FAISS index.")
                except Exception as e:
                    st.error(f"Error reindexing documents: {str(e)}")
                    st.error(f"Details: {type(e).__name__}")
                    st.code(traceback.format_exc())

# Password check for sidebar
def check_password():
    """
    Checks if the password is correct.
    Returns True if the password is correct or already authenticated.
    """
    # If already authenticated, return True
    if st.session_state.authenticated:
        return True
    
    # Otherwise, show the password form with centered design
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h4 style='text-align: center;'>Administrative Area</h4>", unsafe_allow_html=True)
        password = st.text_input("Enter password to access the sidebar:", type="password")
        
        if password:
            if password == "PROJ2025":
                st.session_state.authenticated = True
                st.success("Correct password! Loading administrative area...")
                st.rerun()
                return True
            else:
                st.error("Incorrect password!")
                return False
        else:
            return False

# Button to open admin area (discrete version at the bottom)
def show_admin_button():
    """Displays a very discrete admin button at the bottom of the page"""
    st.markdown("""
        <div class="footer-container">
            <a href="?admin=1" class="admin-link">Admin</a>
        </div>
    """, unsafe_allow_html=True)

    query_params = st.query_params
    if "admin" in query_params:
        st.session_state.show_login = True
        st.rerun()

# Render admin sidebar
def render_admin_sidebar(bucket_name="docs-intranet"):
    """
    Render the admin sidebar with all tools
    """
    with st.sidebar:
        st.title("SOFIA Chat Admin")
        
        # Display AWS status
        try:
            aws_config = sofia_logic.configure_aws()
            st.success(f"AWS configured with region: {aws_config['region']}")
        except Exception as e:
            st.error(f"Error configuring AWS: {str(e)}")
        
        repository = st.session_state.repository
        if repository:
            # Show available documents
            st.subheader("S3 Documents")
            with st.expander("View available documents"):
                documents = repository.list_documents_in_bucket()
                if documents:
                    for doc in documents:
                        st.write(f"- {doc}")
                else:
                    st.write("No documents found in bucket.")
            
            # Add exploration and management functionality
            explore_s3_documents(bucket_name)
            
            # Add upload functionality
            upload_document_section(bucket_name)
            
            # Add index diagnostics
            diagnose_faiss_index(repository)
            
            # Add reindexing section
            add_reindexing_section(bucket_name)
            
            # Add memory cleanup button
            add_memory_cleanup_button()
        
        # Button to exit admin area
        if st.button("Log Out"):
            st.session_state.authenticated = False
            st.session_state.show_login = False
            st.rerun()