import streamlit as st
import io
import sys
from contextlib import redirect_stdout # Import redirect_stdout

# --- Example function that uses print ---
def function_with_prints(n):
    """A dummy function that prints output."""
    print(f"Starting function with n={n}")
    for i in range(n):
        print(f"  Processing iteration {i+1}...")
        # Simulate some work
        # time.sleep(0.1) # Uncomment if you want to see output appear gradually
    print("Function finished.")
    return n * 2

# --- Streamlit App Section ---
st.header("Capturing Print Output")

if st.button("Run function and capture prints"):
    # 1. Create a string buffer
    string_io = io.StringIO()

    # 2. Redirect stdout to the buffer while running the function
    with redirect_stdout(string_io):
        try:
            print("--- Running Function ---")
            result = function_with_prints(3)
            print(f"Function returned: {result}")
            print("--- End of Captured Output ---")
        except Exception as e:
            # Also capture any exceptions that occur within the block
            print(f"\n!!! An error occurred: {e} !!!")
            st.error(f"An error occurred during execution: {e}") # Show error in Streamlit too

    # 3. Get the captured output from the buffer
    captured_string = string_io.getvalue()

    # 4. Display the captured output in the Streamlit app
    st.subheader("Captured Output:")
    st.code(captured_string, language='text') # Use st.code for console-like display

    # You can still display the final result separately if needed
    # st.write(f"Final result: {result}") # 'result' might not be defined if an exception occurred

